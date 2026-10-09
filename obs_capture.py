"""Captura independente de frames a partir de uma fonte de vídeo do OBS.

A captura lê a saída de vídeo do OBS (OBS Virtual Camera, prints de uma
fonte via obs-websocket, um stream local SRT/UDP/RTMP ou um arquivo gravado) e nunca toca na janela do jogo: não faz
screenshot da janela, não muda o foco e não a traz para o primeiro plano.

Uma thread dedicada lê continuamente a fonte e mantém apenas os frames mais
recentes numa fila de tamanho fixo, de modo que o consumidor sempre processa
o frame mais novo em vez de acumular atraso.
"""
from __future__ import annotations

import enum
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Union

import numpy as np

log = logging.getLogger(__name__)


class CaptureStatus(enum.Enum):
    STARTING = "starting"
    CONNECTED = "connected"
    FROZEN = "frozen"
    DISCONNECTED = "disconnected"
    STOPPED = "stopped"


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    timestamp: float  # time.monotonic() do momento da leitura
    index: int


class FrameSource(Protocol):
    def open(self) -> bool: ...
    def read(self) -> tuple[bool, Optional[np.ndarray]]: ...
    def close(self) -> None: ...


class OBSVideoSource:
    """Fonte baseada em cv2.VideoCapture.

    ``source`` pode ser:
      * um inteiro: índice da câmera virtual do OBS ("OBS Virtual Camera");
      * uma string: URL de stream (ex.: ``srt://127.0.0.1:9000``) ou arquivo.
    """

    def __init__(self, source: Union[int, str] = 0, width: Optional[int] = None,
                 height: Optional[int] = None, backend: Optional[int] = None):
        self.source = source
        self.width = width
        self.height = height
        self.backend = backend
        self._cap = None

    def open(self) -> bool:
        import cv2

        self.close()
        src = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        cap = cv2.VideoCapture(src, self.backend) if self.backend is not None else cv2.VideoCapture(src)
        if not cap.isOpened():
            cap.release()
            return False
        if isinstance(src, int):
            if self.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Buffer interno mínimo: preferimos descartar a acumular atraso.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ok, img = self._cap.read()
        return (bool(ok) and img is not None), img

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


@dataclass
class CaptureStats:
    frames_read: int = 0
    frames_dropped: int = 0
    reconnects: int = 0
    last_frame_at: Optional[float] = None
    status_since: float = field(default_factory=time.monotonic)


class FrameGrabber:
    """Lê frames em segundo plano e detecta desconexão e congelamento.

    * Desconexão: a fonte falha ao abrir/ler ou nenhum frame chega em
      ``disconnect_timeout`` segundos. A thread tenta reconectar sozinha.
    * Congelamento: os frames continuam chegando, mas a imagem praticamente
      não muda por ``freeze_seconds`` (ex.: OBS com a fonte travada). A
      diferença é medida numa miniatura em tons de cinza; ``freeze_threshold``
      é a diferença média absoluta (0–255) abaixo da qual o frame é "igual".
    """

    def __init__(self, source: FrameSource, buffer_size: int = 5,
                 disconnect_timeout: float = 2.0, freeze_seconds: float = 3.0,
                 freeze_threshold: float = 0.4, reconnect_delay: float = 1.0,
                 on_status_change: Optional[Callable[[CaptureStatus, CaptureStatus], None]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.source = source
        self.buffer: deque[Frame] = deque(maxlen=max(1, buffer_size))
        self.disconnect_timeout = disconnect_timeout
        self.freeze_seconds = freeze_seconds
        self.freeze_threshold = freeze_threshold
        self.reconnect_delay = reconnect_delay
        self.on_status_change = on_status_change
        self.clock = clock
        self.stats = CaptureStats(status_since=clock())

        self._status = CaptureStatus.STARTING
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._index = 0
        self._last_thumb: Optional[np.ndarray] = None
        self._last_change_at: Optional[float] = None

    # ------------------------------------------------------------------ API
    @property
    def status(self) -> CaptureStatus:
        with self._lock:
            return self._status

    def start(self) -> "FrameGrabber":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="obs-frame-grabber", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        with self._lock:
            self._new_frame.notify_all()
        if self._thread:
            self._thread.join(timeout)
        self.source.close()
        self._set_status(CaptureStatus.STOPPED)

    def latest(self) -> Optional[Frame]:
        """Frame mais recente (ou None). Não remove da fila."""
        self.check_timeout()
        with self._lock:
            return self.buffer[-1] if self.buffer else None

    def recent(self) -> list[Frame]:
        with self._lock:
            return list(self.buffer)

    def wait_for_frame(self, after_index: int = -1, timeout: float = 1.0) -> Optional[Frame]:
        """Bloqueia até chegar um frame com índice > ``after_index``."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while not self._stop.is_set():
                if self.buffer and self.buffer[-1].index > after_index:
                    return self.buffer[-1]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._new_frame.wait(remaining)
        self.check_timeout()
        return None

    def is_healthy(self) -> bool:
        return self.status == CaptureStatus.CONNECTED

    def check_timeout(self) -> None:
        """Marca a captura como desconectada se os frames pararam de chegar."""
        last = self.stats.last_frame_at
        if last is not None and self.status in (CaptureStatus.CONNECTED, CaptureStatus.FROZEN):
            if self.clock() - last > self.disconnect_timeout:
                self._set_status(CaptureStatus.DISCONNECTED)

    # ------------------------------------------------------------ internals
    def _set_status(self, new: CaptureStatus) -> None:
        with self._lock:
            old = self._status
            if old == new:
                return
            self._status = new
            self.stats.status_since = self.clock()
        log.info("captura: %s -> %s", old.value, new.value)
        if self.on_status_change:
            try:
                self.on_status_change(old, new)
            except Exception:  # pragma: no cover - callback do usuário
                log.exception("erro no callback on_status_change")

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.source.open():
                self._set_status(CaptureStatus.DISCONNECTED)
                self._stop.wait(self.reconnect_delay)
                continue
            if self.stats.frames_read:
                self.stats.reconnects += 1
            self._read_loop()
            self.source.close()
            if not self._stop.is_set():
                self._set_status(CaptureStatus.DISCONNECTED)
                self._stop.wait(self.reconnect_delay)

    def _read_loop(self) -> None:
        failures_since = None
        while not self._stop.is_set():
            ok, img = self.source.read()
            t = self.clock()
            if not ok or img is None or img.size == 0:
                failures_since = failures_since or t
                if t - failures_since > self.disconnect_timeout:
                    return  # força reconexão
                self._stop.wait(0.01)
                continue
            failures_since = None
            self.push(img, t)

    def push(self, img: np.ndarray, t: Optional[float] = None) -> Frame:
        """Insere um frame (usado pela thread e útil em testes)."""
        t = self.clock() if t is None else t
        frozen = self._update_freeze_state(img, t)
        with self._lock:
            if len(self.buffer) == self.buffer.maxlen:
                self.stats.frames_dropped += 1
            frame = Frame(img, t, self._index)
            self._index += 1
            self.buffer.append(frame)
            self.stats.frames_read += 1
            self.stats.last_frame_at = t
            self._new_frame.notify_all()
        self._set_status(CaptureStatus.FROZEN if frozen else CaptureStatus.CONNECTED)
        return frame

    def _update_freeze_state(self, img: np.ndarray, t: float) -> bool:
        import cv2

        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)
        if self._last_thumb is None or self._last_thumb.shape != thumb.shape:
            changed = True
        else:
            changed = float(np.mean(np.abs(thumb - self._last_thumb))) > self.freeze_threshold
        if changed or self._last_change_at is None:
            self._last_thumb = thumb
            self._last_change_at = t
            return False
        return (t - self._last_change_at) >= self.freeze_seconds


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".ts", ".webm")


class RecordedSource:
    """Reproduz imagens gravadas (pasta de PNG/JPG ou arquivo de vídeo) em ritmo fixo.

    Usada no modo de observação e nos testes: o restante do sistema recebe os
    frames exatamente como receberia do OBS. ``fps`` controla o ritmo; com
    ``loop=False`` a fonte termina no último quadro (a captura passa a
    "desconectada", o que é o comportamento esperado ao fim da gravação).
    """

    def __init__(self, path: str, fps: float = 10.0, loop: bool = False,
                 sleep: Callable[[float], None] = time.sleep):
        self.path = path
        self.fps = max(0.1, float(fps))
        self.loop = loop
        self.sleep = sleep
        self.files: list[str] = []
        self._pos = 0
        self._cap = None
        self._next_at = 0.0
        self.finished = False

    @staticmethod
    def is_recording(path: Union[int, str]) -> bool:
        import os

        if not isinstance(path, str) or path.isdigit() or "://" in path:
            return False
        return os.path.isdir(path) or path.lower().endswith(IMAGE_EXTENSIONS + VIDEO_EXTENSIONS)

    def open(self) -> bool:
        import os

        import cv2

        self.close()
        if self.finished and not self.loop:
            return False
        self._pos = 0
        if os.path.isdir(self.path):
            self.files = sorted(os.path.join(self.path, f) for f in os.listdir(self.path)
                                if f.lower().endswith(IMAGE_EXTENSIONS))
            return bool(self.files)
        if self.path.lower().endswith(IMAGE_EXTENSIONS):
            self.files = [self.path] if os.path.exists(self.path) else []
            return bool(self.files)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        import cv2

        now = time.monotonic()
        if self._next_at > now:
            self.sleep(self._next_at - now)
        self._next_at = max(now, self._next_at) + 1.0 / self.fps
        if self._cap is not None:
            ok, img = self._cap.read()
            if not ok and self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self._cap.read()
            if not ok:
                self.finished = True
            return (bool(ok) and img is not None), img
        if self._pos >= len(self.files):
            if not self.loop or not self.files:
                self.finished = True
                return False, None
            self._pos = 0
        img = cv2.imread(self.files[self._pos], cv2.IMREAD_COLOR)
        self._pos += 1
        return img is not None, img

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class OBSScreenshotSource:
    """Captura tirando prints de uma fonte/cena pelo obs-websocket (OBS 28+).

    Não precisa da Câmera Virtual: o OBS renderiza a fonte escolhida e devolve
    a imagem (request ``GetSourceScreenshot`` do protocolo obs-websocket v5).
    Ative em *Ferramentas → Configurações do Servidor WebSocket* no OBS.

    URL no campo de fonte: ``obsws://[senha@]host[:porta][/NomeDaFonte]``.
    Sem nome de fonte, usa a cena de programa atual. Sem senha na URL, usa
    ``capture.obs_password`` da configuração.
    """

    def __init__(self, host: str = "localhost", port: int = 4455, password: Optional[str] = None,
                 source_name: Optional[str] = None, fps: float = 10.0, image_format: str = "jpg",
                 quality: int = 90, width: Optional[int] = None, timeout: float = 3.0,
                 connect: Optional[Callable[..., object]] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.host, self.port = host, int(port)
        self.password = password or None
        self.source_name = source_name or None
        self.fps = max(0.5, float(fps))
        self.image_format = image_format
        self.quality = int(quality)
        self.width = width
        self.timeout = timeout
        self._connect = connect
        self.sleep = sleep
        self._ws = None
        self._req = 0
        self._next_at = 0.0
        self.last_error: Optional[str] = None

    @classmethod
    def from_url(cls, url: str, **kw) -> "OBSScreenshotSource":
        from urllib.parse import unquote, urlparse

        u = urlparse(url)
        if u.scheme != "obsws":
            raise ValueError(f"URL do obs-websocket inválida: {url!r}")
        password = unquote(u.username) if u.username else kw.pop("password", None)
        kw.pop("password", None)
        source = unquote(u.path.lstrip("/")) or kw.pop("source_name", None)
        kw.pop("source_name", None)
        return cls(u.hostname or "localhost", u.port or 4455, password, source, **kw)

    @staticmethod
    def is_obsws(src) -> bool:
        return isinstance(src, str) and src.lower().startswith("obsws://")

    @staticmethod
    def auth_string(password: str, salt: str, challenge: str) -> str:
        import base64
        import hashlib

        secret = base64.b64encode(hashlib.sha256((password + salt).encode()).digest()).decode()
        return base64.b64encode(hashlib.sha256((secret + challenge).encode()).digest()).decode()

    # -------------------------------------------------------------- protocolo
    def _send(self, op: int, d: dict) -> None:
        import json

        self._ws.send(json.dumps({"op": op, "d": d}))

    def _recv(self) -> dict:
        import json

        return json.loads(self._ws.recv())

    def _request(self, request_type: str, data: Optional[dict] = None) -> dict:
        self._req += 1
        rid = str(self._req)
        payload = {"requestType": request_type, "requestId": rid}
        if data:
            payload["requestData"] = data
        self._send(6, payload)
        while True:
            msg = self._recv()
            if msg.get("op") == 7 and msg["d"].get("requestId") == rid:
                status = msg["d"].get("requestStatus", {})
                if not status.get("result"):
                    raise RuntimeError(f"{request_type} falhou: {status.get('comment') or status.get('code')}")
                return msg["d"].get("responseData") or {}
            # op 5 (eventos) e respostas antigas são ignorados

    def open(self) -> bool:
        self.close()
        try:
            if self._connect is not None:
                self._ws = self._connect(f"ws://{self.host}:{self.port}", timeout=self.timeout)
            else:
                import websocket  # pacote websocket-client

                self._ws = websocket.create_connection(f"ws://{self.host}:{self.port}", timeout=self.timeout)
            hello = self._recv()
            if hello.get("op") != 0:
                raise RuntimeError("resposta inesperada do OBS (esperado Hello)")
            identify = {"rpcVersion": 1, "eventSubscriptions": 0}
            auth = hello["d"].get("authentication")
            if auth:
                if not self.password:
                    raise RuntimeError("o OBS exige senha do WebSocket (capture.obs_password)")
                identify["authentication"] = self.auth_string(self.password, auth["salt"], auth["challenge"])
            self._send(1, identify)
            msg = self._recv()
            if msg.get("op") != 2:
                raise RuntimeError("OBS recusou a identificação (senha errada?)")
            if self.source_name is None:
                data = self._request("GetCurrentProgramScene")
                self.source_name = data.get("currentProgramSceneName") or data.get("sceneName")
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("obs-websocket: %s", self.last_error)
            self.close()
            return False

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        import base64

        import cv2

        if self._ws is None:
            return False, None
        now = time.monotonic()
        if self._next_at > now:
            self.sleep(self._next_at - now)
        self._next_at = max(now, self._next_at) + 1.0 / self.fps
        data = {"sourceName": self.source_name, "imageFormat": self.image_format}
        if self.image_format in ("jpg", "jpeg", "webp"):
            data["imageCompressionQuality"] = self.quality
        if self.width:
            data["imageWidth"] = int(self.width)
        try:
            resp = self._request("GetSourceScreenshot", data)
            b64 = resp["imageData"].split(",", 1)[-1]
            buf = np.frombuffer(base64.b64decode(b64), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.debug("obs-websocket: falha no print: %s", self.last_error)
            if "falhou" not in str(exc):  # conexão caiu: força reconexão
                self.close()
            return False, None
        return img is not None, img

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


def source_from_config(cfg: dict) -> FrameSource:
    c = cfg["capture"]
    src = c["source"]
    if OBSScreenshotSource.is_obsws(src):
        return OBSScreenshotSource.from_url(src, password=c.get("obs_password"),
                                            fps=c.get("screenshot_fps", 10.0),
                                            image_format=c.get("screenshot_format", "jpg"),
                                            quality=c.get("screenshot_quality", 90))
    if RecordedSource.is_recording(src):
        return RecordedSource(src, c.get("playback_fps", 10.0), c.get("loop_playback", False))
    return OBSVideoSource(src, c.get("width"), c.get("height"))


def grabber_from_config(cfg: dict, on_status_change=None) -> FrameGrabber:
    c = cfg["capture"]
    source = source_from_config(cfg)
    # Em gravações, quadros repetidos são legítimos (cena parada): o limiar de
    # congelamento continua valendo, mas a desconexão só ocorre no fim do arquivo.
    return FrameGrabber(source, buffer_size=c["buffer_size"],
                        disconnect_timeout=c["disconnect_timeout"],
                        freeze_seconds=c["freeze_seconds"],
                        freeze_threshold=c["freeze_threshold"],
                        reconnect_delay=c["reconnect_delay"],
                        on_status_change=on_status_change)


def iter_recording(path: str, step: int = 1):
    """Itera todos os quadros de uma gravação (pasta de imagens ou vídeo) sem esperar.

    ``step`` > 1 pula quadros (útil para vídeos longos).
    """
    src = RecordedSource(path, fps=1e9, loop=False, sleep=lambda _s: None)
    if not src.open():
        raise FileNotFoundError(f"gravação não encontrada ou vazia: {path}")
    try:
        i = 0
        while True:
            ok, img = src.read()
            if not ok:
                break
            if i % max(1, step) == 0:
                yield img
            i += 1
    finally:
        src.close()


def save_screenshot(image: np.ndarray, folder: str = "prints", prefix: str = "obs") -> str:
    """Salva um frame em PNG (sem perdas) e devolve o caminho."""
    import os

    import cv2

    os.makedirs(folder, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"{prefix}_{stamp}_{int(time.time() * 1000) % 1000:03d}.png")
    if not cv2.imwrite(path, image):
        raise OSError(f"falha ao gravar {path}")
    return path
