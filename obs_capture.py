"""Captura independente de frames a partir de uma fonte de vídeo do OBS.

A captura lê a saída de vídeo do OBS (OBS Virtual Camera, um stream local
SRT/UDP/RTMP ou um arquivo gravado) e nunca toca na janela do jogo: não faz
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


def grabber_from_config(cfg: dict) -> FrameGrabber:
    c = cfg["capture"]
    source = OBSVideoSource(c["source"], c.get("width"), c.get("height"))
    return FrameGrabber(source, buffer_size=c["buffer_size"],
                        disconnect_timeout=c["disconnect_timeout"],
                        freeze_seconds=c["freeze_seconds"],
                        freeze_threshold=c["freeze_threshold"],
                        reconnect_delay=c["reconnect_delay"])
