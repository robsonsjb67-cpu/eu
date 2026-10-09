"""Leitor da Battle List.

Lê apenas a imagem da região selecionada da Battle List (vinda do OBS) e
produz um estado observado: linhas, nomes (OCR opcional), percentual de vida
e entradas/saídas. Este módulo não envia nenhum comando ao jogo.

Estratégia de detecção:
  1. Cada criatura da Battle List tem uma barra de vida colorida (verde,
     amarela, vermelha...). As barras são encontradas como componentes
     conectados finos e largos numa máscara de cor saturada.
  2. Cada barra define uma linha; o nome fica logo acima da barra e o ícone
     da criatura à esquerda.
  3. Um rastreador associa as linhas entre frames, atribuindo um ID
     temporário a cada entrada, para não contar o mesmo monstro duas vezes.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import cv2
import numpy as np

from vision_common import Region, match_known_name, normalize_name

log = logging.getLogger(__name__)

try:  # OCR é opcional
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - depende do ambiente
    pytesseract = None


# --------------------------------------------------------------------------
# Detecção por frame
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BattleRow:
    """Uma linha da Battle List observada em um único frame."""

    row_index: int
    bar_box: tuple[int, int, int, int]       # (x, y, w, h) relativo à região
    hp_percent: float                         # 0–100
    bar_color: str                            # green / yellow / red / ...
    name: Optional[str] = None                # nome normalizado (ou None)
    raw_name: Optional[str] = None            # texto bruto do OCR
    highlighted: Optional[str] = None         # moldura colorida no ícone (ex.: alvo)
    icon_signature: Optional[np.ndarray] = field(default=None, compare=False, repr=False)

    @property
    def y_center(self) -> float:
        return self.bar_box[1] + self.bar_box[3] / 2


# Cores aproximadas (HSV OpenCV: H 0–179) das barras de vida do Tibia.
_HUE_NAMES = (
    (0, 10, "red"),
    (10, 22, "orange"),
    (22, 38, "yellow"),
    (38, 90, "green"),
    (170, 180, "red"),
)


def _classify_hue(h: float) -> str:
    for lo, hi, name in _HUE_NAMES:
        if lo <= h < hi:
            return name
    return "other"


class BattleListReader:
    """Extrai linhas da Battle List de um frame."""

    def __init__(self, region: Optional[Region] = None, bar_full_width: Optional[int] = None,
                 min_bar_width: int = 1, bar_min_height: int = 2, bar_max_height: int = 6,
                 name_height: int = 12, ocr: bool = True, known_names: Iterable[str] = ()):
        self.region = region
        self.bar_full_width = bar_full_width
        self.min_bar_width = min_bar_width
        self.bar_min_height = bar_min_height
        self.bar_max_height = bar_max_height
        self.name_height = name_height
        self.ocr_enabled = ocr and pytesseract is not None
        self.known_names = list(known_names)
        self._learned_full_width = 0
        self._bar_left: Optional[int] = None
        self._ocr_cache: dict[bytes, Optional[str]] = {}

    @classmethod
    def from_config(cls, cfg: dict, known_names: Iterable[str] = ()) -> "BattleListReader":
        b = cfg["battle_list"]
        return cls(Region.from_any(cfg["regions"].get("battle_list")), b.get("bar_full_width"),
                   b.get("min_bar_width", 1), b.get("bar_min_height", 2), b.get("bar_max_height", 6),
                   b.get("name_height", 12), b.get("ocr", True), known_names)

    # ------------------------------------------------------------------
    def read(self, frame: np.ndarray) -> list[BattleRow]:
        """Recebe o frame completo e devolve as linhas detectadas, de cima para baixo."""
        if self.region is None:
            raise ValueError("Região da Battle List não configurada (use: main.py select-region battle_list)")
        roi = self.region.crop(frame)
        if roi.size == 0:
            return []
        return self.read_roi(roi)

    def read_roi(self, roi: np.ndarray) -> list[BattleRow]:
        bars = self._find_bars(roi)
        if not bars:
            return []
        full_w = self._full_width(bars)
        rows: list[BattleRow] = []
        for i, (x, y, w, h, hue) in enumerate(bars):
            hp = max(0.0, min(100.0, 100.0 * w / full_w)) if full_w else 100.0
            raw = self._read_name(roi, x, y) if self.ocr_enabled else None
            name = match_known_name(raw, self.known_names) if raw else None
            rows.append(BattleRow(i, (x, y, w, h), round(hp, 1), _classify_hue(hue), name, raw,
                                  self._highlight(roi, x, y, h), self._icon_signature(roi, x, y, h)))
        return rows

    # ------------------------------------------------------------------
    def _find_bars(self, roi: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat, val = hsv[..., 1], hsv[..., 2]
        # Barras: cor saturada; vermelho escuro (vida muito baixa) tem valor menor.
        mask = ((sat >= 120) & (val >= 70)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        bars = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (self.bar_min_height <= h <= self.bar_max_height):
                continue
            if w < self.min_bar_width or w < h:
                continue
            if area < 0.8 * w * h:  # barras são retângulos cheios; texto não
                continue
            hue = float(np.median(hsv[..., 0][labels == i]))
            bars.append((int(x), int(y), int(w), int(h), hue))
        if not bars:
            return []
        # Todas as barras começam na mesma coluna; descarta ruído desalinhado.
        lefts = np.array([b[0] for b in bars])
        left = int(np.median(lefts)) if self._bar_left is None else self._bar_left
        aligned = [b for b in bars if abs(b[0] - left) <= 2]
        if aligned and self._bar_left is None and len(aligned) >= 1:
            self._bar_left = int(np.median([b[0] for b in aligned]))
        return sorted(aligned, key=lambda b: b[1])

    def _full_width(self, bars) -> int:
        if self.bar_full_width:
            return int(self.bar_full_width)
        # Aprende a largura total como a maior barra vista (vida cheia é comum).
        self._learned_full_width = max(self._learned_full_width, max(b[2] for b in bars))
        return self._learned_full_width

    def _read_name(self, roi: np.ndarray, x: int, y: int) -> Optional[str]:
        y0 = max(0, y - self.name_height - 1)
        crop = roi[y0:max(y0 + 1, y - 1), x:]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        key = cv2.resize(gray, (32, 8), interpolation=cv2.INTER_AREA).tobytes()
        if key in self._ocr_cache:
            return self._ocr_cache[key]
        big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(bw) > 127:  # texto claro em fundo escuro -> inverte para o tesseract
            bw = 255 - bw
        try:
            text = pytesseract.image_to_string(bw, config="--psm 7").strip()
        except Exception:  # pragma: no cover
            log.debug("falha no OCR", exc_info=True)
            text = ""
        result = text if len(normalize_name(text) or "") >= 2 else None
        if len(self._ocr_cache) > 512:
            self._ocr_cache.clear()
        self._ocr_cache[key] = result
        return result

    def _icon_box(self, x: int, y: int, h: int) -> tuple[int, int, int, int]:
        size = self.name_height + h + 6
        x1 = max(0, x - 2)
        return max(0, x1 - size), max(0, y + h - size + 2), x1, y + h + 2

    def _highlight(self, roi: np.ndarray, x: int, y: int, h: int) -> Optional[str]:
        x0, y0, x1, y1 = self._icon_box(x, y, h)
        icon = roi[y0:y1, x0:x1]
        if icon.shape[0] < 4 or icon.shape[1] < 4:
            return None
        border = np.concatenate([icon[0], icon[-1], icon[:, 0], icon[:, -1]]).astype(int)
        b, g, r = border[:, 0], border[:, 1], border[:, 2]
        if np.mean((r > 150) & (g < 80) & (b < 80)) > 0.6:
            return "red"
        if np.mean((r > 200) & (g > 200) & (b > 200)) > 0.6:
            return "white"
        return None

    def _icon_signature(self, roi: np.ndarray, x: int, y: int, h: int) -> Optional[np.ndarray]:
        x0, y0, x1, y1 = self._icon_box(x, y, h)
        icon = roi[y0:y1, x0:x1]
        if icon.shape[0] < 4 or icon.shape[1] < 4:
            return None
        hist = cv2.calcHist([icon], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3).flatten()
        return hist / (hist.sum() or 1)


# --------------------------------------------------------------------------
# Rastreamento entre frames
# --------------------------------------------------------------------------
@dataclass
class BattleTrack:
    """Entrada da Battle List acompanhada ao longo do tempo (ID temporário)."""

    track_id: int
    name: Optional[str]
    hp_percent: float
    first_seen: float
    last_seen: float
    row_index: int
    y_center: float
    hits: int = 1
    confirmed: bool = False
    active: bool = True
    left_at: Optional[float] = None
    highlighted: Optional[str] = None
    hp_history: list[tuple[float, float]] = field(default_factory=list)
    name_votes: dict[str, int] = field(default_factory=dict)
    icon_signature: Optional[np.ndarray] = field(default=None, repr=False)

    def vote_name(self, name: Optional[str]) -> None:
        if name:
            self.name_votes[name] = self.name_votes.get(name, 0) + 1
            self.name = max(self.name_votes.items(), key=lambda kv: kv[1])[0]


@dataclass(frozen=True)
class BattleEvent:
    kind: str           # entered / left / hp_changed
    track_id: int
    name: Optional[str]
    hp_percent: float
    timestamp: float
    previous_hp: Optional[float] = None


class BattleListTracker:
    """Associa linhas entre frames e emite eventos de entrada/saída/vida.

    * Uma linha nova só vira entrada confirmada depois de ``confirm_frames``
      observações seguidas (evita contar ruído ou OCR instável).
    * Uma entrada só é considerada "saiu" depois de sumir por
      ``leave_grace_seconds`` (evita contar o mesmo monstro de novo quando a
      linha pisca por um frame).
    * A associação usa nome, vida, assinatura do ícone e ordem relativa;
      não depende da posição absoluta, porque as linhas sobem quando uma
      criatura sai da lista.
    """

    def __init__(self, confirm_frames: int = 2, leave_grace_seconds: float = 1.0,
                 hp_change_threshold: float = 3.0, max_hp_gain: float = 25.0):
        self.confirm_frames = max(1, confirm_frames)
        self.leave_grace_seconds = leave_grace_seconds
        self.hp_change_threshold = hp_change_threshold
        self.max_hp_gain = max_hp_gain
        self.tracks: dict[int, BattleTrack] = {}
        self._ids = itertools.count(1)
        self.total_confirmed = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "BattleListTracker":
        b = cfg["battle_list"]
        return cls(b.get("confirm_frames", 2), b.get("leave_grace_seconds", 1.0))

    def active_tracks(self, confirmed_only: bool = True) -> list[BattleTrack]:
        return sorted((t for t in self.tracks.values()
                       if t.active and (t.confirmed or not confirmed_only)),
                      key=lambda t: t.row_index)

    def update(self, rows: list[BattleRow], t: float) -> list[BattleEvent]:
        events: list[BattleEvent] = []
        candidates = [tr for tr in self.tracks.values() if tr.active]
        pairs = self._associate(candidates, rows)
        matched_rows = set()
        for tr, row in pairs:
            matched_rows.add(row.row_index)
            prev_hp = tr.hp_percent
            tr.last_seen = t
            tr.row_index = row.row_index
            tr.y_center = row.y_center
            tr.hits += 1
            tr.highlighted = row.highlighted
            tr.vote_name(row.name)
            if row.icon_signature is not None:
                tr.icon_signature = row.icon_signature
            tr.hp_percent = row.hp_percent
            tr.hp_history.append((t, row.hp_percent))
            del tr.hp_history[:-200]
            if not tr.confirmed and tr.hits >= self.confirm_frames:
                tr.confirmed = True
                self.total_confirmed += 1
                events.append(BattleEvent("entered", tr.track_id, tr.name, tr.hp_percent, tr.first_seen))
            elif tr.confirmed and abs(row.hp_percent - prev_hp) >= self.hp_change_threshold:
                events.append(BattleEvent("hp_changed", tr.track_id, tr.name, row.hp_percent, t, prev_hp))

        for row in rows:
            if row.row_index in matched_rows:
                continue
            tr = BattleTrack(next(self._ids), row.name, row.hp_percent, t, t, row.row_index,
                             row.y_center, highlighted=row.highlighted,
                             hp_history=[(t, row.hp_percent)], icon_signature=row.icon_signature)
            tr.vote_name(row.name)
            self.tracks[tr.track_id] = tr
            if self.confirm_frames <= 1:
                tr.confirmed = True
                self.total_confirmed += 1
                events.append(BattleEvent("entered", tr.track_id, tr.name, tr.hp_percent, t))

        for tr in list(self.tracks.values()):
            if not tr.active or tr.last_seen == t:
                continue
            if not tr.confirmed:
                # Candidato que não se confirmou: descarta sem contar.
                if t - tr.last_seen > self.leave_grace_seconds:
                    del self.tracks[tr.track_id]
                continue
            if t - tr.last_seen > self.leave_grace_seconds:
                tr.active = False
                tr.left_at = tr.last_seen
                events.append(BattleEvent("left", tr.track_id, tr.name, tr.hp_percent, t))
        self._prune(t)
        return events

    # ------------------------------------------------------------------
    def _cost(self, tr: BattleTrack, row: BattleRow) -> float:
        if tr.name and row.name and tr.name != row.name:
            return float("inf")
        if row.hp_percent - tr.hp_percent > self.max_hp_gain:
            return float("inf")  # vida não sobe tanto entre frames
        cost = 0.0
        # Deslocar uma linha é comum (alguém acima saiu); mudança de vida pesa mais.
        cost += abs(tr.row_index - row.row_index) * 0.5
        cost += abs(tr.hp_percent - row.hp_percent) / 10.0
        if tr.icon_signature is not None and row.icon_signature is not None:
            cost += float(np.abs(tr.icon_signature - row.icon_signature).sum())  # 0..2
        if tr.name and row.name:
            cost -= 0.5
        return cost

    def _associate(self, tracks: list[BattleTrack], rows: list[BattleRow]):
        """Associação gulosa de menor custo preservando a ordem das linhas.

        A Battle List mantém a ordem relativa das criaturas que continuam
        nela; por isso descartamos pares que cruzariam um par já aceito.
        """
        options = []
        for tr in tracks:
            for row in rows:
                c = self._cost(tr, row)
                if c != float("inf"):
                    options.append((c, tr.row_index, row.row_index, tr, row))
        options.sort(key=lambda o: (o[0], o[1]))
        used_t, used_r, pairs = set(), set(), []
        for c, ti, ri, tr, row in options:
            if tr.track_id in used_t or ri in used_r:
                continue
            if any((ti - pti) * (ri - pri) < 0 for _, pti, pri in pairs):
                continue
            pairs.append((tr, ti, ri))
            used_t.add(tr.track_id)
            used_r.add(ri)
        return [(tr, next(r for r in rows if r.row_index == ri)) for tr, _, ri in pairs]

    def _prune(self, t: float, keep_seconds: float = 60.0) -> None:
        for tid in [tid for tid, tr in self.tracks.items()
                    if not tr.active and tr.left_at is not None and t - tr.left_at > keep_seconds]:
            del self.tracks[tid]
