"""Monitor de HP (Exura Vita): estima a vida pela barra capturada no OBS.

Leitura da barra
----------------
1. Converte a região para HSV e marca os pixels com a cor de preenchimento
   (faixas configuráveis; o padrão cobre o vermelho da barra de HP).
2. Usa só as linhas centrais da barra (bordas e brilho atrapalham) e decide,
   por coluna, se ela está preenchida.
3. A barra enche da esquerda para a direita: procura o ponto de corte que
   melhor separa "cheio" de "vazio". As colunas que contradizem esse corte e
   as colunas ambíguas reduzem a **confiança** da leitura.

Segurança
---------
* Leituras com confiança baixa, desatualizadas (frame antigo, captura
  congelada/desconectada) ou sem região são marcadas como inválidas e **não
  disparam** alertas nem a recomendação de cura.
* Este módulo não aperta teclas: ele estima o HP, registra eventos de vida
  baixa e emite alertas visuais/sonoros recomendando a cura (Exura Vita).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from logger import play_alert_sound, record_event
from vision_common import Region

log = logging.getLogger(__name__)

HSVRange = Sequence[Sequence[int]]   # ((h, s, v), (h, s, v))


@dataclass(frozen=True)
class BarReading:
    percent: Optional[float]          # 0–100, None quando não há leitura
    confidence: float                 # 0–1
    reasons: tuple[str, ...] = ()


class BarReader:
    """Estima o preenchimento de uma barra horizontal que enche da esquerda."""

    def __init__(self, fill_hsv_ranges: Sequence[HSVRange], column_fill_ratio: float = 0.5,
                 border_px: int = 1, full_span: Optional[int] = None, left_offset: int = 0):
        self.ranges = [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in fill_hsv_ranges]
        self.column_fill_ratio = column_fill_ratio
        self.border_px = max(0, border_px)
        self.full_span = full_span          # largura (colunas) com 100%; calibrável
        self.left_offset = left_offset

    def fill_mask(self, roi: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = np.zeros(roi.shape[:2], np.uint8)
        for lo, hi in self.ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        return mask

    def column_profile(self, roi: np.ndarray) -> np.ndarray:
        """Fração de pixels de preenchimento por coluna (linhas centrais)."""
        b = self.border_px
        h, w = roi.shape[:2]
        if h - 2 * b >= 3 and w - 2 * b >= 4:
            roi = roi[b:h - b, b:w - b]
        mask = self.fill_mask(roi) > 0
        hh = mask.shape[0]
        top, bot = int(hh * 0.2), max(int(hh * 0.2) + 1, int(np.ceil(hh * 0.8)))
        return mask[top:bot].mean(axis=0)

    def read(self, roi: Optional[np.ndarray]) -> BarReading:
        if roi is None or roi.size == 0 or roi.ndim != 3:
            return BarReading(None, 0.0, ("recorte vazio",))
        if roi.shape[1] < 8:
            return BarReading(None, 0.0, ("barra estreita demais (< 8 px)",))
        prof = self.column_profile(roi)[self.left_offset:]
        w = len(prof)
        filled = prof >= self.column_fill_ratio
        # Ponto de corte k que minimiza erros: vazias antes de k + cheias depois de k.
        empty_before = np.concatenate([[0], np.cumsum(~filled)])
        filled_after = np.concatenate([np.cumsum(filled[::-1])[::-1], [0]])
        errors = empty_before + filled_after
        k = int(np.argmin(errors))
        err = int(errors[k])
        ambiguous = int(np.sum((prof > 0.2) & (prof < 0.8)))
        full = self.full_span or w
        percent = float(np.clip(100.0 * k / max(1, full), 0.0, 100.0))
        conf = 1.0 - (err + 0.5 * ambiguous) / w
        reasons = []
        if err:
            reasons.append(f"{err} coluna(s) contradizem o preenchimento contínuo")
        if ambiguous > 0.1 * w:
            reasons.append(f"{ambiguous} coluna(s) ambíguas (cor misturada)")
        if not filled.any():
            # Barra totalmente vazia é indistinguível de região errada/tela de morte.
            conf = min(conf, 0.3)
            reasons.append("nenhuma cor de preenchimento na região (HP 0 ou região errada)")
        return BarReading(round(percent, 1), float(np.clip(conf, 0.0, 1.0)), tuple(reasons))

    def calibrate_full(self, roi: np.ndarray) -> int:
        """Calibra a largura de 100% com a barra cheia na tela."""
        self.full_span = None
        prof = self.column_profile(roi)
        filled = np.flatnonzero(prof >= self.column_fill_ratio)
        if len(filled) < 4:
            raise ValueError("barra não parece preenchida; calibre com o HP cheio")
        self.left_offset = int(filled[0])
        self.full_span = int(filled[-1] - filled[0] + 1)
        return self.full_span


@dataclass
class HealthReading:
    timestamp: float
    percent: Optional[float]            # suavizado
    raw_percent: Optional[float]
    confidence: float
    valid: bool
    stale: bool = False
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.percent is None:
            return "HP: sem leitura (" + "; ".join(self.reasons) + ")"
        tag = "" if self.valid else " [incerto: " + "; ".join(self.reasons) + "]"
        return f"HP ≈ {self.percent:.0f}% (confiança {self.confidence:.2f}){tag}"


class HealthMonitor:
    """Lê a barra de HP, detecta vida baixa e emite alertas."""

    source = "health"

    def __init__(self, reader: BarReader, region: Optional[Region] = None,
                 threshold_percent: float = 50.0, hysteresis_percent: float = 5.0,
                 min_confidence: float = 0.7, stale_seconds: float = 1.5, smoothing: float = 0.5,
                 alert_sound: bool = True, alert_cooldown_seconds: float = 3.0,
                 clock: Callable[[], float] = time.monotonic,
                 sound: Callable[[], None] = play_alert_sound):
        self.reader = reader
        self.region = region
        self.threshold_percent = threshold_percent
        self.hysteresis_percent = hysteresis_percent
        self.min_confidence = min_confidence
        self.stale_seconds = stale_seconds
        self.smoothing = min(1.0, max(0.0, smoothing))
        self.alert_sound = alert_sound
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.clock = clock
        self.sound = sound
        self.last: Optional[HealthReading] = None
        self.low = False
        self.low_events: list[tuple[float, float]] = []
        self._smoothed: Optional[float] = None
        self._last_alert = -1e9
        self._alert_listeners: list[Callable[[str, str], None]] = []
        self._uncertain_logged = False

    @classmethod
    def from_config(cls, cfg: dict) -> "HealthMonitor":
        h = cfg["health"]
        reader = BarReader(h["fill_hsv_ranges"], h["column_fill_ratio"], h.get("border_px", 1),
                           h.get("full_span"), h.get("left_offset", 0))
        return cls(reader, Region.from_any(cfg["regions"].get("hp_bar")), h["threshold_percent"],
                   h["hysteresis_percent"], h["min_confidence"], h["stale_seconds"], h["smoothing"],
                   h["alert_sound"], h["alert_cooldown_seconds"])

    def add_alert_listener(self, cb: Callable[[str, str], None]) -> None:
        """``cb(nível, mensagem)`` — usado pela interface para alertas visuais."""
        self._alert_listeners.append(cb)

    # ------------------------------------------------------------------
    def update(self, frame: Optional[np.ndarray], t: float, capture_ok: bool = True) -> HealthReading:
        reasons: list[str] = []
        if self.region is None:
            reading = HealthReading(t, None, None, 0.0, False, reasons=["região da barra de HP não configurada"])
            self.last = reading
            return reading
        bar = self.reader.read(self.region.crop(frame) if frame is not None else None)
        reasons += bar.reasons
        valid = bar.percent is not None and bar.confidence >= self.min_confidence
        if bar.percent is not None and bar.confidence < self.min_confidence:
            reasons.append(f"confiança {bar.confidence:.2f} < mínimo {self.min_confidence:.2f}")
        if not capture_ok:
            valid = False
            reasons.append("captura congelada/desconectada")
        if valid:
            # Quedas valem na hora (segurança); subidas são suavizadas contra ruído.
            if self._smoothed is None or bar.percent <= self._smoothed:
                self._smoothed = bar.percent
            else:
                a = self.smoothing
                self._smoothed = a * bar.percent + (1 - a) * self._smoothed
        percent = self._smoothed if valid else bar.percent
        reading = HealthReading(t, None if percent is None else round(percent, 1), bar.percent,
                                bar.confidence, valid, False, reasons)
        self.last = reading
        self._evaluate(reading)
        return reading

    def current(self, now: Optional[float] = None) -> Optional[HealthReading]:
        """Última leitura, marcada como desatualizada se for antiga demais."""
        if self.last is None:
            return None
        now = self.clock() if now is None else now
        if now - self.last.timestamp > self.stale_seconds and not self.last.stale:
            self.last = HealthReading(self.last.timestamp, self.last.percent, self.last.raw_percent,
                                      self.last.confidence, False, True,
                                      self.last.reasons + [f"leitura desatualizada (> {self.stale_seconds:.1f}s)"])
        return self.last

    @property
    def heal_recommended(self) -> bool:
        """Verdadeiro só com leitura válida, recente e abaixo do limite."""
        r = self.current()
        return bool(r and r.valid and not r.stale and r.percent is not None
                    and r.percent < self.threshold_percent)

    # ------------------------------------------------------------------
    def _evaluate(self, r: HealthReading) -> None:
        if not r.valid or r.percent is None:
            if not self._uncertain_logged and self.region is not None:
                record_event(self.source, "uncertain", "leitura de HP incerta — nenhuma decisão tomada: "
                             + "; ".join(r.reasons), logging.WARNING)
                self._uncertain_logged = True
            return
        self._uncertain_logged = False
        if not self.low and r.percent < self.threshold_percent:
            self.low = True
            self.low_events.append((r.timestamp, r.percent))
            del self.low_events[:-500]
            record_event(self.source, "hp_low",
                         f"HP baixo: {r.percent:.0f}% < {self.threshold_percent:.0f}% — usar Exura Vita",
                         logging.WARNING, percent=r.percent, confidence=r.confidence)
            self._alert("critical", f"HP {r.percent:.0f}% — Exura Vita recomendado")
        elif self.low and r.percent < self.threshold_percent:
            self._alert("critical", f"HP {r.percent:.0f}% — Exura Vita recomendado", repeat=True)
        elif self.low and r.percent >= self.threshold_percent + self.hysteresis_percent:
            self.low = False
            record_event(self.source, "hp_recovered", f"HP recuperado: {r.percent:.0f}%", percent=r.percent)
            self._alert("info", f"HP recuperado: {r.percent:.0f}%")

    def _alert(self, level: str, message: str, repeat: bool = False) -> None:
        now = self.clock()
        if repeat and now - self._last_alert < self.alert_cooldown_seconds:
            return
        self._last_alert = now
        if level == "critical" and self.alert_sound:
            try:
                self.sound()
            except Exception:
                log.exception("falha ao tocar o alerta sonoro")
        for cb in list(self._alert_listeners):
            try:
                cb(level, message)
            except Exception:
                log.exception("erro em listener de alerta")
