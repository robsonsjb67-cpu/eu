"""SIO (suporte): monitoramento visual da barra de vida de um aliado.

O usuário escolhe a região da barra do aliado (ex.: na Party List) e,
opcionalmente, uma imagem de referência do nome dele para confirmar a
identidade.

O que este módulo NÃO presume:
  * que a barra na região pertence ao aliado — sem a imagem de referência
    do nome, a identidade fica "não verificada";
  * que a porcentagem lida é o HP real — ela é uma estimativa visual com
    confiança, e leituras incertas são registradas como tal.

Eventos de "aliado com HP baixo" só são emitidos com leitura válida **e**
identidade confirmada. Com identidade não verificada, o evento é
registrado como observação, sem alerta. Nenhuma ação é executada.
"""
from __future__ import annotations

import enum
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from health_monitor import BarReader
from logger import play_alert_sound, record_event
from vision_common import Region

log = logging.getLogger(__name__)


class Identity(enum.Enum):
    NOT_VERIFIED = "não verificada"      # sem referência cadastrada
    CONFIRMED = "confirmada"
    NOT_CONFIRMED = "não confirmada"     # referência não encontrada na região


@dataclass
class SioReading:
    timestamp: float
    percent: Optional[float]
    confidence: float
    valid: bool
    identity: Identity
    identity_score: Optional[float] = None
    stale: bool = False
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        hp = "sem leitura" if self.percent is None else f"≈ {self.percent:.0f}%"
        state = "válida" if self.valid else "incerta"
        return (f"barra {hp} (confiança {self.confidence:.2f}, leitura {state}); "
                f"identidade {self.identity.value}")


class SioMonitor:
    source = "sio"

    def __init__(self, reader: BarReader, region: Optional[Region] = None,
                 name_region: Optional[Region] = None, identity_template: Optional[np.ndarray] = None,
                 identity_threshold: float = 0.85, label: str = "Aliado",
                 threshold_percent: float = 60.0, hysteresis_percent: float = 5.0,
                 min_confidence: float = 0.7, stale_seconds: float = 1.5,
                 alert_sound: bool = True, alert_cooldown_seconds: float = 3.0,
                 clock: Callable[[], float] = time.monotonic,
                 sound: Callable[[], None] = play_alert_sound):
        self.reader = reader
        self.region = region
        self.name_region = name_region
        self.identity_template = identity_template
        self.identity_threshold = identity_threshold
        self.label = label
        self.threshold_percent = threshold_percent
        self.hysteresis_percent = hysteresis_percent
        self.min_confidence = min_confidence
        self.stale_seconds = stale_seconds
        self.alert_sound = alert_sound
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.clock = clock
        self.sound = sound
        self.last: Optional[SioReading] = None
        self.low = False
        self._identity: Optional[Identity] = None
        self._visible: Optional[bool] = None
        self._last_alert = -1e9
        self._alert_listeners: list[Callable[[str, str], None]] = []

    @classmethod
    def from_config(cls, cfg: dict) -> "SioMonitor":
        s = cfg["sio"]
        tpl = None
        if s.get("identity_template") and os.path.exists(s["identity_template"]):
            tpl = cv2.imread(s["identity_template"], cv2.IMREAD_COLOR)
        reader = BarReader(s["fill_hsv_ranges"], s["column_fill_ratio"], s.get("border_px", 1),
                           s.get("full_span"), s.get("left_offset", 0))
        return cls(reader, Region.from_any(cfg["regions"].get("sio")),
                   Region.from_any(cfg["regions"].get("sio_name")), tpl, s["identity_threshold"],
                   s["label"], s["threshold_percent"], s["hysteresis_percent"], s["min_confidence"],
                   s["stale_seconds"], s["alert_sound"], s["alert_cooldown_seconds"])

    def add_alert_listener(self, cb: Callable[[str, str], None]) -> None:
        self._alert_listeners.append(cb)

    def set_identity_template(self, image: Optional[np.ndarray]) -> None:
        self.identity_template = image
        self._identity = None

    # ------------------------------------------------------------------
    def check_identity(self, frame: Optional[np.ndarray]) -> tuple[Identity, Optional[float]]:
        if self.identity_template is None:
            return Identity.NOT_VERIFIED, None
        region = self.name_region or self.region
        if frame is None or region is None:
            return Identity.NOT_CONFIRMED, None
        area = region.crop(frame)
        tpl = self.identity_template
        if area.shape[0] < tpl.shape[0] or area.shape[1] < tpl.shape[1]:
            return Identity.NOT_CONFIRMED, None
        res = np.nan_to_num(cv2.matchTemplate(area, tpl, cv2.TM_CCOEFF_NORMED))
        score = float(res.max())
        return (Identity.CONFIRMED if score >= self.identity_threshold else Identity.NOT_CONFIRMED), score

    def update(self, frame: Optional[np.ndarray], t: float, capture_ok: bool = True) -> SioReading:
        if self.region is None:
            r = SioReading(t, None, 0.0, False, Identity.NOT_VERIFIED,
                           reasons=["região de referência do SIO não configurada"])
            self.last = r
            return r
        bar = self.reader.read(self.region.crop(frame) if frame is not None else None)
        identity, score = self.check_identity(frame)
        reasons = list(bar.reasons)
        valid = bar.percent is not None and bar.confidence >= self.min_confidence and capture_ok
        if bar.percent is not None and bar.confidence < self.min_confidence:
            reasons.append(f"confiança {bar.confidence:.2f} < mínimo {self.min_confidence:.2f}")
        if not capture_ok:
            reasons.append("captura congelada/desconectada")
        if identity == Identity.NOT_VERIFIED:
            reasons.append("sem imagem de referência do nome: a barra pode não ser do aliado")
        elif identity == Identity.NOT_CONFIRMED:
            reasons.append(f"nome de referência não encontrado (score {score if score is not None else 0:.2f})")
        r = SioReading(t, bar.percent, bar.confidence, valid, identity, score, False, reasons)
        self.last = r
        self._evaluate(r)
        return r

    def current(self, now: Optional[float] = None) -> Optional[SioReading]:
        if self.last is None:
            return None
        now = self.clock() if now is None else now
        if now - self.last.timestamp > self.stale_seconds and not self.last.stale:
            r = self.last
            self.last = SioReading(r.timestamp, r.percent, r.confidence, False, r.identity,
                                   r.identity_score, True,
                                   r.reasons + [f"leitura desatualizada (> {self.stale_seconds:.1f}s)"])
        return self.last

    # ------------------------------------------------------------------
    def _evaluate(self, r: SioReading) -> None:
        if r.identity != self._identity:
            if self._identity is not None or r.identity != Identity.NOT_VERIFIED:
                record_event(self.source, "identity", f"{self.label}: identidade {r.identity.value}"
                             + (f" (score {r.identity_score:.2f})" if r.identity_score is not None else ""))
            self._identity = r.identity
        visible = r.valid and r.percent is not None and r.percent > 0
        if visible != self._visible:
            if self._visible is not None:
                record_event(self.source, "bar_visible" if visible else "bar_lost",
                             f"{self.label}: barra {'legível' if visible else 'ilegível/ausente'} — "
                             + ("; ".join(r.reasons) or r.describe()))
            self._visible = visible
        if not r.valid or r.percent is None:
            return
        confirmed = r.identity == Identity.CONFIRMED
        if not self.low and r.percent < self.threshold_percent:
            self.low = True
            if confirmed:
                record_event(self.source, "ally_low",
                             f"{self.label} com HP baixo: {r.percent:.0f}% — Exura Sio recomendado",
                             logging.WARNING, percent=r.percent)
                self._alert("critical", f"{self.label} {r.percent:.0f}% — Exura Sio recomendado")
            else:
                record_event(self.source, "bar_low_unconfirmed",
                             f"barra monitorada em {r.percent:.0f}% (identidade {r.identity.value}; "
                             "sem alerta)", percent=r.percent)
        elif self.low and r.percent >= self.threshold_percent + self.hysteresis_percent:
            self.low = False
            record_event(self.source, "ally_recovered", f"{self.label}: barra em {r.percent:.0f}%")
        elif self.low and confirmed:
            self._alert("critical", f"{self.label} {r.percent:.0f}% — Exura Sio recomendado", repeat=True)

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
