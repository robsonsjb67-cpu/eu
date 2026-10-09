"""Motor de análise: liga captura → módulos independentes → resultado por frame.

Cada módulo (CaveBot, Battle, HP, SIO) roda isolado: uma exceção em um
deles é registrada e não interrompe os demais. O motor não depende do Qt,
então a interface, a linha de comando e os testes usam o mesmo código.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from battle_attack import BattleListReader, BattleListTracker, BattleRow
from cave_navigation import Localization, MinimapLocalizer, ReferenceLibrary
from cavebot import CaveBot, CaveBotStatus
from config import ConfigStore
from health_monitor import HealthMonitor, HealthReading
from logger import record_event
from monster_detector import MonsterDetector, TemplateLibrary, VisualTrack, VisualTracker
from obs_capture import CaptureStatus
from sio_monitor import SioMonitor, SioReading
from spell_timers import TimerManager
from target_fusion import FusedTarget, TargetFusion

log = logging.getLogger(__name__)

MODULES = ("cavebot", "battle", "health", "sio")


@dataclass
class Snapshot:
    timestamp: float
    frame_index: int
    capture_status: CaptureStatus
    minimap: Optional[np.ndarray] = None
    localization: Optional[Localization] = None
    cavebot: Optional[CaveBotStatus] = None
    battle_rows: list[BattleRow] = field(default_factory=list)
    targets: list[FusedTarget] = field(default_factory=list)
    visual_tracks: list[VisualTrack] = field(default_factory=list)
    health: Optional[HealthReading] = None
    sio: Optional[SioReading] = None
    errors: dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0


class AnalysisEngine:
    def __init__(self, store: ConfigStore):
        self.store = store
        self.lock = threading.RLock()
        self.enabled: dict[str, bool] = {m: True for m in MODULES}
        self.halted = False                    # parada de emergência
        self._error_logged: dict[str, float] = {}
        self.timers = TimerManager.from_config(store.data)
        self.cavebot: Optional[CaveBot] = None
        # Ouvintes de alerta (nível, mensagem) religados a cada rebuild do HP/SIO.
        self.alert_listeners: list = []
        self.rebuild()

    # ------------------------------------------------------------ montagem
    def rebuild(self, which: Optional[str] = None) -> None:
        """Recria módulos a partir da configuração (após mudar regiões/limiares)."""
        cfg = self.store.data
        with self.lock:
            if which in (None, "cavebot"):
                lib = ReferenceLibrary(cfg["cavebot"]["references_dir"])
                self.localizer = MinimapLocalizer.from_config(cfg, lib)
                old = self.cavebot
                self.cavebot = CaveBot.from_config(cfg, self.localizer)
                if old is not None and old.route is not None:
                    self.cavebot.route = old.route
            if which in (None, "battle"):
                self.templates = TemplateLibrary(cfg["detector"]["templates_dir"])
                self.battle_reader = BattleListReader.from_config(cfg, known_names=self.templates.names)
                self.battle_tracker = BattleListTracker.from_config(cfg)
                self.detector = MonsterDetector.from_config(cfg, self.templates)
                self.visual_tracker = VisualTracker.from_config(cfg)
                self.fusion = TargetFusion.from_config(cfg)
            if which in (None, "health"):
                self.health = HealthMonitor.from_config(cfg)
                for cb in self.alert_listeners:
                    self.health.add_alert_listener(cb)
            if which in (None, "sio"):
                self.sio = SioMonitor.from_config(cfg)
                for cb in self.alert_listeners:
                    self.sio.add_alert_listener(cb)

    def add_alert_listener(self, cb) -> None:
        with self.lock:
            self.alert_listeners.append(cb)
            self.health.add_alert_listener(cb)
            self.sio.add_alert_listener(cb)

    def set_enabled(self, module: str, on: bool) -> None:
        with self.lock:
            self.enabled[module] = on
        record_event("engine", "module", f"módulo {module} {'ativado' if on else 'desativado'}")

    def emergency_stop(self) -> None:
        with self.lock:
            self.halted = True
            if self.cavebot:
                self.cavebot.stop()
            self.timers.stop_all()
        record_event("engine", "emergency_stop", "PARADA DE EMERGÊNCIA: análise interrompida, "
                     "CaveBot parado e temporizadores zerados", logging.CRITICAL)

    def release(self) -> None:
        with self.lock:
            self.halted = False
        record_event("engine", "released", "análise liberada após parada de emergência")

    # ------------------------------------------------------------ frame
    def process(self, image: np.ndarray, t: float, frame_index: int = 0,
                capture_status: CaptureStatus = CaptureStatus.CONNECTED) -> Snapshot:
        start = time.perf_counter()
        snap = Snapshot(t, frame_index, capture_status)
        capture_ok = capture_status == CaptureStatus.CONNECTED
        with self.lock:
            self.timers.tick()
            if self.halted:
                return snap
            if self.enabled["cavebot"]:
                self._guard(snap, "cavebot", self._run_cavebot, image, t, capture_ok)
            if self.enabled["battle"] and capture_ok:
                self._guard(snap, "battle", self._run_battle, image, t)
            if self.enabled["health"]:
                self._guard(snap, "health", self._run_health, image, t, capture_ok)
            if self.enabled["sio"]:
                self._guard(snap, "sio", self._run_sio, image, t, capture_ok)
        snap.elapsed_ms = (time.perf_counter() - start) * 1000
        return snap

    def _guard(self, snap: Snapshot, name: str, fn, *args: Any) -> None:
        try:
            fn(snap, *args)
        except Exception as exc:  # módulo com erro não derruba os outros
            snap.errors[name] = f"{type(exc).__name__}: {exc}"
            now = time.monotonic()
            if now - self._error_logged.get(name, -1e9) > 10:
                self._error_logged[name] = now
                log.exception("erro no módulo %s", name)
                record_event("engine", "module_error", f"erro no módulo {name}: {exc}", logging.ERROR)

    def _run_cavebot(self, snap: Snapshot, image: np.ndarray, t: float, capture_ok: bool) -> None:
        snap.minimap = self.localizer.crop(image)
        if capture_ok:
            loc = self.localizer.locate(image, t)
        else:
            from cave_navigation import LocStatus

            loc = Localization(LocStatus.LOW_CONFIDENCE, t, reasons=["captura congelada/desconectada"])
        snap.localization = loc
        assert self.cavebot is not None
        self.cavebot.update(loc)
        snap.cavebot = self.cavebot.status()

    def _run_battle(self, snap: Snapshot, image: np.ndarray, t: float) -> None:
        if self.battle_reader.region is not None:
            snap.battle_rows = self.battle_reader.read(image)
            for ev in self.battle_tracker.update(snap.battle_rows, t):
                name = ev.name or "?"
                if ev.kind == "entered":
                    record_event("battle", "entered", f"entrou #{ev.track_id} {name} ({ev.hp_percent:.0f}%)")
                elif ev.kind == "left":
                    record_event("battle", "left", f"saiu #{ev.track_id} {name}")
                elif ev.kind == "hp_changed":
                    record_event("battle", "hp_changed", f"#{ev.track_id} {name}: "
                                 f"{ev.previous_hp:.0f}% → {ev.hp_percent:.0f}%", logging.DEBUG)
        for kind, tr in self.visual_tracker.update(self.detector.detect(image, t), t):
            record_event("battle", f"visual_{kind}",
                         f"visual #{tr.track_id} {tr.name} {'apareceu' if kind == 'appeared' else 'sumiu'} "
                         f"(confiança {tr.confidence:.2f})")
        snap.visual_tracks = self.visual_tracker.active_tracks()
        snap.targets = self.fusion.update(self.battle_tracker.active_tracks(),
                                          list(self.visual_tracker.tracks.values()), t)

    def _run_health(self, snap: Snapshot, image: np.ndarray, t: float, capture_ok: bool) -> None:
        snap.health = self.health.update(image, t, capture_ok)

    def _run_sio(self, snap: Snapshot, image: np.ndarray, t: float, capture_ok: bool) -> None:
        snap.sio = self.sio.update(image, t, capture_ok)
