"""Temporizadores de magias (contagem regressiva configurável).

Padrões: Utani Gran Hur 30 s, Utamo Vita 50 s (editáveis na interface e
salvos em ``config.json``). Cada temporizador tem iniciar, pausar,
continuar e reiniciar, além de registro de eventos. O temporizador só mede
tempo: ele avisa quando a magia deve estar perto de acabar, sem lançá-la.
"""
from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from logger import play_alert_sound, record_event

log = logging.getLogger(__name__)


class TimerState(enum.Enum):
    IDLE = "parado"
    RUNNING = "contando"
    PAUSED = "pausado"
    EXPIRED = "expirado"


@dataclass
class TimerEvent:
    timestamp: float
    timer: str
    kind: str          # started / paused / resumed / reset / warning / expired / interval_changed
    remaining: float


class SpellTimer:
    def __init__(self, name: str, seconds: float, warn_seconds: float = 5.0, sound: bool = True,
                 clock: Callable[[], float] = time.monotonic,
                 beep: Callable[[], None] = play_alert_sound):
        if seconds <= 0:
            raise ValueError("o intervalo precisa ser positivo")
        self.name = name
        self.seconds = float(seconds)
        self.warn_seconds = max(0.0, float(warn_seconds))
        self.sound = sound
        self.clock = clock
        self.beep = beep
        self.state = TimerState.IDLE
        self.events: list[TimerEvent] = []
        self._deadline: Optional[float] = None
        self._remaining = self.seconds
        self._warned = False

    # ------------------------------------------------------------ controles
    def start(self) -> None:
        """Inicia (ou reinicia do zero) a contagem — use ao lançar a magia."""
        self._deadline = self.clock() + self.seconds
        self._remaining = self.seconds
        self._warned = False
        self.state = TimerState.RUNNING
        self._log("started", f"{self.name}: contagem iniciada ({self.seconds:.0f}s)")

    def pause(self) -> None:
        if self.state != TimerState.RUNNING:
            return
        self._remaining = self.remaining()
        self._deadline = None
        self.state = TimerState.PAUSED
        self._log("paused", f"{self.name}: pausado com {self._remaining:.1f}s")

    def resume(self) -> None:
        if self.state != TimerState.PAUSED:
            return
        self._deadline = self.clock() + self._remaining
        self.state = TimerState.RUNNING
        self._log("resumed", f"{self.name}: retomado ({self._remaining:.1f}s)")

    def reset(self) -> None:
        self._deadline = None
        self._remaining = self.seconds
        self._warned = False
        self.state = TimerState.IDLE
        self._log("reset", f"{self.name}: reiniciado")

    def set_interval(self, seconds: float, warn_seconds: Optional[float] = None) -> None:
        if seconds <= 0:
            raise ValueError("o intervalo precisa ser positivo")
        self.seconds = float(seconds)
        if warn_seconds is not None:
            self.warn_seconds = max(0.0, float(warn_seconds))
        if self.state == TimerState.IDLE:
            self._remaining = self.seconds
        self._log("interval_changed", f"{self.name}: intervalo agora {self.seconds:.0f}s")

    # ------------------------------------------------------------ leitura
    def remaining(self) -> float:
        if self.state == TimerState.RUNNING and self._deadline is not None:
            return max(0.0, self._deadline - self.clock())
        if self.state == TimerState.EXPIRED:
            return 0.0
        return self._remaining

    def progress(self) -> float:
        """Fração já decorrida (0–1)."""
        return 1.0 - self.remaining() / self.seconds

    def tick(self) -> list[TimerEvent]:
        """Atualiza avisos/expiração; chame periodicamente (ex.: 10x/s)."""
        out: list[TimerEvent] = []
        if self.state != TimerState.RUNNING:
            return out
        rem = self.remaining()
        if not self._warned and self.warn_seconds > 0 and rem <= self.warn_seconds and rem > 0:
            self._warned = True
            out.append(self._log("warning", f"{self.name}: faltam {rem:.0f}s", logging.WARNING))
            if self.sound:
                self.beep()
        if rem <= 0:
            self.state = TimerState.EXPIRED
            self._deadline = None
            self._remaining = 0.0
            out.append(self._log("expired", f"{self.name}: tempo esgotado — renovar a magia",
                                 logging.WARNING))
            if self.sound:
                self.beep()
        return out

    def format(self) -> str:
        rem = self.remaining()
        return f"{int(rem // 60):02d}:{rem % 60:04.1f}"

    def to_dict(self) -> dict:
        return {"name": self.name, "seconds": self.seconds, "warn_seconds": self.warn_seconds,
                "sound": self.sound}

    def _log(self, kind: str, message: str, level: int = logging.INFO) -> TimerEvent:
        ev = TimerEvent(self.clock(), self.name, kind, self.remaining())
        self.events.append(ev)
        del self.events[:-200]
        record_event("timers", kind, message, level, timer=self.name)
        return ev


@dataclass
class TimerManager:
    timers: dict[str, SpellTimer] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict, clock: Callable[[], float] = time.monotonic,
                    beep: Callable[[], None] = play_alert_sound) -> "TimerManager":
        mgr = cls()
        for t in cfg["spell_timers"]["timers"]:
            try:
                mgr.add(SpellTimer(t["name"], t["seconds"], t.get("warn_seconds", 5.0),
                                   t.get("sound", True), clock, beep))
            except (KeyError, ValueError):
                log.exception("temporizador inválido na configuração: %r", t)
        return mgr

    def add(self, timer: SpellTimer) -> SpellTimer:
        self.timers[timer.name] = timer
        return timer

    def remove(self, name: str) -> None:
        self.timers.pop(name, None)

    def get(self, name: str) -> SpellTimer:
        return self.timers[name]

    def tick(self) -> list[TimerEvent]:
        out = []
        for t in self.timers.values():
            out += t.tick()
        return out

    def stop_all(self) -> None:
        for t in self.timers.values():
            if t.state != TimerState.IDLE:
                t.reset()

    def to_config(self) -> list[dict]:
        return [t.to_dict() for t in self.timers.values()]
