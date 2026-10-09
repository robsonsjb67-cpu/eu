"""Registro de eventos: arquivo rotativo, console e um histórico em memória.

Os módulos de análise usam ``logging`` normalmente e, para eventos que a
interface deve exibir (waypoint reconhecido, HP baixo, alvo entrou...),
chamam ``record_event``. O ``EventLog`` é seguro entre threads e não depende
do Qt: a interface só assina as novidades.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Optional

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


@dataclass(frozen=True)
class Event:
    timestamp: float               # time.time()
    source: str                    # módulo: capture, cavebot, battle, health, sio, timers...
    kind: str                      # identificador curto: waypoint_reached, hp_low...
    message: str
    level: int = logging.INFO
    data: dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"{ts} [{self.source}] {self.message}"


class EventLog:
    """Histórico circular de eventos com assinantes."""

    def __init__(self, maxlen: int = 2000):
        self._events: deque[Event] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[Event], None]] = []

    def record(self, source: str, kind: str, message: str, level: int = logging.INFO,
               **data: Any) -> Event:
        ev = Event(time.time(), source, kind, message, level, data)
        with self._lock:
            self._events.append(ev)
            subs = list(self._subscribers)
        logging.getLogger(f"eu.{source}").log(level, "%s", message)
        for cb in subs:
            try:
                cb(ev)
            except Exception:  # um assinante com erro não derruba os demais
                logging.getLogger(__name__).exception("erro em assinante do EventLog")
        return ev

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def events(self, source: Optional[str] = None, kind: Optional[str] = None) -> list[Event]:
        with self._lock:
            items = list(self._events)
        return [e for e in items if (source is None or e.source == source)
                and (kind is None or e.kind == kind)]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


#: Histórico global usado por padrão pelos módulos.
EVENTS = EventLog()


def record_event(source: str, kind: str, message: str, level: int = logging.INFO, **data: Any) -> Event:
    return EVENTS.record(source, kind, message, level, **data)


def setup_logging(cfg: Optional[dict] = None) -> None:
    """Configura console + arquivo rotativo. Pode ser chamada mais de uma vez."""
    lc = (cfg or {}).get("logging", {})
    level = getattr(logging, str(lc.get("level", "INFO")).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        if getattr(h, "_eu_handler", False):
            root.removeHandler(h)
            h.close()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    console._eu_handler = True  # type: ignore[attr-defined]
    root.addHandler(console)
    path = lc.get("file")
    if path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            fh = RotatingFileHandler(path, maxBytes=int(lc.get("max_bytes", 2_000_000)),
                                     backupCount=int(lc.get("backup_count", 3)), encoding="utf-8")
            fh.setFormatter(logging.Formatter(LOG_FORMAT))
            fh._eu_handler = True  # type: ignore[attr-defined]
            root.addHandler(fh)
        except OSError:
            root.exception("não foi possível abrir o arquivo de log %s", path)


def play_alert_sound() -> None:
    """Bipe de alerta sem bloquear (winsound no Windows, sino do terminal fora dele)."""
    def _beep() -> None:
        try:
            import winsound  # type: ignore

            winsound.Beep(1000, 250)
        except Exception:
            try:
                print("\a", end="", flush=True)
            except Exception:
                pass

    threading.Thread(target=_beep, name="alert-beep", daemon=True).start()
