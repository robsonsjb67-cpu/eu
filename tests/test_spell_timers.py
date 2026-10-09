import pytest

from config import DEFAULT_CONFIG
from spell_timers import SpellTimer, TimerManager, TimerState


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t


def test_defaults_from_config():
    mgr = TimerManager.from_config(DEFAULT_CONFIG, beep=lambda: None)
    assert mgr.get("Utani Gran Hur").seconds == 30
    assert mgr.get("Utamo Vita").seconds == 50


def test_countdown_pause_resume_reset_and_events():
    clock, beeps = Clock(), []
    t = SpellTimer("Utani Gran Hur", 30, warn_seconds=5, clock=clock, beep=lambda: beeps.append(1))
    assert t.remaining() == 30 and t.state == TimerState.IDLE
    t.start()
    clock.t += 10
    assert t.remaining() == pytest.approx(20)
    t.pause()
    clock.t += 100                     # pausado não conta
    assert t.remaining() == pytest.approx(20)
    t.resume()
    clock.t += 16
    assert [e.kind for e in t.tick()] == ["warning"]
    assert t.tick() == []              # aviso só uma vez
    clock.t += 5
    assert [e.kind for e in t.tick()] == ["expired"]
    assert t.state == TimerState.EXPIRED and t.remaining() == 0
    assert len(beeps) == 2
    t.reset()
    assert t.state == TimerState.IDLE and t.remaining() == 30
    kinds = [e.kind for e in t.events]
    assert kinds == ["started", "paused", "resumed", "warning", "expired", "reset"]


def test_restart_while_running_and_interval_change():
    clock = Clock()
    t = SpellTimer("Utamo Vita", 50, clock=clock, beep=lambda: None)
    t.start()
    clock.t += 40
    t.start()                          # relançou a magia: recomeça do zero
    assert t.remaining() == pytest.approx(50)
    t.reset()
    t.set_interval(45)
    assert t.remaining() == 45
    with pytest.raises(ValueError):
        t.set_interval(0)
