import numpy as np
import pytest

from config import DEFAULT_CONFIG
from health_monitor import BarReader, HealthMonitor
from logger import EVENTS
from sio_monitor import Identity, SioMonitor
from synthetic import HP_REGION, SIO_REGION, frame, hp_bar
from vision_common import Region

HP_RANGES = DEFAULT_CONFIG["health"]["fill_hsv_ranges"]


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


@pytest.mark.parametrize("pct", [100, 87, 50, 31, 12, 3])
def test_bar_reader_accuracy(pct):
    r = BarReader(HP_RANGES).read(hp_bar(pct))
    assert abs(r.percent - pct) <= 1.0 and r.confidence > 0.95


def test_bar_reader_flags_noise_and_empty():
    reader = BarReader(HP_RANGES)
    noisy = hp_bar(60)
    rng = np.random.default_rng(0)
    cols = rng.choice(np.arange(130, 198), 30, replace=False)
    noisy[2:10, cols] = (0, 0, 200)            # "vermelho" espalhado depois do fim da barra
    r = reader.read(noisy)
    assert r.confidence < 0.9 and any("contradizem" in x for x in r.reasons)
    empty = reader.read(np.full((12, 200, 3), 30, np.uint8))
    assert empty.confidence <= 0.3


def test_calibrate_full_span():
    reader = BarReader(HP_RANGES)
    full = np.full((12, 240, 3), 25, np.uint8)
    full[:, 20:220] = hp_bar(100)                # barra com margem dos dois lados
    assert reader.calibrate_full(full) == 198
    half = np.full((12, 240, 3), 25, np.uint8)
    half[:, 20:220] = hp_bar(50)
    assert abs(reader.read(half).percent - 50) <= 1


def monitor(clock, sounds):
    return HealthMonitor(BarReader(HP_RANGES), Region(*HP_REGION), threshold_percent=50,
                         hysteresis_percent=5, min_confidence=0.7, stale_seconds=1.0,
                         clock=clock, sound=lambda: sounds.append(1))


def test_low_hp_event_alert_and_recovery_with_hysteresis():
    clock, sounds, alerts = Clock(), [], []
    m = monitor(clock, sounds)
    m.add_alert_listener(lambda lvl, msg: alerts.append((lvl, msg)))
    EVENTS.clear()
    for t, hp in enumerate([90, 70, 45, 40, 52, 54, 60]):
        clock.t = t * 0.1
        m.update(frame(hp=hp), clock.t)
    kinds = [e.kind for e in EVENTS.events(source="health")]
    assert kinds == ["hp_low", "hp_recovered"]
    assert len(m.low_events) == 1 and abs(m.low_events[0][1] - 45) <= 1
    assert alerts[0][0] == "critical" and "Exura Vita" in alerts[0][1]
    assert sounds


def test_uncertain_readings_never_trigger():
    clock, sounds = Clock(), []
    m = monitor(clock, sounds)
    EVENTS.clear()
    m.update(frame(hp=None), 0.0)                    # sem barra na região
    assert not m.last.valid and not m.low
    m.update(frame(hp=20), 0.1, capture_ok=False)    # captura congelada
    assert not m.last.valid and not m.low and not m.heal_recommended
    assert not sounds
    assert [e.kind for e in EVENTS.events(source="health")] == ["uncertain"]


def test_stale_reading():
    clock, sounds = Clock(), []
    m = monitor(clock, sounds)
    m.update(frame(hp=30), 0.0)
    assert m.heal_recommended
    clock.t = 5.0
    r = m.current()
    assert r.stale and not r.valid and not m.heal_recommended


def sio(clock, template=None):
    s = DEFAULT_CONFIG["sio"]
    return SioMonitor(BarReader(s["fill_hsv_ranges"]), Region(*SIO_REGION),
                      name_region=Region(0, 0, 200, 60), identity_template=template,
                      threshold_percent=60, clock=clock, sound=lambda: None)


def test_sio_without_identity_does_not_alert():
    clock, alerts = Clock(), []
    m = sio(clock)
    m.add_alert_listener(lambda *a: alerts.append(a))
    EVENTS.clear()
    m.update(frame(sio=100), 0)
    r = m.update(frame(sio=40), 0.1)
    assert r.valid and abs(r.percent - 40) <= 1.5
    assert r.identity == Identity.NOT_VERIFIED
    assert not alerts
    assert [e.kind for e in EVENTS.events(source="sio")] == ["bar_low_unconfirmed"]


def test_sio_with_confirmed_identity_alerts_and_detects_mismatch():
    clock, alerts = Clock(), []
    img = frame(sio=100)
    tpl = img[5:30, 10:60].copy()                    # "nome" do aliado
    m = sio(clock, template=tpl)
    m.add_alert_listener(lambda *a: alerts.append(a))
    assert m.update(img, 0).identity == Identity.CONFIRMED
    low = frame(sio=30)
    low[5:30, 10:60] = tpl
    m.update(low, 0.1)
    assert alerts and "Exura Sio" in alerts[0][1]
    other = frame(sio=30, seed=5)                    # sem o nome: outra pessoa?
    other[0:60, 0:200] = 0
    other[40:46, 20:140] = frame(sio=30)[40:46, 20:140]
    assert m.update(other, 0.2).identity == Identity.NOT_CONFIRMED
