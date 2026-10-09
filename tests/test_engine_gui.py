import os

import pytest

from analysis import AnalysisEngine
from config import ConfigStore
from logger import EVENTS
from obs_capture import CaptureStatus, FrameGrabber
from synthetic import HP_REGION, MINIMAP_REGION, frame, world_map


def store(tmp_path):
    st = ConfigStore(str(tmp_path / "config.json"))
    st.set("cavebot.routes_dir", str(tmp_path / "routes"), save=False)
    st.set("cavebot.references_dir", str(tmp_path / "refs"), save=False)
    st.set("detector.templates_dir", str(tmp_path / "tpl"), save=False)
    st.set("regions.hp_bar", dict(zip("xywh", HP_REGION)), save=False)
    st.set("regions.minimap", dict(zip("xywh", MINIMAP_REGION)))
    return st


def test_engine_runs_modules_independently(tmp_path):
    eng = AnalysisEngine(store(tmp_path))
    snap = eng.process(frame(world_map(), (100, 100), hp=40), 0.0)
    assert snap.health.valid and abs(snap.health.percent - 40) <= 1
    assert snap.localization.status.value == "sem referências"
    assert snap.minimap.shape[:2] == (MINIMAP_REGION[3], MINIMAP_REGION[2])
    # Um módulo quebrado não derruba os outros.
    eng.battle_reader.read = lambda img: 1 / 0
    eng.store.data["regions"]["battle_list"] = {"x": 0, "y": 0, "w": 10, "h": 10}
    eng.battle_reader.region = object()
    snap = eng.process(frame(hp=80), 0.1)
    assert "battle" in snap.errors and snap.health.valid


def test_emergency_stop_halts_analysis(tmp_path):
    eng = AnalysisEngine(store(tmp_path))
    eng.timers.get("Utamo Vita").start()
    eng.emergency_stop()
    snap = eng.process(frame(hp=10), 0.0)
    assert snap.health is None
    assert eng.timers.get("Utamo Vita").state.value == "parado"
    assert EVENTS.events(kind="emergency_stop")
    eng.release()
    assert eng.process(frame(hp=10), 0.1).health is not None


def test_frozen_capture_makes_readings_invalid(tmp_path):
    eng = AnalysisEngine(store(tmp_path))
    snap = eng.process(frame(hp=10), 0.0, capture_status=CaptureStatus.FROZEN)
    assert not snap.health.valid and not eng.health.low


def test_gui_smoke(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # PySide6 ou bibliotecas gráficas do sistema ausentes
        pytest.skip(f"Qt indisponível: {exc}")

    from interface import MainWindow, to_qimage

    app = QApplication.instance() or QApplication([])
    win = MainWindow(store(tmp_path))
    try:
        snap = win.engine.process(frame(world_map(), (100, 100), hp=35), 0.0)
        win._on_snapshot(snap)
        app.processEvents()
        assert win.hp_value.text().startswith("35")
        win.emergency_stop()
        assert win.release_btn.isEnabled()
        win._tick_ui()
        assert set(win._timer_widgets) == {"Utani Gran Hur", "Utamo Vita"}
        img = to_qimage(frame(hp=50))
        assert img.width() == 640 and img.height() == 360
    finally:
        win.close()


def test_gui_print_and_record(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        pytest.skip(f"Qt indisponível: {exc}")
    import time as _time

    import cv2

    from interface import MainWindow

    app = QApplication.instance() or QApplication([])
    st = store(tmp_path)
    st.set("capture.screenshots_dir", str(tmp_path / "prints"))
    win = MainWindow(st)
    try:
        img = frame(hp=60)
        win.grabber = FrameGrabber(_NullSource())
        win.grabber.push(img)
        win.take_screenshot()
        shots = list((tmp_path / "prints").glob("*.png"))
        assert len(shots) == 1 and cv2.imread(str(shots[0])).shape == img.shape
        win.record_btn.setChecked(True)
        for i in range(3):
            win._maybe_record(frame(hp=50 + i), 1000.0 + i)
        win.record_btn.setChecked(False)
        rec = next((tmp_path / "prints").glob("gravacao_*"))
        for _ in range(50):
            if len(list(rec.glob("*.png"))) == 3:
                break
            _time.sleep(0.05)
        assert len(list(rec.glob("*.png"))) == 3
    finally:
        win.grabber = None
        win.close()


class _NullSource:
    def open(self): return False
    def read(self): return False, None
    def close(self): pass
