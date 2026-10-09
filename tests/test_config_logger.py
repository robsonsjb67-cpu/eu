import json
import logging

from config import DEFAULT_CONFIG, ConfigStore, load_config, save_config
from logger import EventLog


def test_missing_file_gives_defaults(tmp_path):
    cfg = load_config(str(tmp_path / "nao_existe.json"))
    assert cfg == DEFAULT_CONFIG and cfg is not DEFAULT_CONFIG


def test_partial_file_is_merged_and_defaults_untouched(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"health": {"threshold_percent": 33}}))
    cfg = load_config(str(p))
    assert cfg["health"]["threshold_percent"] == 33
    assert cfg["health"]["min_confidence"] == DEFAULT_CONFIG["health"]["min_confidence"]
    cfg["health"]["fill_hsv_ranges"].clear()
    assert DEFAULT_CONFIG["health"]["fill_hsv_ranges"]


def test_corrupted_file_falls_back_and_is_kept(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{ isto não é json")
    assert load_config(str(p)) == DEFAULT_CONFIG
    assert (tmp_path / "c.json.corrompido").exists()


def test_store_set_persists(tmp_path):
    p = str(tmp_path / "c.json")
    st = ConfigStore(p)
    st.set("regions.hp_bar", {"x": 1, "y": 2, "w": 3, "h": 4})
    st.set("spell_timers.timers", [{"name": "X", "seconds": 9}])
    again = ConfigStore(p)
    assert again.get("regions.hp_bar") == {"x": 1, "y": 2, "w": 3, "h": 4}
    assert again.get("spell_timers.timers")[0]["seconds"] == 9
    assert again.get("nao.existe", 5) == 5
    save_config(again.data, p)


def test_event_log_subscribers_and_filters():
    log = EventLog(maxlen=3)
    got = []
    log.subscribe(got.append)
    log.subscribe(lambda ev: 1 / 0)  # assinante com erro não atrapalha
    for i in range(5):
        log.record("health" if i % 2 else "cavebot", "k", f"m{i}", logging.INFO)
    assert len(got) == 5
    assert [e.message for e in log.events()] == ["m2", "m3", "m4"]
    assert [e.message for e in log.events(source="health")] == ["m3"]
