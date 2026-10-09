"""Reconhecimento do minimapa, CaveBot e modo de observação com imagens gravadas."""
import cv2
import numpy as np
import pytest

from cave_navigation import LocStatus, MinimapLocalizer, ReferenceLibrary, WaypointMatcher, calibration_report
from cavebot import BotState, CaveBot, run_observation
from obs_capture import iter_recording
from route_manager import Route, Waypoint
from synthetic import MINIMAP_REGION, PPS, frame, minimap_at, world_map
from vision_common import Region

WORLD = world_map()


@pytest.fixture
def lib(tmp_path):
    lib = ReferenceLibrary(str(tmp_path / "refs"))
    lib.add(minimap_at(WORLD, 100, 100), "A", (1000, 1000, 7), ref_id="a")
    lib.add(minimap_at(WORLD, 112, 100), "B", (1012, 1000, 7), ref_id="b")
    lib.add(minimap_at(WORLD, 112, 112), "C", None, ref_id="c")          # sem coordenadas
    return lib


def localizer(lib):
    return MinimapLocalizer(lib, Region(*MINIMAP_REGION), pixels_per_sqm=PPS)


def test_library_persists(lib):
    again = ReferenceLibrary(lib.directory)
    assert set(again.refs) == {"a", "b", "c"}
    assert again.refs["a"].position == (1000, 1000, 7) and again.refs["c"].position is None
    again.remove("c")
    assert set(ReferenceLibrary(lib.directory).refs) == {"a", "b"}


def test_exact_and_offset_positions(lib):
    loc = localizer(lib)
    r = loc.locate(frame(WORLD, (100, 100)), 0)
    assert r.ok and r.ref_id == "a" and r.position == (1000, 1000, 7) and r.confidence > 0.95
    r = loc.locate(frame(WORLD, (103, 98)), 0)
    assert r.ok and r.position == (1003, 998, 7)


def test_reference_without_coordinates_never_invents_position(lib):
    r = localizer(lib).locate(frame(WORLD, (112, 113)), 0)
    assert r.ok and r.ref_id == "c" and r.position is None
    assert r.offset_px == (0.0, PPS * 1.0)
    assert any("sem coordenadas" in x for x in r.reasons)


def test_unknown_place_is_low_confidence(lib):
    r = localizer(lib).locate(frame(WORLD, (220, 40)), 0)
    assert r.status == LocStatus.LOW_CONFIDENCE and r.position is None and r.ref_id is None
    assert "limiar" in r.reasons[0]


def test_blank_minimap_and_missing_region(lib):
    blank = np.zeros((360, 640, 3), np.uint8)
    assert localizer(lib).locate(blank, 0).status == LocStatus.NO_DETAIL
    assert MinimapLocalizer(lib, None).locate(blank, 0).status == LocStatus.NO_REGION


def test_ambiguous_duplicate_references(tmp_path):
    lib = ReferenceLibrary(str(tmp_path))
    mm = minimap_at(WORLD, 50, 50)
    lib.add(mm, "X", (10, 10, 7), ref_id="x")
    lib.add(mm, "Y", (90, 90, 7), ref_id="y")       # mesma imagem, outro lugar
    r = localizer(lib).locate(frame(WORLD, (50, 50)), 0)
    assert r.status == LocStatus.AMBIGUOUS and r.position is None


def test_floor_change_suspected(lib):
    loc = localizer(lib)
    loc.locate(frame(WORLD, (100, 100)), 0)
    r = loc.locate(frame(WORLD, (101, 100)), 0.1)
    assert not r.floor_change_suspected                 # andar 1 sqm não é troca de andar
    other_floor = world_map(seed=99)
    r = loc.locate(frame(other_floor, (100, 100)), 0.2)
    assert r.floor_change_suspected and not r.ok


def test_matcher(lib):
    m = WaypointMatcher(lib, PPS)
    loc = localizer(lib)
    wp = Waypoint("A", reference_ids=["a"], radius=1)
    assert m.check(loc.locate(frame(WORLD, (101, 100)), 0), wp)[0]
    ok, why = m.check(loc.locate(frame(WORLD, (105, 100)), 0), wp)
    assert not ok and "raio" in why
    by_pos = Waypoint("P", position=(1012, 1001, 7), radius=1)
    assert m.check(loc.locate(frame(WORLD, (112, 100)), 0), by_pos)[0]
    wrong_floor = Waypoint("P", position=(1012, 1000, 6), radius=1)
    assert "andar" in m.check(loc.locate(frame(WORLD, (112, 100)), 0), wrong_floor)[1]


def test_calibration_report():
    img = frame(WORLD, (100, 100))
    assert calibration_report(img, Region(*MINIMAP_REGION)) == []
    issues = calibration_report(img, Region(0, 200, 300, 40))
    assert any("retangular" in i for i in issues)
    assert calibration_report(img, None) == ["região do minimapa não definida"]


def make_route():
    r = Route("teste")
    r.add(Waypoint("A", reference_ids=["a"], radius=1))
    r.add(Waypoint("B", reference_ids=["b"], radius=1))
    r.add(Waypoint("C", reference_ids=["c"], radius=1))
    return r


def path(*points, frames_each=4):
    out = []
    for p in points:
        out += [frame(WORLD, p, seed=len(out))] * frames_each
    return out


def test_cavebot_requires_visual_confirmation_not_time(lib):
    clock = [0.0]
    bot = CaveBot(WaypointMatcher(lib, PPS), confirm_frames=3, lost_pause_seconds=1.0, clock=lambda: clock[0])
    loc = localizer(lib)
    bot.start(make_route())
    # Muito tempo parado longe do waypoint A: nunca conclui por tempo.
    for i in range(50):
        clock[0] = i * 0.1
        bot.update(loc.locate(frame(WORLD, (106, 100)), clock[0]))
    assert bot.index == 0 and not bot.hits
    for i in range(3):
        clock[0] += 0.1
        bot.update(loc.locate(frame(WORLD, (100, 100)), clock[0]))
    assert bot.index == 1 and bot.hits[0].name == "A"


def test_cavebot_pauses_when_lost_and_resumes(lib):
    clock = [0.0]
    bot = CaveBot(WaypointMatcher(lib, PPS), confirm_frames=2, lost_pause_seconds=0.5, clock=lambda: clock[0])
    loc = localizer(lib)
    bot.start(make_route())
    blank = np.zeros((360, 640, 3), np.uint8)
    for i in range(10):
        clock[0] = i * 0.1
        bot.update(loc.locate(blank, clock[0]))
    assert bot.state == BotState.WAITING_POSITION
    assert any("referência perdida" in f.reason for f in bot.failures)
    # Enquanto aguarda, nada é concluído.
    assert bot.index == 0
    clock[0] = 2.0
    bot.update(loc.locate(frame(WORLD, (100, 100)), clock[0]))
    assert bot.state == BotState.RUNNING


def test_cavebot_controls_and_out_of_order(lib):
    bot = CaveBot(WaypointMatcher(lib, PPS), confirm_frames=1)
    loc = localizer(lib)
    bot.start(make_route())
    bot.pause()
    bot.update(loc.locate(frame(WORLD, (100, 100)), 0))
    assert bot.index == 0                                    # pausado não avança
    bot.resume()
    bot.update(loc.locate(frame(WORLD, (112, 100)), 0.1))    # está em B, esperado A
    assert bot.index == 0 and "fora" not in bot.failures[-1].reason
    assert "corresponde ao waypoint #2" in bot.failures[-1].reason
    bot.set_current(1)
    bot.update(loc.locate(frame(WORLD, (112, 100)), 0.2))
    st = bot.status()
    assert st.current.name == "C" and st.next is None and st.completed == 2
    bot.stop()
    assert bot.state == BotState.STOPPED


def test_observation_mode_with_recorded_images(lib, tmp_path):
    """Grava uma "sessão" em PNGs e testa a rota em cima dela."""
    rec = tmp_path / "gravacao"
    rec.mkdir()
    frames = path((100, 100), (104, 100), (108, 100), (112, 100), (112, 104), (112, 108), (112, 112))
    frames += [np.zeros((360, 640, 3), np.uint8)] * 2   # tela preta (OBS piscou)
    for i, f in enumerate(frames):
        cv2.imwrite(str(rec / f"{i:05d}.png"), f)
    rep = run_observation(make_route(), iter_recording(str(rec)), localizer(lib), confirm_frames=3)
    assert [h.name for h in rep.hits] == ["A", "B", "C"]
    assert rep.final_state == BotState.FINISHED.value
    assert "Waypoints reconhecidos (3)" in rep.summary()


def test_observation_reports_failures(lib, tmp_path):
    frames = path((100, 100), (180, 30), frames_each=30)    # perde a referência depois de A
    rep = run_observation(make_route(), frames, localizer(lib), confirm_frames=3, lost_pause_seconds=1.0)
    assert [h.name for h in rep.hits] == ["A"]
    assert rep.final_state == BotState.WAITING_POSITION.value
    assert any("referência perdida" in f.reason for f in rep.failures)


def test_staying_on_completed_waypoint_is_not_a_failure(lib):
    bot = CaveBot(WaypointMatcher(lib, PPS), confirm_frames=2)
    loc = localizer(lib)
    bot.start(make_route())
    for i in range(10):
        bot.update(loc.locate(frame(WORLD, (100, 100)), i * 0.1))
    assert bot.index == 1 and bot.failures == []
