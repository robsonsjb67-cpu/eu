"""Mapeador: junta prints do minimapa num mapa, por andar."""
import cv2
import numpy as np

from cave_navigation import MinimapLocalizer, ReferenceLibrary
from map_builder import MapBuilder, build_from_frames
from obs_capture import iter_recording
from synthetic import MINIMAP_REGION, PPS, frame, minimap_at, world_map
from vision_common import Region

WORLD = world_map()
OTHER_FLOOR = world_map(seed=99)
WALK = ([(100 + i, 100) for i in range(40)] + [(139, 100 + i) for i in range(30)]
        + [(139 - i, 129) for i in range(30)])


def walk(builder, points, world=WORLD):
    return [builder.update(minimap_at(world, *p)) for p in points]


def mismatch(fm, last, world=WORLD):
    """Fração dos pixels do mapa montado que diferem do mapa verdadeiro."""
    dx, dy = last[0] * PPS + 1 - fm.pos[0], last[1] * PPS + 1 - fm.pos[1]
    ys, xs = np.nonzero(fm.known)
    return (np.abs(fm.canvas[ys, xs].astype(int) - world[ys + dy, xs + dx].astype(int)).sum(1) > 0).mean()


def test_stitches_walk_exactly_and_grows():
    b = MapBuilder(pixels_per_sqm=PPS)
    b.start()
    ups = walk(b, WALK)
    assert ups[0].status == "started" and {u.status for u in ups[1:]} == {"added"}
    fm = b.floor_map
    assert fm.frames == len(WALK)
    assert mismatch(fm, WALK[-1]) == 0.0
    img, _ = b.cropped(fm.name)
    # 40 x 30 sqm percorridos + o minimapa em volta
    assert img.shape[1] >= 39 * PPS + 100 and img.shape[0] >= 29 * PPS + 100


def test_inactive_and_blank_prints_do_nothing():
    b = MapBuilder(pixels_per_sqm=PPS)
    assert b.update(minimap_at(WORLD, 100, 100)).status == "idle"
    b.start()
    assert b.update(np.zeros((106, 106, 3), np.uint8)).status == "no_detail"
    assert b.update(None).status == "no_detail"
    assert not b.floors


def test_floor_change_starts_new_map_and_relocates_back():
    b = MapBuilder(pixels_per_sqm=PPS, new_floor_after=3)
    b.start()
    walk(b, WALK[:20])
    first = b.current
    ups = walk(b, [(150, 150)] * 3, OTHER_FLOOR)            # desceu a escada
    assert [u.status for u in ups] == ["lost", "lost", "new_floor"]
    second = b.current
    assert second != first
    walk(b, [(150 + i, 150) for i in range(10)], OTHER_FLOOR)
    up = b.update(minimap_at(WORLD, 110, 100))               # subiu de volta
    assert up.status == "relocated" and b.current == first
    assert mismatch(b.floors[first], (110, 100)) == 0.0
    assert mismatch(b.floors[second], (159, 150), OTHER_FLOOR) == 0.0


def test_save_load_roundtrip(tmp_path):
    b = MapBuilder(pixels_per_sqm=PPS)
    b.start()
    walk(b, WALK[:30])
    b.calibrate(1129, 1100, 7)
    b.save(str(tmp_path))
    c = MapBuilder(pixels_per_sqm=PPS)
    c.load(str(tmp_path))
    assert list(c.floors) == list(b.floors) and c.current == b.current
    fa, fb = b.floor_map, c.floor_map
    assert np.array_equal(fa.canvas, fb.canvas) and np.array_equal(fa.known, fb.known)
    assert fb.anchor == fa.anchor and fb.floor == 7
    assert c.current_world_position() == (1129, 1100, 7)
    c.start()                                                # continua mapeando de onde parou
    assert c.update(minimap_at(WORLD, 131, 100)).status == "added"


def test_calibrated_map_gives_exact_cavebot_positions(tmp_path):
    b = MapBuilder(pixels_per_sqm=PPS)
    b.start()
    walk(b, WALK)
    last = WALK[-1]
    b.calibrate(1000 + last[0], 1000 + last[1], 7)
    img, center = b.reference_for(b.current)
    lib = ReferenceLibrary(str(tmp_path))
    lib.add(img, "mapa", center)
    loc = MinimapLocalizer(lib, pixels_per_sqm=PPS)
    for p in WALK[::4]:
        r = loc.locate_minimap(minimap_at(WORLD, *p), 0)
        assert r.ok and r.position == (1000 + p[0], 1000 + p[1], 7), p
    assert loc.locate_minimap(minimap_at(WORLD, 220, 40), 0).position is None


def test_uncalibrated_map_has_no_coordinates():
    b = MapBuilder(pixels_per_sqm=PPS)
    b.start()
    walk(b, WALK[:10])
    assert b.current_world_position() is None
    assert b.reference_for(b.current)[1] is None


def test_build_from_recorded_prints(tmp_path):
    rec = tmp_path / "prints"
    rec.mkdir()
    for i, p in enumerate(WALK[::2]):
        cv2.imwrite(str(rec / f"{i:05d}.png"), frame(WORLD, p, hp=80, seed=i))
    b, counts = build_from_frames(iter_recording(str(rec)), Region(*MINIMAP_REGION).crop,
                                  MapBuilder(pixels_per_sqm=PPS))
    assert counts == {"started": 1, "added": len(WALK[::2]) - 1}
    assert mismatch(b.floor_map, WALK[::2][-1]) == 0.0
