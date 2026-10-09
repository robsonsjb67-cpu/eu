import numpy as np
import cv2

from monster_detector import MonsterDetector, TemplateLibrary, VisualTracker
from vision_common import Region


def sprite(color):
    s = np.zeros((32, 32, 3), np.uint8)
    cv2.circle(s, (16, 16), 12, color, -1)
    cv2.rectangle(s, (10, 10), (14, 14), (255, 255, 255), -1)
    cv2.line(s, (8, 24), (24, 28), (30, 30, 30), 2)
    return s


def scene(positions, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 120, (300, 400, 3), dtype=np.uint8)
    for (x, y), spr in positions:
        img[y:y + 32, x:x + 32] = spr
    return img


def test_detects_registered_templates(tmp_path):
    lib = TemplateLibrary(str(tmp_path))
    rat, troll = sprite((40, 90, 160)), sprite((40, 160, 40))
    lib.add("Rat", rat)
    lib.add("Troll", troll)
    assert lib.names == ["rat", "troll"]
    det = MonsterDetector(lib, Region(0, 0, 400, 300), threshold=0.8)
    found = det.detect(scene([((50, 60), rat), ((200, 100), troll), ((300, 200), rat)]), 1.0)
    got = sorted((d.name, d.box[:2]) for d in found)
    assert got == [("rat", (50, 60)), ("rat", (300, 200)), ("troll", (200, 100))]
    assert all(d.confidence >= 0.8 and d.timestamp == 1.0 for d in found)


def test_masked_template_ignores_background(tmp_path):
    lib = TemplateLibrary(str(tmp_path))
    spr = sprite((40, 90, 160))
    mask = np.zeros((32, 32), np.uint8)
    cv2.circle(mask, (16, 16), 12, 255, -1)
    lib.add("Rat", spr, mask)
    assert lib.templates[0].mask is not None
    img = scene([])
    # Mesmo sprite com fundo diferente fora da máscara.
    patch = img[100:132, 100:132]
    patch[mask > 0] = spr[mask > 0]
    found = MonsterDetector(lib, threshold=0.8).detect(img, 0)
    assert [(d.name, d.box[:2]) for d in found] == [("rat", (100, 100))]


def test_tracker_requires_hits_and_tolerates_gaps(tmp_path):
    lib = TemplateLibrary(str(tmp_path))
    rat = sprite((40, 90, 160))
    lib.add("Rat", rat)
    det = MonsterDetector(lib, threshold=0.8)
    tr = VisualTracker(min_hits=2, max_missed_seconds=0.5)
    frames = [scene([((50, 60), rat)]), scene([((54, 60), rat)]), scene([]), scene([((58, 60), rat)])]
    for i, f in enumerate(frames):
        tr.update(det.detect(f, i * 0.1), i * 0.1)
    act = tr.active_tracks()
    assert len(act) == 1 and act[0].hits == 3
    for i in range(4, 15):
        tr.update([], i * 0.1)
    assert tr.active_tracks() == []
