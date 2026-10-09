import numpy as np
import cv2

from battle_attack import BattleListReader, BattleListTracker

FULL = 100


def battle_image(hps, row_h=22):
    img = np.full((row_h * 8, 160, 3), 40, np.uint8)
    for i, hp in enumerate(hps):
        y = i * row_h + 15
        cv2.rectangle(img, (29, y - 1), (30 + FULL, y + 4), (0, 0, 0), -1)
        color = (0, 192, 0) if hp > 60 else (0, 192, 192) if hp > 30 else (0, 0, 192)
        w = max(1, int(FULL * hp / 100))
        img[y:y + 4, 30:30 + w] = color
        cv2.putText(img, "Rat", (30, y - 3), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 200, 200), 1)
    return img


def reader():
    return BattleListReader(bar_full_width=FULL, ocr=False)


def test_reads_rows_and_hp():
    rows = reader().read_roi(battle_image([100, 50, 20]))
    assert [r.row_index for r in rows] == [0, 1, 2]
    assert [round(r.hp_percent) for r in rows] == [100, 50, 20]
    assert [r.bar_color for r in rows] == ["green", "yellow", "red"]


def test_tracker_counts_each_monster_once():
    rd, tr = reader(), BattleListTracker(confirm_frames=2, leave_grace_seconds=0.5)
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.update(rd.read_roi(battle_image([100, 80])), t)
    ids = [x.track_id for x in tr.active_tracks()]
    assert len(ids) == 2 and tr.total_confirmed == 2
    # Primeira criatura morre: a segunda sobe para a linha 0, mas mantém o ID.
    events = []
    for _ in range(10):
        t += 0.1
        events += tr.update(rd.read_roi(battle_image([75])), t)
    assert [x.track_id for x in tr.active_tracks()] == [ids[1]]
    assert [e.kind for e in events if e.kind == "left"] == ["left"]
    assert tr.total_confirmed == 2


def test_flicker_does_not_recount():
    rd, tr = reader(), BattleListTracker(confirm_frames=2, leave_grace_seconds=0.5)
    seq = [[100], [100], [], [100], [100]]
    for i, hps in enumerate(seq):
        tr.update(rd.read_roi(battle_image(hps)), i * 0.1)
    assert tr.total_confirmed == 1


def test_hp_change_event():
    rd, tr = reader(), BattleListTracker(confirm_frames=1)
    tr.update(rd.read_roi(battle_image([100])), 0.0)
    ev = tr.update(rd.read_roi(battle_image([60])), 0.1)
    assert ev[0].kind == "hp_changed" and ev[0].previous_hp == 100
