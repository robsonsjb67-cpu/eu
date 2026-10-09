from battle_attack import BattleTrack
from monster_detector import VisualTrack
from target_fusion import TargetFusion, TargetStatus


def b(i, name, first, row=0, hp=100.0):
    return BattleTrack(i, name, hp, first, first, row, 0, confirmed=True)


def v(i, name, first, last, conf=0.9):
    return VisualTrack(i, name, (10 * i, 10, 32, 32), conf, first, last, confirmed=True)


def statuses(f):
    return {x.target_id: x.status for x in f.snapshot()}


def test_confirmed_probable_and_uncertain():
    f = TargetFusion()
    battle = [b(1, "rat", 0.0, 0), b(2, "rat", 0.5, 1), b(3, None, 1.0, 2)]
    visual = [v(1, "rat", 0.1, 2.0), v(2, "troll", 1.2, 2.0)]
    out = {x.battle_track_id: x for x in f.update(battle, visual, 2.0)}
    assert out[1].status == TargetStatus.CONFIRMED
    assert out[2].status == TargetStatus.PROBABLE
    assert out[1].candidate_visual_ids == [1]
    # Sem nome: só evidência temporal -> provável, com o candidato listado.
    assert out[3].status == TargetStatus.PROBABLE and out[3].name == "troll"
    assert len(f.snapshot()) == 3  # troll coberto pela entrada sem nome; sem duplicata


def test_visual_only_is_uncertain_then_merged():
    f = TargetFusion()
    f.update([], [v(1, "rat", 0.0, 1.0)], 1.0)
    (only,) = f.snapshot()
    assert only.status == TargetStatus.UNCERTAIN and only.source == "visual"
    f.update([b(5, "rat", 1.1)], [v(1, "rat", 0.0, 1.2)], 1.2)
    snap = f.snapshot()
    assert len(snap) == 1 and snap[0].status == TargetStatus.CONFIRMED


def test_stable_ids_and_gone():
    f = TargetFusion(retain_gone_seconds=1.0)
    f.update([b(1, "rat", 0.0)], [v(1, "rat", 0.0, 0.5)], 0.5)
    tid = f.snapshot()[0].target_id
    f.update([b(1, "rat", 0.0)], [v(1, "rat", 0.0, 0.6)], 0.6)
    assert f.snapshot()[0].target_id == tid
    f.update([], [], 0.7)
    assert statuses(f)[tid] == TargetStatus.GONE
    assert "saiu da Battle List" in f.explain(tid)
    f.update([], [], 2.0)
    assert f.snapshot() == []


def test_more_entries_than_visible_never_pairs_rows_directly():
    f = TargetFusion()
    out = f.update([b(1, "rat", 0, 0), b(2, "rat", 0, 1)], [v(1, "rat", 0, 1.0), v(2, "rat", 0, 1.0)], 1.0)
    assert all(x.status == TargetStatus.CONFIRMED for x in out)
    assert all(x.candidate_visual_ids == [1, 2] for x in out)
