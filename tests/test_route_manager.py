import json

import pytest

from route_manager import Route, RouteError, RouteManager, Waypoint


def test_crud_and_json_roundtrip(tmp_path):
    rm = RouteManager(str(tmp_path))
    r = rm.create("Rotworm Cave")
    with pytest.raises(RouteError):
        rm.create("Rotworm Cave")
    r.add(Waypoint("Entrada", position=(100, 200, 7), reference_ids=["ref_a"]))
    r.add(Waypoint("Escada", kind="ladder", radius=0))
    r.add(Waypoint("Meio"), index=1)
    r.loop = True
    rm.save(r)
    data = json.loads((tmp_path / "Rotworm_Cave.json").read_text(encoding="utf-8"))
    assert [w["name"] for w in data["waypoints"]] == ["Entrada", "Meio", "Escada"]
    loaded = rm.load("Rotworm Cave")
    assert loaded.loop and loaded.waypoints[0].position == (100, 200, 7)
    assert loaded.waypoints[2].changes_floor
    assert rm.list_routes() == ["Rotworm Cave"]
    rm.rename("Rotworm Cave", "Rot 2")
    assert rm.list_routes() == ["Rot 2"]
    rm.delete("Rot 2")
    assert rm.list_routes() == []
    with pytest.raises(RouteError):
        rm.load("Rot 2")


def test_reorder_remove_and_validate():
    r = Route("x")
    a, b, c = Waypoint("a", reference_ids=["r"]), Waypoint("b"), Waypoint("c", position=(1, 1, 7))
    for w in (a, b, c):
        r.add(w)
    r.move(0, 2)
    assert [w.name for w in r.waypoints] == ["b", "c", "a"]
    problems = r.validate()
    assert len(problems) == 1 and "'b'" in problems[0]
    r.remove(0)
    assert r.index_of(a.id) == 1
    assert Route("vazia").validate() == ["rota sem waypoints"]


def test_invalid_waypoints():
    with pytest.raises(RouteError):
        Waypoint("x", kind="teleporte_magico")
    with pytest.raises(RouteError):
        Waypoint("  ")
    with pytest.raises(RouteError):
        Waypoint("x", position=(1, 2))


def test_corrupted_route_file(tmp_path):
    rm = RouteManager(str(tmp_path))
    (tmp_path / "ruim.json").write_text("{ruim")
    assert rm.list_routes() == []
    with pytest.raises(RouteError):
        rm.load("ruim")
