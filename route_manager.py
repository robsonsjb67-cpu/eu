"""Gerenciador de rotas do CaveBot: waypoints, sequência e persistência em JSON.

Uma rota é uma lista ordenada de waypoints. Cada waypoint é identificado
visualmente por uma ou mais *referências de minimapa* (ver
``cave_navigation.py``) e, opcionalmente, por coordenadas informadas pelo
usuário. Este módulo só cuida dos dados; a validação visual fica em
``cave_navigation.py`` e o acompanhamento em ``cavebot.py``.

Arquivo de rota (``routes/<nome>.json``)::

    {
      "name": "Rotworm Cave",
      "loop": true,
      "description": "...",
      "waypoints": [
        {"id": "a1b2c3", "name": "Entrada", "kind": "walk",
         "position": [32100, 32200, 7], "radius": 2,
         "reference_ids": ["ref_entrada"], "notes": ""}
      ]
    }
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

#: Tipos de waypoint (chave -> descrição exibida).
WAYPOINT_KINDS: dict[str, str] = {
    "walk": "Andar",
    "stand": "Parar / aguardar",
    "ladder": "Escada (sobe)",
    "rope": "Corda (sobe)",
    "hole": "Buraco / descida",
    "stairs_down": "Escada (desce)",
    "door": "Porta",
    "label": "Rótulo / marcador",
}

#: Tipos que normalmente mudam de andar ao serem concluídos.
FLOOR_CHANGE_KINDS = {"ladder", "rope", "hole", "stairs_down"}


class RouteError(Exception):
    """Erro de validação ou de arquivo de rota."""


@dataclass
class Waypoint:
    name: str
    kind: str = "walk"
    position: Optional[tuple[int, int, int]] = None   # (x, y, z) informado pelo usuário
    radius: int = 2                                    # tolerância em sqm
    reference_ids: list[str] = field(default_factory=list)
    notes: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self) -> None:
        if self.kind not in WAYPOINT_KINDS:
            raise RouteError(f"tipo de waypoint inválido: {self.kind!r}")
        if not str(self.name).strip():
            raise RouteError("waypoint sem nome")
        if self.position is not None:
            if len(self.position) != 3:
                raise RouteError("posição deve ter (x, y, z)")
            self.position = tuple(int(v) for v in self.position)  # type: ignore[assignment]
        self.radius = max(0, int(self.radius))
        self.reference_ids = [str(r) for r in self.reference_ids]

    @property
    def changes_floor(self) -> bool:
        return self.kind in FLOOR_CHANGE_KINDS

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["position"] = list(self.position) if self.position else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Waypoint":
        pos = d.get("position")
        return cls(name=d["name"], kind=d.get("kind", "walk"),
                   position=tuple(pos) if pos else None, radius=d.get("radius", 2),
                   reference_ids=list(d.get("reference_ids", [])), notes=d.get("notes", ""),
                   id=d.get("id") or uuid.uuid4().hex[:8])

    def describe(self) -> str:
        pos = f" @ {self.position[0]},{self.position[1]},{self.position[2]}" if self.position else ""
        return f"{self.name} [{WAYPOINT_KINDS[self.kind]}]{pos}"


@dataclass
class Route:
    name: str
    waypoints: list[Waypoint] = field(default_factory=list)
    loop: bool = False
    description: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------ edição
    def add(self, wp: Waypoint, index: Optional[int] = None) -> Waypoint:
        if any(w.id == wp.id for w in self.waypoints):
            raise RouteError(f"waypoint com id repetido: {wp.id}")
        if index is None:
            self.waypoints.append(wp)
        else:
            self.waypoints.insert(max(0, min(index, len(self.waypoints))), wp)
        self._touch()
        return wp

    def remove(self, index: int) -> Waypoint:
        self._check_index(index)
        wp = self.waypoints.pop(index)
        self._touch()
        return wp

    def move(self, index: int, new_index: int) -> None:
        self._check_index(index)
        new_index = max(0, min(new_index, len(self.waypoints) - 1))
        wp = self.waypoints.pop(index)
        self.waypoints.insert(new_index, wp)
        self._touch()

    def replace(self, index: int, wp: Waypoint) -> None:
        self._check_index(index)
        self.waypoints[index] = wp
        self._touch()

    def index_of(self, waypoint_id: str) -> int:
        for i, w in enumerate(self.waypoints):
            if w.id == waypoint_id:
                return i
        raise RouteError(f"waypoint {waypoint_id} não está na rota")

    def validate(self) -> list[str]:
        """Problemas que impedem ou prejudicam o reconhecimento visual."""
        problems = []
        if not self.waypoints:
            problems.append("rota sem waypoints")
        for i, w in enumerate(self.waypoints, 1):
            if not w.reference_ids and w.position is None:
                problems.append(f"#{i} '{w.name}': sem referência de minimapa nem posição — "
                                "não pode ser reconhecido visualmente")
        return problems

    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self.waypoints):
            raise RouteError(f"índice de waypoint fora da rota: {index}")

    def _touch(self) -> None:
        self.updated_at = time.time()

    # ------------------------------------------------------------ JSON
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "loop": self.loop, "description": self.description,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "waypoints": [w.to_dict() for w in self.waypoints]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Route":
        if not isinstance(d, dict) or "name" not in d:
            raise RouteError("arquivo de rota inválido (sem 'name')")
        return cls(name=d["name"], loop=bool(d.get("loop", False)),
                   description=d.get("description", ""),
                   created_at=d.get("created_at", time.time()),
                   updated_at=d.get("updated_at", time.time()),
                   waypoints=[Waypoint.from_dict(w) for w in d.get("waypoints", [])])


def safe_filename(name: str) -> str:
    slug = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE).strip().replace(" ", "_")
    if not slug:
        raise RouteError("nome de rota inválido")
    return slug[:80]


class RouteManager:
    """CRUD de rotas em uma pasta de arquivos JSON."""

    def __init__(self, directory: str = "routes"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def path_for(self, name: str) -> str:
        return os.path.join(self.directory, safe_filename(name) + ".json")

    def list_routes(self) -> list[str]:
        names = []
        for fname in sorted(os.listdir(self.directory)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.directory, fname), encoding="utf-8") as fh:
                    names.append(json.load(fh)["name"])
            except (OSError, ValueError, KeyError):
                continue  # arquivo estranho na pasta: ignora
        return names

    def exists(self, name: str) -> bool:
        return os.path.exists(self.path_for(name))

    def create(self, name: str, overwrite: bool = False) -> Route:
        if self.exists(name) and not overwrite:
            raise RouteError(f"já existe uma rota chamada '{name}'")
        route = Route(name)
        self.save(route)
        return route

    def save(self, route: Route) -> str:
        path = self.path_for(route.name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(route.to_dict(), fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    def load(self, name: str) -> Route:
        path = self.path_for(name)
        try:
            with open(path, encoding="utf-8") as fh:
                return Route.from_dict(json.load(fh))
        except FileNotFoundError:
            raise RouteError(f"rota '{name}' não encontrada") from None
        except ValueError as exc:
            raise RouteError(f"rota '{name}' corrompida: {exc}") from exc

    def delete(self, name: str) -> None:
        try:
            os.remove(self.path_for(name))
        except FileNotFoundError:
            raise RouteError(f"rota '{name}' não encontrada") from None

    def rename(self, old: str, new: str) -> Route:
        route = self.load(old)
        if self.exists(new) and safe_filename(new) != safe_filename(old):
            raise RouteError(f"já existe uma rota chamada '{new}'")
        self.delete(old)
        route.name = new
        self.save(route)
        return route
