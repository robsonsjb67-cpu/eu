"""Reconhecimento de posição pelo minimapa recortado do OBS.

Como funciona
-------------
O minimapa do Tibia é uma janela centrada no personagem (a cruz branca fica
no centro). O usuário cadastra *referências*: recortes do minimapa tirados
com o personagem num ponto conhecido (opcionalmente com as coordenadas
x, y, z desse ponto).

A cada frame:

1. O minimapa atual é recortado e validado (precisa ter detalhe visual; um
   recorte preto, uniforme ou de tamanho errado é rejeitado).
2. A parte central do minimapa atual é procurada em cada referência com
   ``cv2.matchTemplate`` (correlação normalizada, ignorando a cruz central).
   O ponto de melhor correlação dá o deslocamento, em pixels, entre o
   personagem agora e o personagem no momento da referência.
3. A posição só é estimada quando:
     * a melhor correlação passa do limiar;
     * não há outra referência quase tão boa apontando para outro lugar
       (ambiguidade);
     * a referência tem coordenadas — sem elas, informa-se apenas "perto da
       referência X, deslocado N px". **Coordenadas nunca são inventadas.**
4. Mudanças bruscas do minimapa que não se explicam por um pequeno
   deslocamento (andar) indicam possível troca de andar; se a referência
   reconhecida tem outro ``z``, a troca é confirmada pela referência.
"""
from __future__ import annotations

import enum
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from route_manager import Waypoint
from vision_common import Region

log = logging.getLogger(__name__)


class LocStatus(enum.Enum):
    OK = "ok"
    NO_REGION = "sem região"
    NO_REFERENCES = "sem referências"
    NO_DETAIL = "minimapa sem detalhe"
    LOW_CONFIDENCE = "confiança baixa"
    AMBIGUOUS = "ambíguo"


@dataclass
class MinimapReference:
    id: str
    label: str
    image: np.ndarray                                   # BGR, centrado no personagem
    position: Optional[tuple[int, int, int]] = None     # coordenadas do centro (informadas)
    created_at: float = field(default_factory=time.time)

    @property
    def floor(self) -> Optional[int]:
        return self.position[2] if self.position else None

    def meta(self) -> dict:
        return {"id": self.id, "label": self.label, "file": f"{self.id}.png",
                "position": list(self.position) if self.position else None,
                "created_at": self.created_at}


class ReferenceLibrary:
    """Referências visuais do minimapa salvas em ``<dir>/<id>.png`` + ``index.json``."""

    def __init__(self, directory: str = "minimap_refs"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.refs: dict[str, MinimapReference] = {}
        self.reload()

    @property
    def index_path(self) -> str:
        return os.path.join(self.directory, "index.json")

    def reload(self) -> None:
        self.refs = {}
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, encoding="utf-8") as fh:
                items = json.load(fh)
        except (OSError, ValueError):
            log.exception("index.json das referências ilegível")
            return
        for m in items:
            img = cv2.imread(os.path.join(self.directory, m.get("file", f"{m['id']}.png")), cv2.IMREAD_COLOR)
            if img is None:
                log.warning("referência %s sem imagem; ignorada", m.get("id"))
                continue
            pos = m.get("position")
            self.refs[m["id"]] = MinimapReference(m["id"], m.get("label", m["id"]), img,
                                                  tuple(pos) if pos else None,
                                                  m.get("created_at", time.time()))

    def _save_index(self) -> None:
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump([r.meta() for r in self.refs.values()], fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.index_path)

    def add(self, image: np.ndarray, label: str, position: Optional[tuple[int, int, int]] = None,
            ref_id: Optional[str] = None) -> MinimapReference:
        if image is None or image.size == 0 or image.ndim != 3:
            raise ValueError("imagem de referência inválida")
        slug = re.sub(r"[^\w-]+", "_", label.strip().lower()).strip("_")[:30] or "ref"
        ref_id = ref_id or f"{slug}_{uuid.uuid4().hex[:6]}"
        ref = MinimapReference(ref_id, label, image.copy(),
                               tuple(int(v) for v in position) if position else None)
        if not cv2.imwrite(os.path.join(self.directory, f"{ref_id}.png"), ref.image):
            raise OSError("falha ao gravar a imagem da referência")
        self.refs[ref_id] = ref
        self._save_index()
        return ref

    def remove(self, ref_id: str) -> None:
        ref = self.refs.pop(ref_id, None)
        if ref is None:
            return
        try:
            os.remove(os.path.join(self.directory, f"{ref_id}.png"))
        except OSError:
            pass
        self._save_index()

    def update(self, ref_id: str, label: Optional[str] = None,
               position: Optional[tuple[int, int, int]] = None, clear_position: bool = False) -> None:
        ref = self.refs[ref_id]
        if label is not None:
            ref.label = label
        if clear_position:
            ref.position = None
        elif position is not None:
            ref.position = tuple(int(v) for v in position)
        self._save_index()


@dataclass(frozen=True)
class RefMatch:
    ref_id: str
    score: float
    offset_px: tuple[float, float]                       # personagem agora − centro da referência
    position: Optional[tuple[int, int, int]] = None      # só se a referência tem coordenadas


@dataclass
class Localization:
    status: LocStatus
    timestamp: float
    confidence: float = 0.0
    ref_id: Optional[str] = None
    offset_px: Optional[tuple[float, float]] = None
    position: Optional[tuple[int, int, int]] = None
    floor: Optional[int] = None
    floor_change_suspected: bool = False
    reasons: list[str] = field(default_factory=list)
    candidates: list[RefMatch] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == LocStatus.OK

    def describe(self) -> str:
        if not self.ok:
            return f"{self.status.value}: " + "; ".join(self.reasons)
        where = (f"posição {self.position[0]},{self.position[1]},{self.position[2]}" if self.position
                 else f"referência '{self.ref_id}' deslocado {self.offset_px[0]:+.0f},{self.offset_px[1]:+.0f}px")
        return f"{where} (confiança {self.confidence:.2f})"


def minimap_detail(img: np.ndarray) -> float:
    """Desvio-padrão do minimapa em tons de cinza: ~0 para recortes pretos/uniformes."""
    if img is None or img.size == 0:
        return 0.0
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


class MinimapLocalizer:
    """Estima a posição comparando o minimapa atual com as referências."""

    def __init__(self, library: ReferenceLibrary, region: Optional[Region] = None,
                 match_threshold: float = 0.80, ambiguity_margin: float = 0.05,
                 pixels_per_sqm: float = 1.0, patch_fraction: float = 0.6,
                 min_detail: float = 8.0, floor_change_diff: float = 45.0,
                 cross_fraction: float = 0.08):
        self.library = library
        self.region = region
        self.match_threshold = match_threshold
        self.ambiguity_margin = ambiguity_margin
        self.pixels_per_sqm = max(0.1, float(pixels_per_sqm))
        self.patch_fraction = min(0.95, max(0.2, patch_fraction))
        self.min_detail = min_detail
        self.floor_change_diff = floor_change_diff
        self.cross_fraction = cross_fraction
        self._prev: Optional[np.ndarray] = None
        self._last_floor: Optional[int] = None

    @classmethod
    def from_config(cls, cfg: dict, library: Optional[ReferenceLibrary] = None) -> "MinimapLocalizer":
        c = cfg["cavebot"]
        return cls(library or ReferenceLibrary(c["references_dir"]),
                   Region.from_any(cfg["regions"].get("minimap")),
                   c["match_threshold"], c["ambiguity_margin"], c.get("pixels_per_sqm", 1.0),
                   c.get("patch_fraction", 0.6), c.get("min_detail", 8.0), c["floor_change_diff"])

    # ------------------------------------------------------------------
    def crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.region is None:
            return None
        roi = self.region.crop(frame)
        return roi if roi.size else None

    def locate(self, frame: np.ndarray, t: Optional[float] = None) -> Localization:
        t = time.monotonic() if t is None else t
        mini = self.crop(frame)
        if mini is None:
            return Localization(LocStatus.NO_REGION, t, reasons=["região do minimapa não configurada "
                                                                 "ou fora do frame"])
        return self.locate_minimap(mini, t)

    def locate_minimap(self, mini: np.ndarray, t: Optional[float] = None) -> Localization:
        t = time.monotonic() if t is None else t
        floor_change = self._floor_change_check(mini)
        if not self.library.refs:
            return Localization(LocStatus.NO_REFERENCES, t, floor_change_suspected=floor_change,
                                reasons=["nenhuma referência de minimapa cadastrada"])
        detail = minimap_detail(mini)
        if detail < self.min_detail:
            return Localization(LocStatus.NO_DETAIL, t, floor_change_suspected=floor_change,
                                reasons=[f"minimapa uniforme/escuro (detalhe {detail:.1f} < {self.min_detail}); "
                                         "região errada, área inexplorada ou carregando"])
        patch, mask, (px, py) = self._central_patch(mini)
        matches: list[RefMatch] = []
        for ref in self.library.refs.values():
            m = self._match(ref, patch, mask, (px, py), mini.shape)
            if m is not None:
                matches.append(m)
        matches.sort(key=lambda m: m.score, reverse=True)
        loc = Localization(LocStatus.LOW_CONFIDENCE, t, floor_change_suspected=floor_change,
                           candidates=matches[:5])
        if not matches:
            loc.reasons.append("nenhuma referência compatível em tamanho com o minimapa atual")
            return loc
        best = matches[0]
        loc.confidence = best.score
        if best.score < self.match_threshold:
            loc.reasons.append(f"melhor referência '{best.ref_id}' com similaridade {best.score:.2f} "
                               f"< limiar {self.match_threshold:.2f}")
            if floor_change:
                loc.reasons.append("o minimapa mudou bruscamente (possível troca de andar)")
            return loc
        rival = next((m for m in matches[1:]
                      if best.score - m.score < self.ambiguity_margin and not self._agree(best, m)), None)
        if rival is not None:
            loc.status = LocStatus.AMBIGUOUS
            loc.reasons.append(f"'{best.ref_id}' ({best.score:.2f}) e '{rival.ref_id}' ({rival.score:.2f}) "
                               "são parecidas e indicam lugares diferentes")
            return loc

        ref = self.library.refs[best.ref_id]
        loc.status = LocStatus.OK
        loc.ref_id, loc.offset_px, loc.position = best.ref_id, best.offset_px, best.position
        loc.floor = ref.floor
        if loc.floor is not None and self._last_floor is not None and loc.floor != self._last_floor:
            loc.floor_change_suspected = True
            loc.reasons.append(f"andar mudou de {self._last_floor} para {loc.floor} (pela referência)")
        if loc.floor is not None:
            self._last_floor = loc.floor
        if ref.position is None:
            loc.reasons.append("referência sem coordenadas: posição absoluta não estimada")
        return loc

    def reset(self) -> None:
        self._prev = None
        self._last_floor = None

    # ------------------------------------------------------------------
    def _central_patch(self, mini: np.ndarray):
        h, w = mini.shape[:2]
        ph, pw = max(8, int(h * self.patch_fraction)), max(8, int(w * self.patch_fraction))
        py, px = (h - ph) // 2, (w - pw) // 2
        patch = mini[py:py + ph, px:px + pw]
        mask = np.full((ph, pw), 255, np.uint8)
        # Ignora a cruz do personagem (centro), que não pertence ao mapa.
        cr = max(2, int(min(h, w) * self.cross_fraction))
        cy, cx = h // 2 - py, w // 2 - px
        mask[max(0, cy - cr):cy + cr + 1, max(0, cx - cr):cx + cr + 1] = 0
        return patch, mask, (px, py)

    def _match(self, ref: MinimapReference, patch: np.ndarray, mask: np.ndarray,
               origin: tuple[int, int], mini_shape) -> Optional[RefMatch]:
        img = ref.image
        if img.shape[0] < patch.shape[0] or img.shape[1] < patch.shape[1]:
            return None
        res = cv2.matchTemplate(img, patch, cv2.TM_CCOEFF_NORMED, mask=mask)
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        _, score, _, (mx, my) = cv2.minMaxLoc(res)
        # Centro da referência = posição do personagem nela.
        rcx, rcy = img.shape[1] / 2.0, img.shape[0] / 2.0
        # Centro do minimapa atual, expresso em coordenadas da referência.
        ccx = mx + (mini_shape[1] / 2.0 - origin[0])
        ccy = my + (mini_shape[0] / 2.0 - origin[1])
        off = (round(ccx - rcx, 1), round(ccy - rcy, 1))
        pos = None
        if ref.position is not None:
            pos = (int(round(ref.position[0] + off[0] / self.pixels_per_sqm)),
                   int(round(ref.position[1] + off[1] / self.pixels_per_sqm)),
                   ref.position[2])
        return RefMatch(ref.id, float(min(1.0, max(-1.0, score))), off, pos)

    def _agree(self, a: RefMatch, b: RefMatch, tolerance_sqm: float = 2.0) -> bool:
        """Duas referências concordam se indicam o mesmo lugar (exige coordenadas)."""
        if a.position is None or b.position is None:
            return False
        return (a.position[2] == b.position[2]
                and max(abs(a.position[0] - b.position[0]), abs(a.position[1] - b.position[1])) <= tolerance_sqm)

    def _floor_change_check(self, mini: np.ndarray) -> bool:
        """Mudança grande que não se explica por um pequeno deslocamento do mapa."""
        gray = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY) if mini.ndim == 3 else mini
        prev, self._prev = self._prev, gray.copy()
        if prev is None or prev.shape != gray.shape:
            return False
        diff = float(np.mean(cv2.absdiff(prev, gray)))
        if diff < self.floor_change_diff:
            return False
        # Andar alguns sqm desloca o mapa: o centro do anterior aparece no atual.
        h, w = gray.shape
        ph, pw = int(h * 0.6), int(w * 0.6)
        patch = prev[(h - ph) // 2:(h - ph) // 2 + ph, (w - pw) // 2:(w - pw) // 2 + pw]
        if patch.std() < 1e-3:
            return True
        res = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)
        return float(np.nan_to_num(res).max()) < 0.6


class WaypointMatcher:
    """Decide se uma localização corresponde a um waypoint (com motivo)."""

    def __init__(self, library: ReferenceLibrary, pixels_per_sqm: float = 1.0):
        self.library = library
        self.pixels_per_sqm = max(0.1, float(pixels_per_sqm))

    def check(self, loc: Localization, wp: Waypoint) -> tuple[bool, str]:
        if not loc.ok:
            return False, f"posição indeterminada ({loc.status.value})"
        if not wp.reference_ids and wp.position is None:
            return False, "waypoint sem referência visual nem posição"
        radius_px = (wp.radius + 0.5) * self.pixels_per_sqm
        if loc.ref_id in wp.reference_ids:
            dx, dy = loc.offset_px or (0.0, 0.0)
            if max(abs(dx), abs(dy)) <= radius_px:
                return True, f"referência '{loc.ref_id}' reconhecida (confiança {loc.confidence:.2f})"
            return False, (f"na referência do waypoint, mas a {max(abs(dx), abs(dy)) / self.pixels_per_sqm:.1f} sqm "
                           f"(raio {wp.radius})")
        if wp.position is not None and loc.position is not None:
            if loc.position[2] != wp.position[2]:
                return False, f"andar {loc.position[2]} ≠ andar do waypoint {wp.position[2]}"
            dist = max(abs(loc.position[0] - wp.position[0]), abs(loc.position[1] - wp.position[1]))
            if dist <= wp.radius:
                return True, f"posição {loc.position} dentro do raio {wp.radius} (confiança {loc.confidence:.2f})"
            return False, f"a {dist} sqm do waypoint (raio {wp.radius})"
        if wp.position is not None:
            return False, "referência atual sem coordenadas; não dá para comparar com a posição do waypoint"
        return False, f"referência atual '{loc.ref_id}' não pertence a este waypoint"


def calibration_report(frame: np.ndarray, region: Optional[Region], min_detail: float = 8.0) -> list[str]:
    """Verificações simples para ajudar a calibrar a região do minimapa."""
    issues = []
    if region is None:
        return ["região do minimapa não definida"]
    fh, fw = frame.shape[:2]
    if region.x < 0 or region.y < 0 or region.x + region.w > fw or region.y + region.h > fh:
        issues.append(f"região {region} ultrapassa o frame {fw}x{fh}")
    mini = region.crop(frame)
    if mini.size == 0:
        return issues + ["recorte vazio"]
    if abs(mini.shape[0] - mini.shape[1]) > 0.25 * max(mini.shape[:2]):
        issues.append("recorte muito retangular; o minimapa do Tibia é aproximadamente quadrado")
    d = minimap_detail(mini)
    if d < min_detail:
        issues.append(f"pouco detalhe visual ({d:.1f}); verifique se a região cobre o minimapa")
    h, w = mini.shape[:2]
    c = mini[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3]
    if c.size and not np.any(np.all(c > 200, axis=-1)):
        issues.append("cruz branca do personagem não encontrada no centro; centralize a região no minimapa")
    return issues
