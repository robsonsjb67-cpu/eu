"""Detector visual de monstros por templates (OpenCV).

Compara imagens de referência cadastradas pelo usuário com a área de jogo
capturada pelo OBS. Organização dos templates::

    templates/
      rat/
        rat_01.png          # vários frames/direções da animação
        rat_02.png
        meta.json           # opcional: {"threshold": 0.85}
      cave_rat/...

PNGs com canal alfa usam a transparência como máscara, de modo que o chão
atrás do monstro não influencia a comparação. Isso reduz falsos negativos
com fundos diferentes e com efeitos visuais ao redor da criatura.

Tratamento de sobreposição, efeitos e animações:
  * várias imagens por monstro (frames da animação e direções);
  * máscara alfa e recortes parciais (ex.: só a metade de cima) para
    criaturas parcialmente cobertas;
  * supressão de não-máximos entre todos os monstros (uma mesma região não
    vira duas detecções);
  * rastreamento temporal: uma detecção só é confirmada após ``min_hits``
    observações e sobrevive a falhas curtas (efeito passando por cima).
"""
from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from vision_common import Region, non_max_suppression, normalize_name


@dataclass
class MonsterTemplate:
    name: str
    path: str
    image: np.ndarray                 # BGR
    mask: Optional[np.ndarray] = None  # uint8 0/255, mesmo tamanho
    threshold: Optional[float] = None

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


class TemplateLibrary:
    """Carrega e cadastra imagens de referência dos monstros."""

    def __init__(self, directory: str = "templates"):
        self.directory = directory
        self.templates: list[MonsterTemplate] = []
        self.reload()

    @property
    def names(self) -> list[str]:
        return sorted({t.name for t in self.templates})

    def reload(self) -> None:
        self.templates = []
        if not os.path.isdir(self.directory):
            return
        for entry in sorted(os.listdir(self.directory)):
            folder = os.path.join(self.directory, entry)
            if not os.path.isdir(folder):
                continue
            name = normalize_name(entry.replace("_", " "))
            threshold = None
            meta_path = os.path.join(folder, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
                threshold = meta.get("threshold")
                name = normalize_name(meta.get("name", name))
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".png", ".bmp", ".jpg", ".jpeg")):
                    tpl = self._load(os.path.join(folder, fname), name, threshold)
                    if tpl is not None:
                        self.templates.append(tpl)

    @staticmethod
    def _load(path: str, name: str, threshold: Optional[float]) -> Optional[MonsterTemplate]:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[0] < 4 or img.shape[1] < 4:
            return None
        mask = None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            alpha = img[..., 3]
            img = img[..., :3].copy()
            if alpha.min() < 255:
                mask = np.where(alpha > 127, 255, 0).astype(np.uint8)
                if cv2.countNonZero(mask) < 16:
                    return None
        return MonsterTemplate(name, path, img, mask, threshold)

    def add(self, name: str, image: np.ndarray, mask: Optional[np.ndarray] = None,
            threshold: Optional[float] = None) -> str:
        """Salva uma nova imagem de referência e recarrega a biblioteca."""
        norm = normalize_name(name)
        if not norm:
            raise ValueError("nome inválido")
        folder = os.path.join(self.directory, norm.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)
        out = image
        if mask is not None:
            out = np.dstack([image[..., :3], mask])
        path = os.path.join(folder, f"{norm.replace(' ', '_')}_{int(time.time() * 1000)}.png")
        cv2.imwrite(path, out)
        if threshold is not None:
            with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"name": norm, "threshold": threshold}, fh)
        self.reload()
        return path


@dataclass(frozen=True)
class Detection:
    name: str
    box: tuple[int, int, int, int]  # (x, y, w, h) em coordenadas do frame
    confidence: float
    timestamp: float
    template_path: str = ""

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return x + w // 2, y + h // 2


class MonsterDetector:
    """Procura os templates na área de jogo de um frame."""

    def __init__(self, library: TemplateLibrary, region: Optional[Region] = None,
                 threshold: float = 0.8, nms_iou: float = 0.3, scale: float = 1.0,
                 max_per_template: int = 30):
        self.library = library
        self.region = region
        self.threshold = threshold
        self.nms_iou = nms_iou
        self.scale = scale
        self.max_per_template = max_per_template

    @classmethod
    def from_config(cls, cfg: dict, library: Optional[TemplateLibrary] = None) -> "MonsterDetector":
        d = cfg["detector"]
        return cls(library or TemplateLibrary(d["templates_dir"]),
                   Region.from_any(cfg["regions"].get("game_area")),
                   d["threshold"], d["nms_iou"], d.get("scale", 1.0))

    def detect(self, frame: np.ndarray, t: Optional[float] = None) -> list[Detection]:
        t = time.monotonic() if t is None else t
        if self.region is not None:
            area = self.region.crop(frame)
            ox, oy = max(0, self.region.x), max(0, self.region.y)
        else:
            area, ox, oy = frame, 0, 0
        if area.size == 0 or not self.library.templates:
            return []
        s = self.scale
        work = cv2.resize(area, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1.0 else area

        boxes, scores, meta = [], [], []
        for tpl in self.library.templates:
            img, mask = tpl.image, tpl.mask
            if s != 1.0:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                if mask is not None:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            th, tw = img.shape[:2]
            if th > work.shape[0] or tw > work.shape[1]:
                continue
            res = cv2.matchTemplate(work, img, cv2.TM_CCOEFF_NORMED, mask=mask)
            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            thr = tpl.threshold if tpl.threshold is not None else self.threshold
            for (y, x), score in self._peaks(res, thr, (tw, th)):
                boxes.append((ox + x / s, oy + y / s, tw / s, th / s))
                scores.append(float(score))
                meta.append(tpl)

        keep = non_max_suppression(boxes, scores, self.nms_iou)
        out = []
        for i in keep:
            x, y, w, h = boxes[i]
            out.append(Detection(meta[i].name, (int(round(x)), int(round(y)), int(round(w)), int(round(h))),
                                 round(scores[i], 4), t, meta[i].path))
        return out

    def _peaks(self, res: np.ndarray, threshold: float, size: tuple[int, int]):
        """Máximos locais acima do limiar (um por vizinhança do tamanho do template)."""
        tw, th = size
        kernel = np.ones((max(3, th // 2) | 1, max(3, tw // 2) | 1), np.uint8)
        local_max = cv2.dilate(res, kernel)
        ys, xs = np.where((res >= threshold) & (res >= local_max - 1e-6))
        if len(ys) == 0:
            return []
        order = np.argsort(res[ys, xs])[::-1][: self.max_per_template]
        return [((int(ys[i]), int(xs[i])), res[ys[i], xs[i]]) for i in order]


# --------------------------------------------------------------------------
# Rastreamento temporal das detecções
# --------------------------------------------------------------------------
@dataclass
class VisualTrack:
    track_id: int
    name: str
    box: tuple[int, int, int, int]
    confidence: float          # média exponencial
    first_seen: float
    last_seen: float
    hits: int = 1
    confirmed: bool = False
    active: bool = True
    lost_at: Optional[float] = None
    history: list[tuple[float, tuple[int, int], float]] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return x + w // 2, y + h // 2

    def visible(self, t: float, tolerance: float = 0.25) -> bool:
        return self.active and (t - self.last_seen) <= tolerance


class VisualTracker:
    """Dá IDs temporários às detecções e as estabiliza no tempo."""

    def __init__(self, min_hits: int = 2, max_missed_seconds: float = 0.8,
                 match_distance: float = 48, ema: float = 0.5):
        self.min_hits = max(1, min_hits)
        self.max_missed_seconds = max_missed_seconds
        self.match_distance = match_distance
        self.ema = ema
        self.tracks: dict[int, VisualTrack] = {}
        self._ids = itertools.count(1)

    @classmethod
    def from_config(cls, cfg: dict) -> "VisualTracker":
        d = cfg["detector"]
        return cls(d["min_hits"], d["max_missed_seconds"], d["match_distance"])

    def active_tracks(self, confirmed_only: bool = True) -> list[VisualTrack]:
        return [tr for tr in self.tracks.values() if tr.active and (tr.confirmed or not confirmed_only)]

    def update(self, detections: list[Detection], t: float) -> list[tuple[str, VisualTrack]]:
        events: list[tuple[str, VisualTrack]] = []
        options = []
        live = [tr for tr in self.tracks.values() if tr.active]
        for di, det in enumerate(detections):
            for tr in live:
                if tr.name != det.name:
                    continue
                dx, dy = det.center[0] - tr.center[0], det.center[1] - tr.center[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= self.match_distance:
                    options.append((dist, di, tr))
        options.sort(key=lambda o: o[0])
        used_d, used_t = set(), set()
        for dist, di, tr in options:
            if di in used_d or tr.track_id in used_t:
                continue
            used_d.add(di)
            used_t.add(tr.track_id)
            det = detections[di]
            tr.box = det.box
            tr.last_seen = t
            tr.hits += 1
            tr.confidence = self.ema * det.confidence + (1 - self.ema) * tr.confidence
            tr.history.append((t, det.center, det.confidence))
            del tr.history[:-100]
            if not tr.confirmed and tr.hits >= self.min_hits:
                tr.confirmed = True
                events.append(("appeared", tr))

        for di, det in enumerate(detections):
            if di in used_d:
                continue
            tr = VisualTrack(next(self._ids), det.name, det.box, det.confidence, t, t,
                             history=[(t, det.center, det.confidence)])
            self.tracks[tr.track_id] = tr
            if self.min_hits <= 1:
                tr.confirmed = True
                events.append(("appeared", tr))

        for tr in list(self.tracks.values()):
            if not tr.active or tr.last_seen == t:
                continue
            if t - tr.last_seen > self.max_missed_seconds:
                if tr.confirmed:
                    tr.active = False
                    tr.lost_at = tr.last_seen
                    events.append(("disappeared", tr))
                else:
                    del self.tracks[tr.track_id]
        for tid in [k for k, tr in self.tracks.items()
                    if not tr.active and tr.lost_at is not None and t - tr.lost_at > 60]:
            del self.tracks[tid]
        return events
