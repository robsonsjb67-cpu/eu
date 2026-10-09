"""Mapeador automático: monta um mapa grande juntando prints do minimapa.

Enquanto o personagem anda, cada print do minimapa (recortado do OBS) é
encaixado no mapa já montado:

1. a parte central do minimapa atual é procurada no mapa, perto da última
   posição conhecida (o minimapa do Tibia é pixel-art, então o encaixe é
   exato);
2. se a semelhança passa do limiar, os pixels conhecidos do minimapa são
   colados no mapa (áreas pretas/inexploradas e a cruz do personagem não
   sobrescrevem nada) e a posição do personagem no mapa é atualizada;
3. se não encaixa (teleporte, troca de andar, tela de loading), procura em
   todos os andares já mapeados; se continuar sem encaixe por alguns
   prints, começa um novo andar/segmento.

O mapa de cada andar é salvo em PNG (``maps/<nome>/<andar>.png``) e pode ser
usado como referência do CaveBot: com uma coordenada calibrada, a posição é
estimada em qualquer ponto mapeado. Nenhuma coordenada é inventada — sem
calibração o mapa só tem posições relativas (pixels).
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from logger import record_event

log = logging.getLogger(__name__)


@dataclass
class FloorMap:
    name: str
    canvas: np.ndarray                          # BGR; preto = desconhecido
    known: np.ndarray                           # bool, pixel já visto
    pos: tuple[int, int]                        # personagem no mapa (pixels)
    floor: Optional[int] = None                 # z, se o usuário informar
    # Calibração: pixel do mapa ↔ coordenada do jogo (x, y).
    anchor: Optional[tuple[int, int, int, int]] = None   # (px, py, x, y)
    frames: int = 0

    def shift(self, dx: int, dy: int) -> None:
        """Ajusta referências internas depois de crescer o mapa para cima/esquerda."""
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        if self.anchor:
            px, py, x, y = self.anchor
            self.anchor = (px + dx, py + dy, x, y)

    def world_at(self, px: float, py: float, pixels_per_sqm: float) -> Optional[tuple[int, int, int]]:
        if self.anchor is None or self.floor is None:
            return None
        ax, ay, x, y = self.anchor
        return (int(round(x + (px - ax) / pixels_per_sqm)), int(round(y + (py - ay) / pixels_per_sqm)),
                self.floor)

    def bbox(self) -> Optional[tuple[int, int, int, int]]:
        ys, xs = np.nonzero(self.known)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@dataclass
class MapUpdate:
    status: str          # started / added / relocated / lost / new_floor / no_detail / idle
    score: float = 0.0
    floor: Optional[str] = None
    pos: Optional[tuple[int, int]] = None
    reason: str = ""


@dataclass
class MapBuilder:
    pixels_per_sqm: float = 1.0
    match_threshold: float = 0.85
    search_margin: int = 40
    patch_fraction: float = 0.6
    min_detail: float = 8.0
    black_level: int = 12
    cross_fraction: float = 0.08
    new_floor_after: int = 5
    floors: dict[str, FloorMap] = field(default_factory=dict)
    current: Optional[str] = None
    active: bool = False
    _lost: int = 0
    _pending: Optional[tuple] = None

    @classmethod
    def from_config(cls, cfg: dict) -> "MapBuilder":
        m = cfg.get("map", {})
        return cls(pixels_per_sqm=cfg["cavebot"].get("pixels_per_sqm", 1.0),
                   match_threshold=m.get("match_threshold", 0.85),
                   search_margin=m.get("search_margin", 40),
                   min_detail=cfg["cavebot"].get("min_detail", 8.0),
                   new_floor_after=m.get("new_floor_after", 5))

    # ------------------------------------------------------------ controle
    def start(self) -> None:
        self.active = True
        self._lost = 0
        record_event("map", "start", "mapeamento iniciado: ande pelo local para montar o mapa")

    def stop(self) -> None:
        if self.active:
            self.active = False
            record_event("map", "stop", "mapeamento pausado")

    def new_floor(self, name: Optional[str] = None, floor: Optional[int] = None) -> str:
        """Força o próximo print a começar um andar/segmento novo."""
        name = name or self._next_name()
        self.current = None
        self._pending = (name, floor)
        return name

    @property
    def floor_map(self) -> Optional[FloorMap]:
        return self.floors.get(self.current) if self.current else None

    # ------------------------------------------------------------ prints
    def update(self, mini: Optional[np.ndarray]) -> MapUpdate:
        if not self.active:
            return MapUpdate("idle")
        if mini is None or mini.size == 0 or mini.ndim != 3:
            return MapUpdate("no_detail", reason="sem recorte do minimapa (defina a região)")
        gray = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < self.min_detail:
            return MapUpdate("no_detail", reason="minimapa escuro/uniforme (loading ou área inexplorada)")

        fm = self.floor_map
        if fm is None:
            return self._start_floor(mini)

        score, pos = self._match_near(fm, mini)
        if score >= self.match_threshold:
            self._lost = 0
            self._paste(fm, mini, pos)
            return MapUpdate("added", score, fm.name, fm.pos)

        # Não encaixou perto: teleporte, escada ou outro andar já mapeado?
        best = self._relocate(mini)
        if best is not None and best[0] >= self.match_threshold:
            sc, name, p = best
            self._lost = 0
            if name != self.current:
                record_event("map", "relocated", f"reencontrado no mapa '{name}' (semelhança {sc:.2f})")
            self.current = name
            self._paste(self.floors[name], mini, p)
            return MapUpdate("relocated", sc, name, self.floors[name].pos)

        self._lost += 1
        if self._lost >= self.new_floor_after:
            self._lost = 0
            self.current = None
            upd = self._start_floor(mini)
            upd.status = "new_floor"
            return upd
        return MapUpdate("lost", max(score, best[0] if best else 0.0), self.current,
                         reason=f"print não encaixa no mapa (semelhança {score:.2f}); "
                                f"novo andar em {self.new_floor_after - self._lost} print(s) se continuar")

    # ------------------------------------------------------------ internos
    def _next_name(self) -> str:
        i = len(self.floors) + 1
        while f"andar_{i}" in self.floors:
            i += 1
        return f"andar_{i}"

    def _start_floor(self, mini: np.ndarray) -> MapUpdate:
        name, floor = self._pending or (self._next_name(), None)
        self._pending = None
        h, w = mini.shape[:2]
        fm = FloorMap(name, np.zeros((h, w, 3), np.uint8), np.zeros((h, w), bool),
                      (w // 2, h // 2), floor)
        self.floors[name] = fm
        self.current = name
        self._paste(fm, mini, fm.pos)
        record_event("map", "floor", f"novo mapa '{name}' iniciado")
        return MapUpdate("started", 1.0, name, fm.pos)

    def _valid_mask(self, mini: np.ndarray) -> np.ndarray:
        """Pixels do minimapa que valem: nem pretos (inexplorado) nem a cruz do personagem."""
        valid = mini.max(axis=2) > self.black_level
        h, w = valid.shape
        cr = max(2, int(min(h, w) * self.cross_fraction))
        valid[max(0, h // 2 - cr):h // 2 + cr + 1, max(0, w // 2 - cr):w // 2 + cr + 1] = False
        return valid

    def _patch(self, mini: np.ndarray):
        h, w = mini.shape[:2]
        ph, pw = max(8, int(h * self.patch_fraction)), max(8, int(w * self.patch_fraction))
        py, px = (h - ph) // 2, (w - pw) // 2
        mask = self._valid_mask(mini)[py:py + ph, px:px + pw].astype(np.uint8) * 255
        return mini[py:py + ph, px:px + pw], mask, (px, py)

    def _search(self, fm: FloorMap, mini: np.ndarray, window: Optional[tuple[int, int, int, int]]):
        patch, mask, (px, py) = self._patch(mini)
        if cv2.countNonZero(mask) < 0.3 * mask.size:
            return 0.0, None
        H, W = fm.canvas.shape[:2]
        x0, y0, x1, y1 = window if window else (0, 0, W, H)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        area = fm.canvas[y0:y1, x0:x1]
        if area.shape[0] < patch.shape[0] or area.shape[1] < patch.shape[1]:
            return 0.0, None
        res = np.nan_to_num(cv2.matchTemplate(area, patch, cv2.TM_CCOEFF_NORMED, mask=mask),
                            nan=0.0, posinf=0.0, neginf=0.0)
        _, score, _, (mx, my) = cv2.minMaxLoc(res)
        h, w = mini.shape[:2]
        pos = (x0 + mx - px + w // 2, y0 + my - py + h // 2)
        return float(score), pos

    def _match_near(self, fm: FloorMap, mini: np.ndarray):
        h, w = mini.shape[:2]
        m = self.search_margin
        cx, cy = fm.pos
        return self._search(fm, mini, (cx - w // 2 - m, cy - h // 2 - m, cx + w // 2 + m + 1, cy + h // 2 + m + 1))

    def _relocate(self, mini: np.ndarray):
        best = None
        for name, fm in self.floors.items():
            score, pos = self._search(fm, mini, None)
            if pos is not None and (best is None or score > best[0]):
                best = (score, name, pos)
        return best

    def _paste(self, fm: FloorMap, mini: np.ndarray, pos: tuple[int, int]) -> None:
        h, w = mini.shape[:2]
        x0, y0 = pos[0] - w // 2, pos[1] - h // 2
        # Cresce o mapa se o print sair das bordas.
        H, W = fm.canvas.shape[:2]
        left, top = max(0, -x0), max(0, -y0)
        right, bottom = max(0, x0 + w - W), max(0, y0 + h - H)
        if left or top or right or bottom:
            pad = 64  # cresce em blocos para não realocar a cada passo
            left, top = (left + pad if left else 0), (top + pad if top else 0)
            right, bottom = (right + pad if right else 0), (bottom + pad if bottom else 0)
            fm.canvas = cv2.copyMakeBorder(fm.canvas, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            fm.known = np.pad(fm.known, ((top, bottom), (left, right)))
            fm.shift(left, top)
            x0, y0 = x0 + left, y0 + top
            pos = (pos[0] + left, pos[1] + top)
        valid = self._valid_mask(mini)
        region = fm.canvas[y0:y0 + h, x0:x0 + w]
        region[valid] = mini[valid]
        fm.known[y0:y0 + h, x0:x0 + w] |= valid
        fm.pos = pos
        fm.frames += 1

    # ------------------------------------------------------------ calibração
    def calibrate(self, x: int, y: int, z: int) -> None:
        """Informa a coordenada do jogo onde o personagem está agora."""
        fm = self.floor_map
        if fm is None:
            raise ValueError("nenhum mapa em andamento")
        fm.anchor = (fm.pos[0], fm.pos[1], int(x), int(y))
        fm.floor = int(z)
        record_event("map", "calibrated", f"mapa '{fm.name}' calibrado: posição atual = {x},{y},{z}")

    def current_world_position(self) -> Optional[tuple[int, int, int]]:
        fm = self.floor_map
        return fm.world_at(*fm.pos, self.pixels_per_sqm) if fm else None

    # ------------------------------------------------------------ saída
    def cropped(self, name: str) -> tuple[np.ndarray, tuple[int, int]]:
        """Mapa recortado na área conhecida e o deslocamento (x, y) do recorte."""
        fm = self.floors[name]
        bb = fm.bbox()
        if bb is None:
            return fm.canvas.copy(), (0, 0)
        x0, y0, x1, y1 = bb
        return fm.canvas[y0:y1, x0:x1].copy(), (x0, y0)

    def reference_for(self, name: str) -> tuple[np.ndarray, Optional[tuple[int, int, int]]]:
        """Imagem e coordenada do centro para cadastrar o mapa como referência do CaveBot.

        Com calibração, o recorte é centrado exatamente num pixel que corresponde a um
        sqm inteiro, para a posição estimada não sofrer erro de arredondamento.
        """
        fm = self.floors[name]
        bb = fm.bbox()
        if fm.anchor is None or fm.floor is None or bb is None:
            img, _ = self.cropped(name)
            return img, None
        x0, y0, x1, y1 = bb
        ax, ay, wx, wy = fm.anchor
        pps = self.pixels_per_sqm
        kx = round(((x0 + x1) / 2 - ax) / pps)
        ky = round(((y0 + y1) / 2 - ay) / pps)
        cx, cy = int(round(ax + kx * pps)), int(round(ay + ky * pps))
        hx, hy = max(cx - x0, x1 - cx), max(cy - y0, y1 - cy)
        img = np.zeros((2 * hy, 2 * hx, 3), np.uint8)
        H, W = fm.canvas.shape[:2]
        sx0, sy0, sx1, sy1 = max(0, cx - hx), max(0, cy - hy), min(W, cx + hx), min(H, cy + hy)
        img[sy0 - (cy - hy):sy1 - (cy - hy), sx0 - (cx - hx):sx1 - (cx - hx)] = fm.canvas[sy0:sy1, sx0:sx1]
        return img, (int(wx + kx), int(wy + ky), int(fm.floor))

    def save(self, directory: str) -> list[str]:
        os.makedirs(directory, exist_ok=True)
        meta, paths = [], []
        for name, fm in self.floors.items():
            fname = re.sub(r"[^\w-]+", "_", name) + ".png"
            path = os.path.join(directory, fname)
            # Canal alfa marca o que é desconhecido (transparente).
            rgba = np.dstack([fm.canvas, np.where(fm.known, 255, 0).astype(np.uint8)])
            if not cv2.imwrite(path, rgba):
                raise OSError(f"falha ao gravar {path}")
            paths.append(path)
            meta.append({"name": name, "file": fname, "pos": list(fm.pos), "floor": fm.floor,
                         "anchor": list(fm.anchor) if fm.anchor else None, "frames": fm.frames})
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"pixels_per_sqm": self.pixels_per_sqm, "current": self.current, "floors": meta},
                      fh, indent=2, ensure_ascii=False)
        record_event("map", "saved", f"mapa salvo em {directory} ({len(paths)} andar(es))")
        return paths

    def load(self, directory: str) -> None:
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        self.floors = {}
        for m in meta["floors"]:
            img = cv2.imread(os.path.join(directory, m["file"]), cv2.IMREAD_UNCHANGED)
            if img is None:
                log.warning("mapa %s sem imagem; ignorado", m["name"])
                continue
            known = img[..., 3] > 0 if img.shape[2] == 4 else img.max(axis=2) > self.black_level
            canvas = img[..., :3].copy()
            canvas[~known] = 0
            self.floors[m["name"]] = FloorMap(m["name"], canvas, known, tuple(m["pos"]), m.get("floor"),
                                              tuple(m["anchor"]) if m.get("anchor") else None,
                                              m.get("frames", 0))
        self.current = meta.get("current") if meta.get("current") in self.floors else None
        record_event("map", "loaded", f"mapa carregado de {directory} ({len(self.floors)} andar(es))")

    def preview(self, name: Optional[str] = None, marker: bool = True) -> Optional[np.ndarray]:
        """Imagem do mapa recortado com a posição atual marcada (para a interface)."""
        name = name or self.current
        if not name or name not in self.floors:
            return None
        img, (ox, oy) = self.cropped(name)
        img = np.ascontiguousarray(img)
        if marker and name == self.current:
            fm = self.floors[name]
            p = (fm.pos[0] - ox, fm.pos[1] - oy)
            cv2.circle(img, p, max(3, img.shape[1] // 150), (0, 0, 255), -1)
            cv2.circle(img, p, max(5, img.shape[1] // 100), (255, 255, 255), 1)
        return img


def build_from_frames(frames, crop, builder: Optional[MapBuilder] = None) -> tuple[MapBuilder, dict[str, int]]:
    """Monta o mapa a partir de prints gravados. ``crop(frame)`` devolve o minimapa."""
    b = builder or MapBuilder()
    b.start()
    counts: dict[str, int] = {}
    for img in frames:
        st = b.update(crop(img)).status
        counts[st] = counts.get(st, 0) + 1
    return b, counts
