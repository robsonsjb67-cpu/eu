"""Gerador de imagens sintéticas no estilo do Tibia para os testes.

Simula gravações do OBS sem depender de capturas reais: mapa-múndi com
"tiles" coloridos, minimapa centrado no personagem (com a cruz branca),
barra de HP e um frame completo com regiões fixas.
"""
from __future__ import annotations

import cv2
import numpy as np

PPS = 2          # pixels por sqm no minimapa sintético
MINI = 106       # lado do minimapa (px)

# Regiões do frame sintético (x, y, w, h)
MINIMAP_REGION = (500, 10, MINI, MINI)
HP_REGION = (20, 10, 200, 12)
SIO_REGION = (20, 40, 120, 6)


def world_map(seed: int = 7, size_sqm: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = np.array([[40, 120, 40], [60, 60, 60], [150, 150, 150], [200, 120, 40],
                        [30, 80, 140], [0, 200, 255], [90, 40, 120]], np.uint8)
    tiles = rng.integers(0, len(palette), (size_sqm // 4, size_sqm // 4))
    small = palette[tiles]
    img = cv2.resize(small, (size_sqm * PPS, size_sqm * PPS), interpolation=cv2.INTER_NEAREST)
    # detalhes finos (paredes) para não haver regiões repetidas
    for _ in range(size_sqm * 2):
        x, y = rng.integers(0, size_sqm * PPS, 2)
        cv2.line(img, (int(x), int(y)), (int(x + rng.integers(-20, 20)), int(y + rng.integers(-20, 20))),
                 tuple(int(c) for c in palette[rng.integers(0, len(palette))]), 2)
    return img


def minimap_at(world: np.ndarray, sx: int, sy: int) -> np.ndarray:
    """Minimapa centrado no sqm (sx, sy) do mapa sintético, com a cruz do personagem."""
    cx, cy = sx * PPS + PPS // 2, sy * PPS + PPS // 2
    h = MINI // 2
    out = world[cy - h:cy - h + MINI, cx - h:cx - h + MINI].copy()
    c = MINI // 2
    out[c - 2:c + 3, c] = 255
    out[c, c - 2:c + 3] = 255
    return out


def hp_bar(percent: float, w: int = 200, h: int = 12, fill=(0, 0, 200)) -> np.ndarray:
    bar = np.full((h, w, 3), 25, np.uint8)
    cv2.rectangle(bar, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)
    inner = w - 2
    n = int(round(inner * percent / 100.0))
    bar[1:h - 1, 1:1 + n] = fill
    return bar


def frame(world=None, pos=None, hp=None, sio=None, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(20, 60, (360, 640, 3), dtype=np.uint8)
    if world is not None and pos is not None:
        x, y, w, h = MINIMAP_REGION
        img[y:y + h, x:x + w] = minimap_at(world, *pos)
    if hp is not None:
        x, y, w, h = HP_REGION
        img[y:y + h, x:x + w] = hp_bar(hp, w, h)
    if sio is not None:
        x, y, w, h = SIO_REGION
        img[y:y + h, x:x + w] = hp_bar(sio, w, h, fill=(0, 190, 0))
    return img
