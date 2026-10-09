"""Utilitários compartilhados: regiões de interesse, configuração e geometria."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

# Configuração movida para config.py; reexportada aqui por compatibilidade.
from config import CONFIG_PATH, DEFAULT_CONFIG, load_config, save_config  # noqa: E402,F401


@dataclass(frozen=True)
class Region:
    """Retângulo em coordenadas do frame capturado."""

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_any(cls, value: Any) -> Optional["Region"]:
        if value is None:
            return None
        if isinstance(value, Region):
            return value
        if isinstance(value, dict):
            return cls(int(value["x"]), int(value["y"]), int(value["w"]), int(value["h"]))
        x, y, w, h = value
        return cls(int(x), int(y), int(w), int(h))

    def crop(self, image: np.ndarray) -> np.ndarray:
        ih, iw = image.shape[:2]
        x0, y0 = max(0, self.x), max(0, self.y)
        x1, y1 = min(iw, self.x + self.w), min(ih, self.y + self.h)
        return image[y0:y1, x0:x1]

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @property
    def is_empty(self) -> bool:
        return self.w <= 0 or self.h <= 0


def select_region(image: np.ndarray, title: str) -> Optional[Region]:
    """Abre uma janela para o usuário arrastar um retângulo (ENTER confirma, C cancela)."""
    import cv2

    rect = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    region = Region(*map(int, rect))
    return None if region.is_empty else region


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU entre caixas (x, y, w, h)."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(boxes: Sequence[Sequence[float]], scores: Sequence[float], iou_threshold: float) -> list[int]:
    """Retorna os índices mantidos, do maior para o menor score."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for i in order:
        if all(iou(boxes[i], boxes[k]) <= iou_threshold for k in keep):
            keep.append(i)
    return keep


def normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    cleaned = " ".join("".join(c for c in name if c.isalnum() or c in " '-").split()).lower()
    return cleaned or None


def match_known_name(raw: Optional[str], known: Iterable[str], cutoff: float = 0.75) -> Optional[str]:
    """Corrige erros de OCR aproximando o texto lido dos nomes conhecidos."""
    import difflib

    norm = normalize_name(raw)
    if norm is None:
        return None
    known_norm = {normalize_name(k): k for k in known if normalize_name(k)}
    if norm in known_norm:
        return normalize_name(known_norm[norm])
    close = difflib.get_close_matches(norm, list(known_norm), n=1, cutoff=cutoff)
    return close[0] if close else norm


def now() -> float:
    return time.monotonic()
