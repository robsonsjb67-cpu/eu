"""Utilitários compartilhados: regiões de interesse, configuração e geometria."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

CONFIG_PATH = os.environ.get("EU_CONFIG", "config.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "capture": {
        # Índice da OBS Virtual Camera, ou URL/arquivo (ex.: "srt://127.0.0.1:9000").
        "source": 0,
        "width": 1920,
        "height": 1080,
        "buffer_size": 5,
        "disconnect_timeout": 2.0,
        "freeze_seconds": 3.0,
        "freeze_threshold": 0.4,
        "reconnect_delay": 1.0,
    },
    "regions": {
        "battle_list": None,
        "game_area": None,
    },
    "battle_list": {
        "bar_full_width": None,
        "min_bar_width": 1,
        "bar_min_height": 2,
        "bar_max_height": 6,
        "name_height": 12,
        "confirm_frames": 2,
        "leave_grace_seconds": 1.0,
        "ocr": True,
    },
    "detector": {
        "templates_dir": "templates",
        "threshold": 0.80,
        "nms_iou": 0.3,
        "scale": 1.0,
        "min_hits": 2,
        "max_missed_seconds": 0.8,
        "match_distance": 48,
    },
    "fusion": {
        "time_window": 2.0,
        "retain_gone_seconds": 10.0,
    },
}


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


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str = CONFIG_PATH) -> dict[str, Any]:
    if not os.path.exists(path):
        return _deep_merge(DEFAULT_CONFIG, {})
    with open(path, "r", encoding="utf-8") as fh:
        return _deep_merge(DEFAULT_CONFIG, json.load(fh))


def save_config(config: dict[str, Any], path: str = CONFIG_PATH) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


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
