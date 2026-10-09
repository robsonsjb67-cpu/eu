"""Configuração persistente (JSON) com valores padrão e mesclagem profunda.

O arquivo ``config.json`` (ou o caminho em ``EU_CONFIG``) guarda apenas o
que o usuário alterou; tudo o que faltar vem de ``DEFAULT_CONFIG``. A
gravação é atômica (arquivo temporário + ``os.replace``) para não corromper
a configuração se o programa fechar no meio da escrita.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get("EU_CONFIG", "config.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "capture": {
        # Índice da OBS Virtual Camera, ou URL/arquivo/pasta de imagens gravadas.
        "source": 0,
        "width": 1920,
        "height": 1080,
        "buffer_size": 5,
        "disconnect_timeout": 2.0,
        "freeze_seconds": 3.0,
        "freeze_threshold": 0.4,
        "reconnect_delay": 1.0,
        # Para pastas de imagens/vídeos gravados: quadros por segundo da reprodução.
        "playback_fps": 10.0,
        "loop_playback": False,
        # Pasta onde o botão "Print" e a gravação de frames salvam as imagens.
        "screenshots_dir": "prints",
    },
    "regions": {
        "battle_list": None,
        "game_area": None,
        "minimap": None,
        "hp_bar": None,
        "sio": None,
        "sio_name": None,
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
    "cavebot": {
        "routes_dir": "routes",
        "references_dir": "minimap_refs",
        "last_route": None,
        # Similaridade mínima (TM_CCOEFF_NORMED) para aceitar uma referência.
        "match_threshold": 0.80,
        # Diferença mínima entre a melhor e a segunda melhor referência.
        "ambiguity_margin": 0.05,
        # Frames seguidos reconhecendo o waypoint para considerá-lo concluído.
        "confirm_frames": 3,
        # Segundos sem posição confiável antes de pausar a análise.
        "lost_pause_seconds": 2.0,
        # Variação média (0–255) do minimapa entre frames que sugere troca de andar.
        "floor_change_diff": 45.0,
        "default_radius": 2,
        # Fração central do minimapa atual procurada nas referências.
        "patch_fraction": 0.6,
        # Pixels do minimapa (na imagem do OBS) por sqm; calibre pelo zoom usado.
        "pixels_per_sqm": 1.0,
        # Desvio-padrão mínimo do minimapa para tentar reconhecer (evita recortes pretos).
        "min_detail": 8.0,
    },
    "health": {
        "threshold_percent": 50.0,
        "hysteresis_percent": 5.0,
        "min_confidence": 0.7,
        "stale_seconds": 1.5,
        # Faixas HSV (OpenCV, H 0–179) da cor de preenchimento da barra de HP:
        # vermelho → amarelo → verde (a barra muda de cor conforme a vida cai).
        # Azul (mana) fica de fora.
        "fill_hsv_ranges": [[[0, 90, 70], [90, 255, 255]], [[165, 90, 70], [179, 255, 255]]],
        "column_fill_ratio": 0.5,
        "alert_sound": True,
        "alert_cooldown_seconds": 3.0,
        "smoothing": 0.5,
    },
    "sio": {
        "label": "Aliado",
        "threshold_percent": 60.0,
        "hysteresis_percent": 5.0,
        "min_confidence": 0.7,
        "stale_seconds": 1.5,
        # Barras da Party List / nome do aliado costumam ser verdes até ~60%.
        "fill_hsv_ranges": [[[0, 90, 70], [90, 255, 255]], [[165, 90, 70], [179, 255, 255]]],
        "column_fill_ratio": 0.5,
        # Imagem de referência opcional do nome do aliado (confirmação de identidade).
        "identity_template": None,
        "identity_threshold": 0.85,
        "alert_sound": True,
        "alert_cooldown_seconds": 3.0,
    },
    "spell_timers": {
        "timers": [
            {"name": "Utani Gran Hur", "seconds": 30.0, "warn_seconds": 5.0, "sound": True},
            {"name": "Utamo Vita", "seconds": 50.0, "warn_seconds": 5.0, "sound": True},
        ],
    },
    "ui": {
        "preview_fps": 20,
        "analysis_fps": 10,
        "geometry": None,
    },
    "logging": {
        "level": "INFO",
        "file": "logs/eu.log",
        "max_bytes": 2_000_000,
        "backup_count": 3,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Mescla ``override`` sobre ``base`` sem alterar nenhum dos dois."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str = CONFIG_PATH) -> dict[str, Any]:
    """Carrega a configuração; arquivo ausente ou corrompido cai nos padrões."""
    if not os.path.exists(path):
        return deep_merge(DEFAULT_CONFIG, {})
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("raiz do JSON não é um objeto")
    except (OSError, ValueError) as exc:
        backup = f"{path}.corrompido"
        log.error("config inválida (%s); usando padrões e copiando para %s", exc, backup)
        try:
            os.replace(path, backup)
        except OSError:
            pass
        return deep_merge(DEFAULT_CONFIG, {})
    return deep_merge(DEFAULT_CONFIG, data)


def save_config(config: dict[str, Any], path: str = CONFIG_PATH) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


class ConfigStore:
    """Configuração compartilhada entre módulos, com acesso por caminho e gravação."""

    def __init__(self, path: str = CONFIG_PATH):
        self.path = path
        self.data = load_config(path)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any, save: bool = True) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        if save:
            self.save()

    def save(self) -> None:
        try:
            save_config(self.data, self.path)
        except OSError:
            log.exception("falha ao salvar a configuração em %s", self.path)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]
