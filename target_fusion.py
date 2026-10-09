"""Fusão das detecções da Battle List com as detecções visuais.

Princípios:
  * A Battle List diz *quantas* criaturas de cada tipo existem por perto;
    a imagem diz *onde* há algo parecido com cada tipo. Uma linha da Battle
    List NÃO é tratada como correspondente a um monstro específico na tela:
    cada alvo lista os candidatos visuais compatíveis e, no máximo, uma
    sugestão explicitamente marcada como palpite.
  * Status:
      - CONFIRMED: entrada da Battle List com nome lido, e há criaturas
        visíveis desse tipo em número suficiente para cobri-la.
      - PROBABLE : há evidência parcial (nome sem imagem recente, ou imagem
        compatível surgindo no mesmo intervalo de tempo de uma entrada sem
        nome legível).
      - UNCERTAIN: só uma das fontes (ex.: detecção visual sem entrada na
        Battle List, ou entrada sem nome e sem imagem).
      - GONE     : saiu da Battle List e/ou sumiu da tela; mantido por um
        tempo apenas para inspeção.
  * Os IDs dos alvos são estáveis (derivados dos IDs temporários dos
    rastreadores), evitando duplicações.
  * Nada aqui executa ações: o resultado serve para inspeção.
"""
from __future__ import annotations

import enum
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from battle_attack import BattleTrack
from monster_detector import VisualTrack


class TargetStatus(enum.Enum):
    CONFIRMED = "confirmado"
    PROBABLE = "provável"
    UNCERTAIN = "incerto"
    GONE = "saiu"


@dataclass
class FusedTarget:
    target_id: int
    source: str                         # "battle" ou "visual"
    name: Optional[str]
    status: TargetStatus = TargetStatus.UNCERTAIN
    score: float = 0.0
    battle_track_id: Optional[int] = None
    hp_percent: Optional[float] = None
    candidate_visual_ids: list[int] = field(default_factory=list)
    suggested_visual_id: Optional[int] = None   # palpite, não uma correspondência garantida
    position: Optional[tuple[int, int]] = None  # posição do candidato sugerido / visual
    first_seen: float = 0.0
    updated_at: float = 0.0
    gone_at: Optional[float] = None
    evidence: list[str] = field(default_factory=list)
    status_history: list[tuple[float, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.target_id, "source": self.source, "name": self.name,
            "status": self.status.value, "score": round(self.score, 3),
            "battle_track_id": self.battle_track_id, "hp_percent": self.hp_percent,
            "candidate_visual_ids": list(self.candidate_visual_ids),
            "suggested_visual_id": self.suggested_visual_id, "position": self.position,
            "first_seen": self.first_seen, "updated_at": self.updated_at,
            "gone_at": self.gone_at, "evidence": list(self.evidence),
        }


class TargetFusion:
    def __init__(self, time_window: float = 2.0, retain_gone_seconds: float = 10.0,
                 visible_tolerance: float = 0.3, recent_visual_seconds: float = 3.0):
        self.time_window = time_window
        self.retain_gone_seconds = retain_gone_seconds
        self.visible_tolerance = visible_tolerance
        self.recent_visual_seconds = recent_visual_seconds
        self.targets: dict[int, FusedTarget] = {}
        self._by_battle: dict[int, int] = {}
        self._by_visual: dict[int, int] = {}
        self._ids = itertools.count(1)

    @classmethod
    def from_config(cls, cfg: dict) -> "TargetFusion":
        f = cfg["fusion"]
        return cls(f["time_window"], f["retain_gone_seconds"])

    # ------------------------------------------------------------------
    def update(self, battle_tracks: list[BattleTrack], visual_tracks: list[VisualTrack],
               t: float) -> list[FusedTarget]:
        battle = [b for b in battle_tracks if b.confirmed and b.active]
        visual_all = [v for v in visual_tracks if v.confirmed]
        visible = [v for v in visual_all if v.visible(t, self.visible_tolerance)]
        recent = [v for v in visual_all
                  if v.active or (v.lost_at is not None and t - v.lost_at <= self.recent_visual_seconds)]

        covered_visual: set[int] = set()
        seen_targets: set[int] = set()

        # --- 1. Entradas com nome: agrupa por tipo e compara contagens.
        named = defaultdict(list)
        unnamed = []
        for b in battle:
            (named[b.name] if b.name else unnamed).append(b)

        for name, entries in named.items():
            entries.sort(key=lambda b: b.first_seen)
            vis = sorted((v for v in visible if v.name == name), key=lambda v: v.first_seen)
            rec = [v for v in recent if v.name == name]
            suggestions = self._suggest(entries, vis)
            covered_visual.update(v.track_id for v in vis[: len(entries)])
            for rank, b in enumerate(entries):
                tgt = self._target_for_battle(b, t)
                seen_targets.add(tgt.target_id)
                tgt.name, tgt.hp_percent = name, b.hp_percent
                tgt.candidate_visual_ids = [v.track_id for v in vis]
                sug = suggestions.get(b.track_id)
                tgt.suggested_visual_id = sug.track_id if sug else None
                tgt.position = sug.center if sug else None
                ev = [f"Battle List: '{name}' (linha {b.row_index + 1}, vida {b.hp_percent:.0f}%)",
                      f"{len(entries)} entrada(s) '{name}' na lista, {len(vis)} visível(is) na tela"]
                if rank < len(vis):
                    conf = sum(v.confidence for v in vis) / len(vis)
                    ev.append(f"coberta por detecção visual (confiança média {conf:.2f}); "
                              "correspondência individual não é garantida")
                    self._set(tgt, TargetStatus.CONFIRMED, 0.6 + 0.4 * conf, ev, t)
                elif rec or vis:
                    ev.append("mais entradas do que criaturas visíveis; pode estar oculta, "
                              "fora da área de jogo ou sob efeito visual")
                    self._set(tgt, TargetStatus.PROBABLE, 0.5, ev, t)
                else:
                    ev.append("nenhuma detecção visual desse tipo (sem template ou fora da tela?)")
                    self._set(tgt, TargetStatus.PROBABLE, 0.4, ev, t)

        # --- 2. Entradas sem nome legível: evidência só por tempo.
        free_visual = [v for v in visible if v.track_id not in covered_visual]
        for b in unnamed:
            tgt = self._target_for_battle(b, t)
            seen_targets.add(tgt.target_id)
            tgt.hp_percent = b.hp_percent
            near = [v for v in free_visual if abs(v.first_seen - b.first_seen) <= self.time_window]
            tgt.candidate_visual_ids = [v.track_id for v in near]
            ev = [f"Battle List: nome ilegível (linha {b.row_index + 1}, vida {b.hp_percent:.0f}%)"]
            if near:
                best = min(near, key=lambda v: abs(v.first_seen - b.first_seen))
                tgt.name, tgt.suggested_visual_id, tgt.position = best.name, best.track_id, best.center
                names = sorted({v.name for v in near})
                ev.append(f"apareceu junto (±{self.time_window:.1f}s) com detecção visual: {', '.join(names)}")
                if len(names) == 1 and len(near) == 1:
                    covered_visual.add(best.track_id)
                self._set(tgt, TargetStatus.PROBABLE, 0.35 + 0.2 * best.confidence, ev, t)
            else:
                tgt.name = tgt.suggested_visual_id = tgt.position = None
                ev.append("sem evidência visual no mesmo intervalo")
                self._set(tgt, TargetStatus.UNCERTAIN, 0.2, ev, t)

        # --- 3. Detecções visuais sem entrada correspondente na Battle List.
        for v in visible:
            if v.track_id in covered_visual:
                self._retire_visual_only(v.track_id, t, "agora coberta por uma entrada da Battle List")
                continue
            tgt = self._target_for_visual(v, t)
            seen_targets.add(tgt.target_id)
            tgt.name, tgt.position = v.name, v.center
            tgt.candidate_visual_ids = [v.track_id]
            tgt.suggested_visual_id = v.track_id
            ev = [f"detecção visual '{v.name}' em {v.center} (confiança {v.confidence:.2f})",
                  "sem entrada correspondente na Battle List: pode ser falso positivo, "
                  "jogador/NPC parecido ou nome ainda não lido"]
            self._set(tgt, TargetStatus.UNCERTAIN, 0.3 * v.confidence, ev, t)

        # --- 4. Atualiza quem sumiu.
        for tgt in list(self.targets.values()):
            if tgt.target_id in seen_targets or tgt.status == TargetStatus.GONE:
                continue
            reason = ("saiu da Battle List" if tgt.source == "battle"
                      else "não é mais visto na tela")
            tgt.gone_at = t
            self._set(tgt, TargetStatus.GONE, 0.0, tgt.evidence + [reason], t)
        self._prune(t)
        return self.snapshot()

    # ------------------------------------------------------------------
    def snapshot(self, include_gone: bool = True) -> list[FusedTarget]:
        order = {TargetStatus.CONFIRMED: 0, TargetStatus.PROBABLE: 1,
                 TargetStatus.UNCERTAIN: 2, TargetStatus.GONE: 3}
        items = [x for x in self.targets.values() if include_gone or x.status != TargetStatus.GONE]
        return sorted(items, key=lambda x: (order[x.status], -x.score, x.target_id))

    def explain(self, target_id: int) -> str:
        tgt = self.targets.get(target_id)
        if tgt is None:
            return f"alvo {target_id} não existe (ou já foi descartado)"
        lines = [f"Alvo #{tgt.target_id} [{tgt.status.value}] nome={tgt.name or '?'} "
                 f"score={tgt.score:.2f} origem={tgt.source}"]
        if tgt.battle_track_id is not None:
            lines.append(f"  entrada da Battle List: #{tgt.battle_track_id} vida={tgt.hp_percent}")
        if tgt.candidate_visual_ids:
            lines.append(f"  candidatos visuais: {tgt.candidate_visual_ids} "
                         f"(sugestão: {tgt.suggested_visual_id}, apenas palpite)")
        lines += [f"  - {e}" for e in tgt.evidence]
        lines.append("  histórico: " + " → ".join(f"{s}@{ts:.1f}" for ts, s in tgt.status_history[-6:]))
        return "\n".join(lines)

    def format_table(self, include_gone: bool = False) -> str:
        rows = [f"{'ID':>4} {'STATUS':<11} {'NOME':<18} {'VIDA':>5} {'SCORE':>5}  CANDIDATOS"]
        for x in self.snapshot(include_gone):
            hp = f"{x.hp_percent:.0f}%" if x.hp_percent is not None else "-"
            rows.append(f"{x.target_id:>4} {x.status.value:<11} {(x.name or '?')[:18]:<18} {hp:>5} "
                        f"{x.score:>5.2f}  {x.candidate_visual_ids}")
        return "\n".join(rows)

    # ------------------------------------------------------------------
    def _suggest(self, entries: list[BattleTrack], vis: list[VisualTrack]) -> dict[int, VisualTrack]:
        """Palpite 1-para-1 por proximidade do momento de aparição."""
        pairs = sorted(((abs(b.first_seen - v.first_seen), b, v) for b in entries for v in vis),
                       key=lambda p: p[0])
        used_b, used_v, out = set(), set(), {}
        for _, b, v in pairs:
            if b.track_id in used_b or v.track_id in used_v:
                continue
            used_b.add(b.track_id)
            used_v.add(v.track_id)
            out[b.track_id] = v
        return out

    def _target_for_battle(self, b: BattleTrack, t: float) -> FusedTarget:
        tid = self._by_battle.get(b.track_id)
        if tid is None or tid not in self.targets:
            tid = next(self._ids)
            self._by_battle[b.track_id] = tid
            self.targets[tid] = FusedTarget(tid, "battle", b.name, battle_track_id=b.track_id,
                                            first_seen=b.first_seen, updated_at=t)
        return self.targets[tid]

    def _target_for_visual(self, v: VisualTrack, t: float) -> FusedTarget:
        tid = self._by_visual.get(v.track_id)
        if tid is None or tid not in self.targets or self.targets[tid].status == TargetStatus.GONE:
            tid = next(self._ids)
            self._by_visual[v.track_id] = tid
            self.targets[tid] = FusedTarget(tid, "visual", v.name, first_seen=v.first_seen, updated_at=t)
        return self.targets[tid]

    def _retire_visual_only(self, visual_id: int, t: float, reason: str) -> None:
        """Remove alvo só-visual que passou a ser coberto pela Battle List (evita duplicata)."""
        tid = self._by_visual.pop(visual_id, None)
        if tid is not None:
            self.targets.pop(tid, None)

    def _set(self, tgt: FusedTarget, status: TargetStatus, score: float, evidence: list[str], t: float) -> None:
        if tgt.status != status or not tgt.status_history:
            tgt.status_history.append((t, status.value))
            del tgt.status_history[:-50]
        tgt.status, tgt.score, tgt.evidence, tgt.updated_at = status, score, evidence, t
        if status != TargetStatus.GONE:
            tgt.gone_at = None

    def _prune(self, t: float) -> None:
        for tid in [k for k, x in self.targets.items()
                    if x.status == TargetStatus.GONE and x.gone_at is not None
                    and t - x.gone_at > self.retain_gone_seconds]:
            del self.targets[tid]
            for mapping in (self._by_battle, self._by_visual):
                for key in [k for k, v in mapping.items() if v == tid]:
                    del mapping[key]
