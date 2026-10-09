"""CaveBot: acompanhamento visual de uma rota (modo de observação).

O CaveBot recebe a localização estimada pelo minimapa a cada frame e
acompanha o progresso da rota:

* um waypoint só é concluído quando é **reconhecido visualmente** em
  ``confirm_frames`` frames seguidos — nunca por tempo decorrido;
* sem posição confiável por ``lost_pause_seconds``, a análise entra em
  ``AGUARDANDO_POSICAO`` e registra o motivo; volta sozinha quando a
  referência é reencontrada;
* mudanças de andar inesperadas e waypoints reconhecidos fora de ordem são
  registrados como falhas, com o motivo, para o usuário recalibrar a rota.

Este módulo não envia teclas, cliques nem movimenta o personagem: ele
observa a imagem e informa onde o personagem está na rota.
"""
from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from cave_navigation import Localization, LocStatus, MinimapLocalizer, WaypointMatcher
from logger import record_event
from route_manager import Route, Waypoint

log = logging.getLogger(__name__)


class BotState(enum.Enum):
    STOPPED = "parado"
    RUNNING = "em execução"
    PAUSED = "pausado"
    WAITING_POSITION = "aguardando posição"
    FINISHED = "rota concluída"


@dataclass(frozen=True)
class WaypointHit:
    timestamp: float
    index: int
    waypoint_id: str
    name: str
    reason: str
    confidence: float
    lap: int


@dataclass(frozen=True)
class Failure:
    timestamp: float
    index: Optional[int]
    waypoint: Optional[str]
    reason: str


@dataclass
class CaveBotStatus:
    state: BotState
    route_name: Optional[str]
    current_index: Optional[int]
    current: Optional[Waypoint]
    next: Optional[Waypoint]
    completed: int
    total: int
    lap: int
    localization: Optional[Localization]
    last_reason: str

    @property
    def progress(self) -> float:
        return 0.0 if not self.total else self.completed / self.total


class CaveBot:
    def __init__(self, matcher: WaypointMatcher, confirm_frames: int = 3,
                 lost_pause_seconds: float = 2.0, clock: Callable[[], float] = time.monotonic):
        self.matcher = matcher
        self.confirm_frames = max(1, confirm_frames)
        self.lost_pause_seconds = lost_pause_seconds
        self.clock = clock
        self.route: Optional[Route] = None
        self.state = BotState.STOPPED
        self.index = 0
        self.lap = 0
        self.completed = 0
        self.hits: list[WaypointHit] = []
        self.failures: list[Failure] = []
        self.last_loc: Optional[Localization] = None
        self.last_reason = ""
        self._streak = 0
        self._lost_since: Optional[float] = None
        self._listeners: list[Callable[[str, dict], None]] = []

    @classmethod
    def from_config(cls, cfg: dict, localizer: MinimapLocalizer) -> "CaveBot":
        c = cfg["cavebot"]
        return cls(WaypointMatcher(localizer.library, c.get("pixels_per_sqm", 1.0)),
                   c["confirm_frames"], c["lost_pause_seconds"])

    # ------------------------------------------------------------ controles
    def load_route(self, route: Route) -> None:
        self.stop()
        self.route = route

    def start(self, route: Optional[Route] = None, start_index: int = 0) -> None:
        if route is not None:
            self.route = route
        if self.route is None or not self.route.waypoints:
            raise ValueError("carregue uma rota com waypoints antes de iniciar")
        problems = self.route.validate()
        for p in problems:
            self._fail(None, p)
        self.index = max(0, min(start_index, len(self.route.waypoints) - 1))
        self.lap, self.completed, self._streak, self._lost_since = 0, 0, 0, None
        self.hits.clear()
        self._set_state(BotState.RUNNING, f"rota '{self.route.name}' iniciada no waypoint #{self.index + 1}")

    def pause(self) -> None:
        if self.state in (BotState.RUNNING, BotState.WAITING_POSITION):
            self._set_state(BotState.PAUSED, "pausado pelo usuário")

    def resume(self) -> None:
        if self.state == BotState.PAUSED:
            self._streak, self._lost_since = 0, None
            self._set_state(BotState.RUNNING, "retomado pelo usuário")

    def stop(self) -> None:
        if self.state != BotState.STOPPED:
            self._set_state(BotState.STOPPED, "parado")
        self._streak, self._lost_since = 0, None

    def set_current(self, index: int) -> None:
        """Salta manualmente para um waypoint (ex.: depois de recalibrar)."""
        if self.route is None:
            return
        self.index = max(0, min(index, len(self.route.waypoints) - 1))
        self._streak = 0
        self._emit("jump", index=self.index)
        record_event("cavebot", "jump", f"waypoint atual definido manualmente: #{self.index + 1}")

    def add_listener(self, cb: Callable[[str, dict], None]) -> None:
        self._listeners.append(cb)

    # ------------------------------------------------------------ análise
    def update(self, loc: Localization) -> None:
        self.last_loc = loc
        if self.state not in (BotState.RUNNING, BotState.WAITING_POSITION) or self.route is None:
            return
        now = loc.timestamp
        wp = self.route.waypoints[self.index]

        if loc.floor_change_suspected and not wp.changes_floor:
            prev = self.route.waypoints[self.index - 1]  # índice -1 = último (rotas em loop)
            if not (prev.changes_floor and (self.index > 0 or self.route.loop)):
                self._fail(self.index, "possível mudança de andar inesperada "
                                       f"(waypoint atual '{wp.name}' não troca de andar)")

        if not loc.ok:
            self._streak = 0
            self._lost_since = self._lost_since if self._lost_since is not None else now
            self.last_reason = loc.describe()
            if (self.state == BotState.RUNNING
                    and now - self._lost_since >= self.lost_pause_seconds):
                self._set_state(BotState.WAITING_POSITION,
                                f"análise pausada: posição incerta há {now - self._lost_since:.1f}s — "
                                + loc.describe())
                self._fail(self.index, "referência perdida: " + loc.describe())
            return

        if self.state == BotState.WAITING_POSITION:
            self._set_state(BotState.RUNNING, "referência reencontrada: " + loc.describe())
        self._lost_since = None

        matched, reason = self.matcher.check(loc, wp)
        self.last_reason = reason
        if not matched:
            self._streak = 0
            self._check_out_of_order(loc)
            return
        self._streak += 1
        if self._streak < self.confirm_frames:
            return
        self._complete(wp, reason, loc.confidence, now)

    # ------------------------------------------------------------ estado
    def status(self) -> CaveBotStatus:
        wps = self.route.waypoints if self.route else []
        cur = wps[self.index] if wps and self.state != BotState.FINISHED else None
        nxt = None
        if wps and self.state != BotState.FINISHED:
            ni = self.index + 1
            if ni < len(wps):
                nxt = wps[ni]
            elif self.route and self.route.loop:
                nxt = wps[0]
        return CaveBotStatus(self.state, self.route.name if self.route else None,
                             self.index if wps else None, cur, nxt,
                             len(wps) if self.state == BotState.FINISHED else self.index,
                             len(wps), self.lap, self.last_loc, self.last_reason)

    # ------------------------------------------------------------ internos
    def _complete(self, wp: Waypoint, reason: str, conf: float, now: float) -> None:
        assert self.route is not None
        hit = WaypointHit(now, self.index, wp.id, wp.name, reason, conf, self.lap)
        self.hits.append(hit)
        self.completed += 1
        self._streak = 0
        record_event("cavebot", "waypoint_reached",
                     f"waypoint #{self.index + 1} '{wp.name}' reconhecido — {reason}",
                     index=self.index, waypoint=wp.name, confidence=conf)
        self._emit("waypoint_reached", hit=hit)
        if self.index + 1 < len(self.route.waypoints):
            self.index += 1
        elif self.route.loop:
            self.index = 0
            self.lap += 1
            record_event("cavebot", "lap", f"volta {self.lap} concluída; recomeçando a rota")
        else:
            self._set_state(BotState.FINISHED, f"rota '{self.route.name}' concluída")

    def _check_out_of_order(self, loc: Localization) -> None:
        assert self.route is not None
        n = len(self.route.waypoints)
        # Continuar parado no waypoint recém-concluído é normal; não é falha.
        just_done = (self.index - 1) % n if (self.index > 0 or self.lap > 0) else None
        for i, other in enumerate(self.route.waypoints):
            if i == self.index or i == just_done:
                continue
            ok, _ = self.matcher.check(loc, other)
            if ok:
                msg = (f"posição corresponde ao waypoint #{i + 1} '{other.name}', mas o esperado é "
                       f"#{self.index + 1} '{self.route.waypoints[self.index].name}'")
                if not self.failures or self.failures[-1].reason != msg:
                    self._fail(self.index, msg)
                return

    def _fail(self, index: Optional[int], reason: str) -> None:
        name = self.route.waypoints[index].name if (self.route and index is not None
                                                     and index < len(self.route.waypoints)) else None
        f = Failure(self.clock(), index, name, reason)
        self.failures.append(f)
        del self.failures[:-200]
        record_event("cavebot", "failure", reason, logging.WARNING, index=index, waypoint=name)
        self._emit("failure", failure=f)

    def _set_state(self, state: BotState, reason: str) -> None:
        old, self.state = self.state, state
        self.last_reason = reason
        record_event("cavebot", "state", f"{old.value} → {state.value}: {reason}")
        self._emit("state", old=old, new=state, reason=reason)

    def _emit(self, kind: str, **data) -> None:
        for cb in list(self._listeners):
            try:
                cb(kind, data)
            except Exception:
                log.exception("erro em listener do CaveBot")


# ----------------------------------------------------------------------
# Modo de observação com imagens gravadas
# ----------------------------------------------------------------------
@dataclass
class ObservationReport:
    route: str
    frames: int = 0
    located_frames: int = 0
    hits: list[WaypointHit] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=dict)
    final_state: str = ""
    final_index: Optional[int] = None

    def summary(self) -> str:
        lines = [f"Rota '{self.route}': {self.frames} frames, posição confiável em "
                 f"{self.located_frames} ({100 * self.located_frames / max(1, self.frames):.0f}%)",
                 f"Estado final: {self.final_state} (waypoint atual: "
                 f"{'-' if self.final_index is None else self.final_index + 1})",
                 "Leituras: " + ", ".join(f"{k}={v}" for k, v in sorted(self.status_counts.items()))]
        lines.append(f"Waypoints reconhecidos ({len(self.hits)}):")
        lines += [f"  t={h.timestamp:7.2f}s #{h.index + 1} {h.name} — {h.reason}" for h in self.hits]
        if self.failures:
            lines.append(f"Falhas ({len(self.failures)}):")
            lines += [f"  t={f.timestamp:7.2f}s #{'-' if f.index is None else f.index + 1} {f.reason}"
                      for f in self.failures]
        return "\n".join(lines)


def run_observation(route: Route, frames: Iterable[np.ndarray], localizer: MinimapLocalizer,
                    confirm_frames: int = 3, lost_pause_seconds: float = 2.0,
                    fps: float = 10.0, pixels_per_sqm: Optional[float] = None) -> ObservationReport:
    """Testa uma rota com imagens gravadas (frames completos do OBS), sem tempo real."""
    clock_t = [0.0]
    pps = pixels_per_sqm if pixels_per_sqm is not None else localizer.pixels_per_sqm
    bot = CaveBot(WaypointMatcher(localizer.library, pps), confirm_frames, lost_pause_seconds,
                  clock=lambda: clock_t[0])
    localizer.reset()
    bot.start(route)
    report = ObservationReport(route.name)
    for i, img in enumerate(frames):
        clock_t[0] = i / fps
        loc = localizer.locate(img, clock_t[0])
        report.frames += 1
        report.located_frames += int(loc.ok)
        report.status_counts[loc.status.value] = report.status_counts.get(loc.status.value, 0) + 1
        bot.update(loc)
        if bot.state == BotState.FINISHED:
            break
    report.hits, report.failures = list(bot.hits), list(bot.failures)
    report.final_state = bot.state.value
    report.final_index = bot.index
    return report


def failure_summary(loc_status: LocStatus) -> str:
    """Dica de correção para cada tipo de falha de leitura."""
    return {
        LocStatus.NO_REGION: "Selecione a região do minimapa na aba Captura.",
        LocStatus.NO_REFERENCES: "Cadastre referências do minimapa nos waypoints da rota.",
        LocStatus.NO_DETAIL: "Recalibre a região do minimapa (está escura/uniforme).",
        LocStatus.LOW_CONFIDENCE: "Cadastre uma referência perto deste ponto ou ajuste o limiar.",
        LocStatus.AMBIGUOUS: "Informe coordenadas nas referências parecidas ou remova duplicatas.",
        LocStatus.OK: "",
    }[loc_status]
