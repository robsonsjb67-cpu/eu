"""Ponto de entrada.

Uso:
  python main.py                      # interface gráfica (padrão)
  python main.py gui [--no-connect]
  python main.py select-region minimap|battle_list|game_area|hp_bar|sio|sio_name
  python main.py add-template "Rat" [--image arquivo.png] [--threshold 0.85]
  python main.py run [--no-window]    # inspeção da Battle List/detecções em janela OpenCV
  python main.py observe --route "Minha rota" --recording pasta_ou_video
  python main.py analyze --recording pasta_ou_video   # HP/SIO/Battle sobre uma gravação
  python main.py build-map --recording pasta_ou_video --out maps/caverna  # mapa com os prints

Nada aqui envia teclas ou cliques ao jogo: os módulos observam a imagem do
OBS e exibem/registram o que foi reconhecido.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from battle_attack import BattleListReader, BattleListTracker
from monster_detector import MonsterDetector, TemplateLibrary, VisualTracker
from obs_capture import CaptureStatus, grabber_from_config
from logger import EVENTS, setup_logging
from target_fusion import TargetFusion, TargetStatus
from vision_common import Region, load_config, save_config, select_region

REGION_NAMES = ["minimap", "battle_list", "game_area", "hp_bar", "sio", "sio_name"]

STATUS_COLORS = {
    TargetStatus.CONFIRMED: (0, 200, 0),
    TargetStatus.PROBABLE: (0, 200, 255),
    TargetStatus.UNCERTAIN: (160, 160, 160),
    TargetStatus.GONE: (80, 80, 80),
}


def grab_one(cfg) -> np.ndarray:
    grabber = grabber_from_config(cfg).start()
    try:
        frame = grabber.wait_for_frame(timeout=10)
        if frame is None:
            sys.exit("Nenhum frame recebido do OBS. Verifique se a Virtual Camera/stream está ativa.")
        return frame.image
    finally:
        grabber.stop()


def cmd_select_region(args, cfg) -> None:
    image = grab_one(cfg)
    region = select_region(image, f"Selecione: {args.name} (ENTER confirma, C cancela)")
    if region is None:
        print("Seleção cancelada.")
        return
    cfg["regions"][args.name] = region.to_dict()
    save_config(cfg)
    print(f"Região '{args.name}' salva: {region}")


def cmd_add_template(args, cfg) -> None:
    lib = TemplateLibrary(cfg["detector"]["templates_dir"])
    if args.image:
        img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
        if img is None:
            sys.exit(f"Não foi possível abrir {args.image}")
        mask = img[..., 3] if img.ndim == 3 and img.shape[2] == 4 else None
        path = lib.add(args.name, img[..., :3] if img.ndim == 3 else img, mask, args.threshold)
    else:
        frame = grab_one(cfg)
        game = Region.from_any(cfg["regions"].get("game_area"))
        view = game.crop(frame) if game else frame
        rect = select_region(view, f"Recorte o monstro '{args.name}' (bem justo ao sprite)")
        if rect is None:
            print("Cancelado.")
            return
        path = lib.add(args.name, rect.crop(view).copy(), None, args.threshold)
    print(f"Template salvo em {path}. Dica: cadastre vários frames da animação e direções.")


def draw_overlay(frame, cfg, rows, visual_tracks, fusion: TargetFusion, status: CaptureStatus):
    out = frame.copy()
    for key, color in (("battle_list", (255, 200, 0)), ("game_area", (255, 0, 255))):
        r = Region.from_any(cfg["regions"].get(key))
        if r:
            cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), color, 1)
    bl = Region.from_any(cfg["regions"].get("battle_list"))
    if bl:
        for row in rows:
            x, y, w, h = row.bar_box
            cv2.rectangle(out, (bl.x + x, bl.y + y), (bl.x + x + w, bl.y + y + h), (255, 255, 255), 1)
    by_visual = {}
    for tgt in fusion.snapshot(include_gone=False):
        for vid in tgt.candidate_visual_ids:
            by_visual.setdefault(vid, tgt)
    for v in visual_tracks:
        tgt = by_visual.get(v.track_id)
        color = STATUS_COLORS[tgt.status] if tgt else (160, 160, 160)
        x, y, w, h = v.box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"v{v.track_id} {v.name} {v.confidence:.2f}" + (f" T{tgt.target_id}" if tgt else "")
        cv2.putText(out, label, (x, max(10, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(out, f"captura: {status.value}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if status == CaptureStatus.CONNECTED else (0, 0, 255), 1)
    return out


def cmd_run(args, cfg) -> None:
    lib = TemplateLibrary(cfg["detector"]["templates_dir"])
    reader = BattleListReader.from_config(cfg, known_names=lib.names)
    btracker = BattleListTracker.from_config(cfg)
    detector = MonsterDetector.from_config(cfg, lib)
    vtracker = VisualTracker.from_config(cfg)
    fusion = TargetFusion.from_config(cfg)
    grabber = grabber_from_config(cfg).start()
    print(f"Templates: {lib.names or 'nenhum'}; OCR: {'sim' if reader.ocr_enabled else 'não'}")
    print("Teclas: [i] inspecionar alvos  [p] pausar  [q] sair")

    last_index, paused, last_print = -1, False, 0.0
    try:
        while True:
            frame = grabber.wait_for_frame(last_index, timeout=1.0)
            status = grabber.status
            if frame is None or status != CaptureStatus.CONNECTED:
                print(f"\r[captura {status.value}] aguardando frames válidos...", end="", flush=True)
                if frame is None:
                    continue
            last_index = frame.index
            t = frame.timestamp
            rows = []
            if not paused and status == CaptureStatus.CONNECTED:
                if reader.region:
                    rows = reader.read(frame.image)
                    for ev in btracker.update(rows, t):
                        logging.info("battle: %s #%d %s hp=%.0f", ev.kind, ev.track_id, ev.name, ev.hp_percent)
                vtracker.update(detector.detect(frame.image, t), t)
                fusion.update(btracker.active_tracks(), list(vtracker.tracks.values()), t)
            if not args.no_window:
                cv2.imshow("eu - inspecao", draw_overlay(frame.image, cfg, rows,
                                                         vtracker.active_tracks(), fusion, status))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = not paused
                if key == ord("i"):
                    print("\n" + "\n\n".join(fusion.explain(x.target_id) for x in fusion.snapshot()))
            if time.monotonic() - last_print > 1.0:
                last_print = time.monotonic()
                print("\n" + fusion.format_table() +
                      f"\n(entradas únicas contadas: {btracker.total_confirmed})")
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        cv2.destroyAllWindows()


def cmd_gui(args, cfg) -> None:
    from config import ConfigStore
    from interface import run_gui

    sys.exit(run_gui(ConfigStore(), autoconnect=not getattr(args, "no_connect", False)))


def cmd_observe(args, cfg) -> None:
    """Testa uma rota com imagens gravadas e imprime o relatório."""
    from cave_navigation import MinimapLocalizer
    from cavebot import run_observation
    from obs_capture import iter_recording
    from route_manager import RouteError, RouteManager

    try:
        route = RouteManager(cfg["cavebot"]["routes_dir"]).load(args.route)
    except RouteError as exc:
        sys.exit(str(exc))
    for p in route.validate():
        print(f"aviso: {p}")
    localizer = MinimapLocalizer.from_config(cfg)
    c = cfg["cavebot"]
    report = run_observation(route, iter_recording(args.recording, args.step), localizer,
                             c["confirm_frames"], c["lost_pause_seconds"], fps=args.fps / args.step)
    print(report.summary())


def cmd_analyze(args, cfg) -> None:
    """Roda HP, SIO e Battle sobre uma gravação e imprime os eventos."""
    from analysis import AnalysisEngine
    from config import ConfigStore
    from obs_capture import iter_recording

    store = ConfigStore()
    engine = AnalysisEngine(store)
    engine.set_enabled("cavebot", False)
    EVENTS.subscribe(lambda ev: print(ev.format()))
    for i, img in enumerate(iter_recording(args.recording, args.step)):
        snap = engine.process(img, i * args.step / args.fps, i)
        if args.verbose and snap.health is not None:
            print(f"frame {i}: {snap.health.describe()}")


def cmd_build_map(args, cfg) -> None:
    """Monta o mapa juntando os prints do minimapa de uma gravação."""
    from map_builder import MapBuilder, build_from_frames
    from obs_capture import iter_recording

    region = Region.from_any(cfg["regions"].get("minimap"))
    if region is None:
        sys.exit("Defina a região do minimapa antes (python main.py select-region minimap).")
    builder = MapBuilder.from_config(cfg)
    builder, counts = build_from_frames(iter_recording(args.recording, args.step), region.crop, builder)
    print("Prints: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.calibrate:
        x, y, z = (int(v) for v in args.calibrate.split(","))
        builder.calibrate(x, y, z)
        print(f"Calibrado: último print = {x},{y},{z}")
    for path in builder.save(args.out):
        print(f"salvo: {path}")


def main(argv=None) -> None:
    cfg = load_config()
    setup_logging(cfg)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="interface gráfica (padrão)")
    g.add_argument("--no-connect", action="store_true", help="não conectar ao OBS ao abrir")
    s = sub.add_parser("select-region", help="selecionar região com janela OpenCV")
    s.add_argument("name", choices=REGION_NAMES)
    a = sub.add_parser("add-template", help="cadastrar referência visual de monstro")
    a.add_argument("name")
    a.add_argument("--image")
    a.add_argument("--threshold", type=float)
    r = sub.add_parser("run", help="inspeção da Battle List e detecções (OpenCV)")
    r.add_argument("--no-window", action="store_true")
    o = sub.add_parser("observe", help="testar rota com imagens gravadas")
    o.add_argument("--route", required=True)
    o.add_argument("--recording", required=True)
    o.add_argument("--fps", type=float, default=10.0, help="quadros por segundo da gravação")
    o.add_argument("--step", type=int, default=1, help="processar 1 a cada N quadros")
    z = sub.add_parser("analyze", help="HP/SIO/Battle sobre uma gravação")
    z.add_argument("--recording", required=True)
    z.add_argument("--fps", type=float, default=10.0)
    z.add_argument("--step", type=int, default=1)
    z.add_argument("--verbose", action="store_true")
    b = sub.add_parser("build-map", help="montar mapa com os prints do minimapa de uma gravação")
    b.add_argument("--recording", required=True)
    b.add_argument("--out", default="maps/mapa")
    b.add_argument("--step", type=int, default=1)
    b.add_argument("--calibrate", help="coordenada do ÚLTIMO print: x,y,z")
    args = p.parse_args(argv)
    handlers = {"gui": cmd_gui, "select-region": cmd_select_region, "add-template": cmd_add_template,
                "run": cmd_run, "observe": cmd_observe, "analyze": cmd_analyze,
                "build-map": cmd_build_map}
    try:
        handlers[args.cmd or "gui"](args, cfg)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
