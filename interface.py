"""Interface gráfica (PySide6).

Organização:
  * à esquerda, a prévia do OBS em tempo real, com as regiões desenhadas e
    seleção de regiões arrastando o mouse;
  * à direita, abas: Regiões, CaveBot, Battle, Cura/Suporte, Magias, Logs;
  * no topo, a fonte de captura, o estado da captura, o banner de alertas e
    o botão de PARADA DE EMERGÊNCIA (atalho: F12).

Threads:
  * a captura roda na thread do ``FrameGrabber`` (obs_capture.py);
  * a análise roda num ``QThread`` (``AnalysisWorker``) e entrega um
    ``Snapshot`` por sinal;
  * a prévia é atualizada por um ``QTimer`` que só lê o frame mais recente —
    a interface nunca espera pela captura nem pela análise.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPoint, QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from analysis import AnalysisEngine, Snapshot
from cave_navigation import calibration_report
from cavebot import BotState, failure_summary, run_observation
from config import ConfigStore
from logger import EVENTS, Event
from obs_capture import CaptureStatus, FrameGrabber, grabber_from_config, iter_recording
from route_manager import WAYPOINT_KINDS, Route, RouteError, RouteManager, Waypoint
from spell_timers import SpellTimer, TimerState
from target_fusion import TargetStatus
from vision_common import Region

log = logging.getLogger(__name__)

REGION_LABELS = {
    "minimap": ("Minimapa", QColor(0, 200, 255)),
    "battle_list": ("Battle List", QColor(255, 200, 0)),
    "game_area": ("Área de jogo", QColor(255, 0, 255)),
    "hp_bar": ("Barra de HP", QColor(255, 60, 60)),
    "sio": ("Barra do aliado (SIO)", QColor(60, 255, 60)),
    "sio_name": ("Nome do aliado (SIO)", QColor(150, 255, 150)),
}

STATUS_COLORS = {
    CaptureStatus.CONNECTED: "#2e7d32", CaptureStatus.FROZEN: "#ef6c00",
    CaptureStatus.DISCONNECTED: "#c62828", CaptureStatus.STARTING: "#616161",
    CaptureStatus.STOPPED: "#616161",
}


def to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888).copy()


# ----------------------------------------------------------------------
# Pontes thread → interface
# ----------------------------------------------------------------------
class Bridge(QObject):
    event = Signal(object)        # logger.Event
    alert = Signal(str, str)      # nível, mensagem
    report = Signal(str)          # relatório do modo de observação


class AnalysisWorker(QThread):
    snapshot = Signal(object)

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def run(self) -> None:
        last_index, last_run = -1, 0.0
        while self._running:
            grabber = self.window.grabber
            if grabber is None:
                self.msleep(100)
                continue
            frame = grabber.wait_for_frame(last_index, timeout=0.5)
            status = grabber.status
            if frame is None:
                continue
            fps = max(1, int(self.window.store.get("ui.analysis_fps", 10)))
            wait = 1.0 / fps - (time.monotonic() - last_run)
            if wait > 0:
                self.msleep(int(wait * 1000))
                continue
            last_run = time.monotonic()
            last_index = frame.index
            try:
                snap = self.window.engine.process(frame.image, frame.timestamp, frame.index, status)
            except Exception:
                log.exception("falha na análise do frame")
                continue
            self.snapshot.emit(snap)


# ----------------------------------------------------------------------
# Prévia com seleção de região
# ----------------------------------------------------------------------
class PreviewLabel(QLabel):
    regionSelected = Signal(str, object)   # nome, Region

    def __init__(self):
        super().__init__("Sem imagem do OBS.\nConfigure a fonte e clique em Conectar.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setStyleSheet("background:#111; color:#aaa;")
        self.frame_size: Optional[tuple[int, int]] = None
        self._target_rect = QRect()
        self.selecting: Optional[str] = None
        self._drag_start: Optional[QPoint] = None
        self._drag_now: Optional[QPoint] = None

    def show_frame(self, image: QImage) -> None:
        self.frame_size = (image.width(), image.height())
        pix = QPixmap.fromImage(image).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.FastTransformation)
        x = (self.width() - pix.width()) // 2
        y = (self.height() - pix.height()) // 2
        self._target_rect = QRect(x, y, pix.width(), pix.height())
        if self._drag_start is not None and self._drag_now is not None:
            p = QPainter(pix)
            p.setPen(QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine))
            p.drawRect(QRect(self._drag_start, self._drag_now).normalized().translated(-x, -y))
            p.end()
        self.setPixmap(pix)

    def begin_selection(self, name: str) -> None:
        self.selecting = name
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _to_frame(self, p: QPoint) -> tuple[int, int]:
        r = self._target_rect
        fw, fh = self.frame_size or (1, 1)
        fx = (p.x() - r.x()) * fw / max(1, r.width())
        fy = (p.y() - r.y()) * fh / max(1, r.height())
        return int(min(max(0, fx), fw)), int(min(max(0, fy), fh))

    def mousePressEvent(self, ev):  # noqa: N802
        if self.selecting and self.frame_size and ev.button() == Qt.MouseButton.LeftButton:
            self._drag_start = self._drag_now = ev.position().toPoint()
        elif ev.button() == Qt.MouseButton.RightButton:
            self._cancel()

    def mouseMoveEvent(self, ev):  # noqa: N802
        if self._drag_start is not None:
            self._drag_now = ev.position().toPoint()

    def mouseReleaseEvent(self, ev):  # noqa: N802
        if self._drag_start is None or not self.selecting:
            return
        (x0, y0), (x1, y1) = self._to_frame(self._drag_start), self._to_frame(ev.position().toPoint())
        name = self.selecting
        self._cancel()
        region = Region(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        if region.w >= 3 and region.h >= 3:
            self.regionSelected.emit(name, region)

    def _cancel(self) -> None:
        self.selecting = None
        self._drag_start = self._drag_now = None
        self.unsetCursor()


# ----------------------------------------------------------------------
# Diálogo de waypoint
# ----------------------------------------------------------------------
class WaypointDialog(QDialog):
    def __init__(self, parent, wp: Optional[Waypoint] = None, default_radius: int = 2):
        super().__init__(parent)
        self.setWindowTitle("Waypoint")
        form = QFormLayout(self)
        self.name = QLineEdit(wp.name if wp else "")
        self.kind = QComboBox()
        for key, label in WAYPOINT_KINDS.items():
            self.kind.addItem(label, key)
        if wp:
            self.kind.setCurrentIndex(list(WAYPOINT_KINDS).index(wp.kind))
        self.has_pos = QCheckBox("Informar coordenadas (x, y, z)")
        self.x, self.y, self.z = QSpinBox(), QSpinBox(), QSpinBox()
        for s in (self.x, self.y):
            s.setRange(0, 70000)
        self.z.setRange(0, 15)
        self.z.setValue(7)
        if wp and wp.position:
            self.has_pos.setChecked(True)
            self.x.setValue(wp.position[0])
            self.y.setValue(wp.position[1])
            self.z.setValue(wp.position[2])
        pos_row = QHBoxLayout()
        for s in (self.x, self.y, self.z):
            pos_row.addWidget(s)
        self.radius = QSpinBox()
        self.radius.setRange(0, 20)
        self.radius.setValue(wp.radius if wp else default_radius)
        self.notes = QLineEdit(wp.notes if wp else "")
        self.refs = QLabel(", ".join(wp.reference_ids) if wp and wp.reference_ids else "(nenhuma)")
        self.refs.setWordWrap(True)
        form.addRow("Nome", self.name)
        form.addRow("Tipo", self.kind)
        form.addRow(self.has_pos)
        form.addRow("Posição", pos_row)
        form.addRow("Raio (sqm)", self.radius)
        form.addRow("Notas", self.notes)
        form.addRow("Referências", self.refs)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self._wp = wp

    def result_waypoint(self) -> Waypoint:
        pos = (self.x.value(), self.y.value(), self.z.value()) if self.has_pos.isChecked() else None
        kw = dict(name=self.name.text().strip(), kind=self.kind.currentData(), position=pos,
                  radius=self.radius.value(), notes=self.notes.text(),
                  reference_ids=list(self._wp.reference_ids) if self._wp else [])
        if self._wp:
            kw["id"] = self._wp.id
        return Waypoint(**kw)


# ----------------------------------------------------------------------
# Janela principal
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, store: ConfigStore):
        super().__init__()
        self.store = store
        self.setWindowTitle("eu — análise visual do Tibia via OBS")
        self.engine = AnalysisEngine(store)
        self.routes = RouteManager(store.get("cavebot.routes_dir", "routes"))
        self.route: Optional[Route] = None
        self.grabber: Optional[FrameGrabber] = None
        self.last_snapshot: Optional[Snapshot] = None
        self._preview_index = -1
        self._fps_frames: list[float] = []
        self._timer_widgets: dict[str, dict] = {}

        self.bridge = Bridge()
        self.bridge.event.connect(self._on_event)
        self.bridge.alert.connect(self._on_alert)
        self.bridge.report.connect(self._on_report)
        EVENTS.subscribe(self.bridge.event.emit)
        self._wire_alerts()

        self._build_ui()
        self._load_routes_combo()
        last = store.get("cavebot.last_route")
        if last and last in self.routes.list_routes():
            self.route_combo.setCurrentText(last)
            self._load_route(last)

        self.worker = AnalysisWorker(self)
        self.worker.snapshot.connect(self._on_snapshot)
        self.worker.start()

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_preview)
        self.preview_timer.start(int(1000 / max(1, store.get("ui.preview_fps", 20))))
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._tick_ui)
        self.ui_timer.start(100)

        geo = store.get("ui.geometry")
        if geo:
            self.setGeometry(*geo)
        else:
            self.resize(1400, 850)

    # ------------------------------------------------------------ construção
    def _build_ui(self) -> None:
        root = QWidget()
        lay = QVBoxLayout(root)

        top = QHBoxLayout()
        top.addWidget(QLabel("Fonte OBS:"))
        self.source_edit = QComboBox()
        self.source_edit.setEditable(True)
        self.source_edit.addItems(["0", "1", "2", "srt://127.0.0.1:9000"])
        self.source_edit.setCurrentText(str(self.store.get("capture.source", 0)))
        self.source_edit.setMinimumWidth(220)
        self.source_edit.setToolTip("Índice da OBS Virtual Camera, URL de stream, vídeo ou pasta de imagens")
        top.addWidget(self.source_edit)
        btn = QPushButton("Conectar")
        btn.clicked.connect(self.connect_capture)
        top.addWidget(btn)
        btn = QPushButton("Abrir gravação…")
        btn.clicked.connect(self._open_recording)
        top.addWidget(btn)
        btn = QPushButton("Desconectar")
        btn.clicked.connect(self.disconnect_capture)
        top.addWidget(btn)
        self.capture_label = QLabel("captura: parada")
        self.capture_label.setStyleSheet("padding:4px; color:white; background:#616161;")
        top.addWidget(self.capture_label)
        self.fps_label = QLabel("")
        top.addWidget(self.fps_label)
        top.addStretch(1)
        self.release_btn = QPushButton("Liberar análise")
        self.release_btn.setEnabled(False)
        self.release_btn.clicked.connect(self._release)
        top.addWidget(self.release_btn)
        self.stop_btn = QPushButton("PARADA DE EMERGÊNCIA (F12)")
        self.stop_btn.setStyleSheet("background:#b71c1c; color:white; font-weight:bold; padding:8px 16px;")
        self.stop_btn.clicked.connect(self.emergency_stop)
        top.addWidget(self.stop_btn)
        lay.addLayout(top)
        QShortcut(QKeySequence("F12"), self, activated=self.emergency_stop)

        self.alert_banner = QLabel("")
        self.alert_banner.setVisible(False)
        self.alert_banner.setStyleSheet("background:#c62828; color:white; font-size:16px; padding:6px;")
        lay.addWidget(self.alert_banner)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.preview = PreviewLabel()
        self.preview.regionSelected.connect(self._on_region_selected)
        split.addWidget(self.preview)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_regions_tab(), "Regiões")
        self.tabs.addTab(self._build_cavebot_tab(), "CaveBot")
        self.tabs.addTab(self._build_battle_tab(), "Battle")
        self.tabs.addTab(self._build_health_tab(), "Cura / Suporte")
        self.tabs.addTab(self._build_timers_tab(), "Magias")
        self.tabs.addTab(self._build_logs_tab(), "Logs")
        split.addWidget(self.tabs)
        split.setSizes([800, 600])
        lay.addWidget(split, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Pronto. Nenhum módulo envia teclas ou cliques ao jogo.")

    def _build_regions_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Clique em um botão e arraste sobre a prévia para marcar a região "
                             "(botão direito cancela). As regiões ficam salvas na configuração."))
        grid = QGridLayout()
        self.region_labels: dict[str, QLabel] = {}
        for i, (key, (label, color)) in enumerate(REGION_LABELS.items()):
            b = QPushButton(f"Selecionar {label}")
            b.clicked.connect(lambda _=False, k=key: self._begin_select(k))
            sw = QLabel("■")
            sw.setStyleSheet(f"color:{color.name()}; font-size:18px;")
            val = QLabel()
            self.region_labels[key] = val
            clear = QPushButton("Limpar")
            clear.clicked.connect(lambda _=False, k=key: self._clear_region(k))
            grid.addWidget(sw, i, 0)
            grid.addWidget(b, i, 1)
            grid.addWidget(val, i, 2)
            grid.addWidget(clear, i, 3)
        lay.addLayout(grid)
        self._refresh_region_labels()
        b = QPushButton("Verificar calibração do minimapa")
        b.clicked.connect(self._check_minimap_calibration)
        lay.addWidget(b)
        g = QGroupBox("Módulos de análise (independentes)")
        gl = QHBoxLayout(g)
        for m, label in (("cavebot", "CaveBot"), ("battle", "Battle"), ("health", "HP"), ("sio", "SIO")):
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(lambda on, mm=m: self.engine.set_enabled(mm, on))
            gl.addWidget(cb)
        lay.addWidget(g)
        lay.addStretch(1)
        return w

    def _build_cavebot_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Rota:"))
        self.route_combo = QComboBox()
        self.route_combo.activated.connect(lambda _i: self._load_route(self.route_combo.currentText()))
        row.addWidget(self.route_combo, 1)
        for text, fn in (("Nova", self._new_route), ("Salvar", self._save_route),
                         ("Renomear", self._rename_route), ("Excluir", self._delete_route)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        self.loop_cb = QCheckBox("Repetir")
        self.loop_cb.toggled.connect(self._set_loop)
        row.addWidget(self.loop_cb)
        lay.addLayout(row)

        self.wp_table = QTableWidget(0, 6)
        self.wp_table.setHorizontalHeaderLabels(["#", "Nome", "Tipo", "Posição", "Raio", "Referências"])
        self.wp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.wp_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.wp_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.wp_table.verticalHeader().setVisible(False)
        self.wp_table.doubleClicked.connect(lambda _i: self._edit_waypoint())
        lay.addWidget(self.wp_table, 2)

        row = QHBoxLayout()
        for text, fn in (("Adicionar", self._add_waypoint), ("Editar", self._edit_waypoint),
                         ("Remover", self._remove_waypoint), ("↑", lambda: self._move_waypoint(-1)),
                         ("↓", lambda: self._move_waypoint(1)),
                         ("Usar minimapa atual como referência", self._capture_reference),
                         ("Limpar referências", self._clear_references)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)

        mid = QHBoxLayout()
        self.minimap_label = QLabel("minimapa")
        self.minimap_label.setFixedSize(180, 180)
        self.minimap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.minimap_label.setStyleSheet("background:#000; color:#777;")
        mid.addWidget(self.minimap_label)
        box = QGroupBox("Estado do CaveBot")
        form = QFormLayout(box)
        self.cb_state = QLabel("-")
        self.cb_current = QLabel("-")
        self.cb_next = QLabel("-")
        self.cb_progress = QProgressBar()
        self.cb_loc = QLabel("-")
        self.cb_loc.setWordWrap(True)
        self.cb_reason = QLabel("-")
        self.cb_reason.setWordWrap(True)
        form.addRow("Estado", self.cb_state)
        form.addRow("Atual", self.cb_current)
        form.addRow("Próximo", self.cb_next)
        form.addRow("Progresso", self.cb_progress)
        form.addRow("Posição", self.cb_loc)
        form.addRow("Motivo", self.cb_reason)
        mid.addWidget(box, 1)
        lay.addLayout(mid)

        row = QHBoxLayout()
        for text, fn in (("Iniciar", self._cb_start), ("Pausar", self._cb_pause),
                         ("Continuar", self._cb_resume), ("Parar", self._cb_stop),
                         ("Definir selecionado como atual", self._cb_set_current),
                         ("Testar com gravação…", self._observation_test)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)

        lay.addWidget(QLabel("Falhas e motivos (para recalibrar a rota):"))
        self.failures_list = QListWidget()
        lay.addWidget(self.failures_list, 1)
        return w

    def _build_battle_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.battle_summary = QLabel("Battle List: -")
        lay.addWidget(self.battle_summary)
        self.target_table = QTableWidget(0, 6)
        self.target_table.setHorizontalHeaderLabels(["ID", "Status", "Nome", "Vida", "Confiança", "Evidência"])
        self.target_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.target_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self.target_table, 1)
        row = QHBoxLayout()
        b = QPushButton("Cadastrar referência visual de monstro (selecionar na prévia)")
        b.clicked.connect(self._add_monster_template)
        row.addWidget(b)
        row.addWidget(QLabel("Limiar:"))
        self.det_threshold = QDoubleSpinBox()
        self.det_threshold.setRange(0.5, 0.99)
        self.det_threshold.setSingleStep(0.01)
        self.det_threshold.setValue(self.store.get("detector.threshold", 0.8))
        self.det_threshold.valueChanged.connect(lambda v: self._set_cfg("detector.threshold", v, "battle"))
        row.addWidget(self.det_threshold)
        lay.addLayout(row)
        self.templates_label = QLabel()
        lay.addWidget(self.templates_label)
        self._refresh_templates_label()
        return w

    def _build_health_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("HP do personagem — Exura Vita")
        gl = QFormLayout(g)
        self.hp_value = QLabel("—")
        self.hp_value.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        self.hp_bar = QProgressBar()
        self.hp_bar.setRange(0, 100)
        self.hp_info = QLabel("-")
        self.hp_info.setWordWrap(True)
        self.hp_threshold = QDoubleSpinBox()
        self.hp_threshold.setRange(1, 99)
        self.hp_threshold.setSuffix(" %")
        self.hp_threshold.setValue(self.store.get("health.threshold_percent"))
        self.hp_threshold.valueChanged.connect(self._set_hp_threshold)
        self.hp_conf = QDoubleSpinBox()
        self.hp_conf.setRange(0.3, 1.0)
        self.hp_conf.setSingleStep(0.05)
        self.hp_conf.setValue(self.store.get("health.min_confidence"))
        self.hp_conf.valueChanged.connect(lambda v: self._set_cfg("health.min_confidence", v, "health"))
        self.hp_sound = QCheckBox("Alerta sonoro")
        self.hp_sound.setChecked(self.store.get("health.alert_sound", True))
        self.hp_sound.toggled.connect(lambda v: self._set_cfg("health.alert_sound", v, "health"))
        cal = QPushButton("Calibrar 100% (com HP cheio)")
        cal.clicked.connect(lambda: self._calibrate_bar("health", "hp_bar"))
        gl.addRow(self.hp_value)
        gl.addRow(self.hp_bar)
        gl.addRow("Leitura", self.hp_info)
        gl.addRow("HP mínimo", self.hp_threshold)
        gl.addRow("Confiança mínima", self.hp_conf)
        gl.addRow(self.hp_sound, cal)
        lay.addWidget(g)

        g = QGroupBox("SIO — suporte (monitoramento visual)")
        gl = QFormLayout(g)
        self.sio_label_edit = QLineEdit(self.store.get("sio.label", "Aliado"))
        self.sio_label_edit.editingFinished.connect(
            lambda: self._set_cfg("sio.label", self.sio_label_edit.text() or "Aliado", "sio"))
        self.sio_value = QLabel("—")
        self.sio_value.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.sio_bar = QProgressBar()
        self.sio_bar.setRange(0, 100)
        self.sio_info = QLabel("-")
        self.sio_info.setWordWrap(True)
        self.sio_threshold = QDoubleSpinBox()
        self.sio_threshold.setRange(1, 99)
        self.sio_threshold.setSuffix(" %")
        self.sio_threshold.setValue(self.store.get("sio.threshold_percent"))
        self.sio_threshold.valueChanged.connect(lambda v: self._set_cfg("sio.threshold_percent", v, "sio"))
        ident = QPushButton("Cadastrar nome do aliado (selecionar na prévia)")
        ident.clicked.connect(lambda: self._begin_select("__sio_identity"))
        cal = QPushButton("Calibrar 100%")
        cal.clicked.connect(lambda: self._calibrate_bar("sio", "sio"))
        gl.addRow("Nome exibido", self.sio_label_edit)
        gl.addRow(self.sio_value)
        gl.addRow(self.sio_bar)
        gl.addRow("Leitura", self.sio_info)
        gl.addRow("Limite", self.sio_threshold)
        gl.addRow(ident, cal)
        lay.addWidget(g)
        lay.addStretch(1)
        return w

    def _build_timers_tab(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        self.timers_layout = QVBoxLayout()
        outer.addLayout(self.timers_layout)
        for timer in self.engine.timers.timers.values():
            self._add_timer_widget(timer)
        b = QPushButton("Adicionar temporizador")
        b.clicked.connect(self._new_timer)
        outer.addWidget(b)
        outer.addStretch(1)
        return w

    def _add_timer_widget(self, timer: SpellTimer) -> None:
        g = QGroupBox(timer.name)
        gl = QHBoxLayout(g)
        lbl = QLabel(timer.format())
        lbl.setFont(QFont("Consolas", 20, QFont.Weight.Bold))
        lbl.setMinimumWidth(110)
        bar = QProgressBar()
        bar.setRange(0, 1000)
        state = QLabel(timer.state.value)
        gl.addWidget(lbl)
        gl.addWidget(bar, 1)
        gl.addWidget(state)
        for text, fn in (("Iniciar", timer.start), ("Pausar", timer.pause),
                         ("Continuar", timer.resume), ("Reiniciar", timer.reset)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=fn: self._timer_action(f))
            gl.addWidget(b)
        spin = QDoubleSpinBox()
        spin.setRange(1, 3600)
        spin.setSuffix(" s")
        spin.setValue(timer.seconds)
        spin.valueChanged.connect(lambda v, t=timer: self._set_timer_interval(t, v))
        gl.addWidget(spin)
        self.timers_layout.addWidget(g)
        self._timer_widgets[timer.name] = {"label": lbl, "bar": bar, "state": state, "timer": timer}

    def _build_logs_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Filtro:"))
        self.log_filter = QComboBox()
        self.log_filter.addItems(["todos", "capture", "cavebot", "battle", "health", "sio", "timers", "engine"])
        self.log_filter.currentTextChanged.connect(self._reload_logs)
        row.addWidget(self.log_filter)
        b = QPushButton("Limpar")
        b.clicked.connect(lambda: (EVENTS.clear(), self.log_view.clear()))
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        lay.addWidget(self.log_view, 1)
        return w

    # ------------------------------------------------------------ captura
    def connect_capture(self) -> None:
        self.disconnect_capture()
        text = self.source_edit.currentText().strip()
        source = int(text) if text.isdigit() else text
        self.store.set("capture.source", source)
        cfg = self.store.data

        def on_status(old: CaptureStatus, new: CaptureStatus) -> None:
            level = logging.WARNING if new in (CaptureStatus.FROZEN, CaptureStatus.DISCONNECTED) else logging.INFO
            EVENTS.record("capture", "status", f"captura: {old.value} → {new.value}", level)

        try:
            self.grabber = grabber_from_config(cfg, on_status_change=on_status).start()
        except Exception as exc:
            QMessageBox.critical(self, "Captura", f"Não foi possível iniciar a captura: {exc}")
            self.grabber = None
            return
        EVENTS.record("capture", "connect", f"conectando à fonte {source!r}")

    def disconnect_capture(self) -> None:
        if self.grabber is not None:
            g, self.grabber = self.grabber, None
            threading.Thread(target=g.stop, daemon=True).start()
            self._preview_index = -1

    def _open_recording(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pasta com imagens gravadas")
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Vídeo gravado pelo OBS", "",
                                                  "Vídeos (*.mp4 *.mkv *.avi *.mov *.flv *.ts);;Todos (*)")
        if path:
            self.source_edit.setCurrentText(path)
            self.connect_capture()

    def _update_preview(self) -> None:
        g = self.grabber
        if g is None:
            self._set_capture_label(CaptureStatus.STOPPED)
            return
        frame = g.latest()
        self._set_capture_label(g.status)
        if frame is None or frame.index == self._preview_index:
            if frame is not None and self.preview.selecting:
                self.preview.show_frame(to_qimage(self._overlay(frame.image)))
            return
        self._preview_index = frame.index
        now = time.monotonic()
        self._fps_frames = [t for t in self._fps_frames if now - t < 1.0] + [now]
        self.fps_label.setText(f"{len(self._fps_frames)} fps | {frame.image.shape[1]}x{frame.image.shape[0]}")
        self.preview.show_frame(to_qimage(self._overlay(frame.image)))

    def _overlay(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        for key, (label, color) in REGION_LABELS.items():
            r = Region.from_any(self.store.get(f"regions.{key}"))
            if r:
                bgr = (color.blue(), color.green(), color.red())
                cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), bgr, 2)
                cv2.putText(out, label, (r.x, max(12, r.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1)
        snap = self.last_snapshot
        if snap:
            fused = {}
            for tgt in snap.targets:
                for vid in tgt.candidate_visual_ids:
                    fused.setdefault(vid, tgt)
            for v in snap.visual_tracks:
                x, y, w, h = v.box
                tgt = fused.get(v.track_id)
                color = (0, 200, 0) if tgt and tgt.status == TargetStatus.CONFIRMED else (0, 200, 255)
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
                cv2.putText(out, f"{v.name} {v.confidence:.2f}", (x, max(10, y - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return out

    def _set_capture_label(self, status: CaptureStatus) -> None:
        text = {"connected": "conectada", "frozen": "CONGELADA", "disconnected": "DESCONECTADA",
                "starting": "iniciando", "stopped": "parada"}[status.value]
        self.capture_label.setText(f"captura: {text}")
        self.capture_label.setStyleSheet(f"padding:4px; color:white; background:{STATUS_COLORS[status]};")

    def _latest_image(self) -> Optional[np.ndarray]:
        f = self.grabber.latest() if self.grabber else None
        if f is None:
            QMessageBox.information(self, "Captura", "Nenhum frame disponível. Conecte o OBS primeiro.")
            return None
        return f.image

    # ------------------------------------------------------------ regiões
    def _begin_select(self, key: str) -> None:
        if self.grabber is None or self.grabber.latest() is None:
            QMessageBox.information(self, "Seleção", "Conecte a captura para selecionar na prévia.")
            return
        self.preview.begin_selection(key)
        self.statusBar().showMessage("Arraste sobre a prévia para selecionar (botão direito cancela).")

    def _on_region_selected(self, key: str, region: Region) -> None:
        if key == "__sio_identity":
            img = self._latest_image()
            if img is None:
                return
            path = "sio_identity.png"
            cv2.imwrite(path, region.crop(img))
            self._set_cfg("sio.identity_template", path, "sio")
            EVENTS.record("sio", "identity_template", "imagem de referência do nome do aliado cadastrada")
            return
        if key.startswith("__template:"):
            img = self._latest_image()
            if img is None:
                return
            name = key.split(":", 1)[1]
            path = self.engine.templates.add(name, region.crop(img).copy())
            self.engine.rebuild("battle")
            self._refresh_templates_label()
            EVENTS.record("battle", "template", f"referência visual de '{name}' salva em {path}")
            return
        self.store.set(f"regions.{key}", region.to_dict())
        module = {"minimap": "cavebot", "battle_list": "battle", "game_area": "battle",
                  "hp_bar": "health", "sio": "sio", "sio_name": "sio"}[key]
        self.engine.rebuild(module)
        self._refresh_region_labels()
        EVENTS.record("capture", "region", f"região '{REGION_LABELS[key][0]}' definida: {region}")
        self.statusBar().showMessage(f"Região {REGION_LABELS[key][0]} salva.")

    def _clear_region(self, key: str) -> None:
        self.store.set(f"regions.{key}", None)
        self.engine.rebuild()
        self._refresh_region_labels()

    def _refresh_region_labels(self) -> None:
        for key, lbl in self.region_labels.items():
            r = Region.from_any(self.store.get(f"regions.{key}"))
            lbl.setText(f"x={r.x} y={r.y} {r.w}x{r.h}" if r else "(não definida)")

    def _check_minimap_calibration(self) -> None:
        img = self._latest_image()
        if img is None:
            return
        issues = calibration_report(img, Region.from_any(self.store.get("regions.minimap")),
                                    self.store.get("cavebot.min_detail", 8.0))
        QMessageBox.information(self, "Calibração do minimapa",
                                "Nenhum problema encontrado." if not issues else "\n".join(f"• {i}" for i in issues))

    # ------------------------------------------------------------ rotas
    def _load_routes_combo(self) -> None:
        cur = self.route_combo.currentText()
        self.route_combo.clear()
        self.route_combo.addItems(self.routes.list_routes())
        if cur:
            self.route_combo.setCurrentText(cur)

    def _load_route(self, name: str) -> None:
        if not name:
            return
        try:
            self.route = self.routes.load(name)
        except RouteError as exc:
            QMessageBox.warning(self, "Rota", str(exc))
            return
        with self.engine.lock:
            assert self.engine.cavebot is not None
            self.engine.cavebot.load_route(self.route)
        self.loop_cb.setChecked(self.route.loop)
        self.store.set("cavebot.last_route", name)
        self._refresh_waypoints()
        EVENTS.record("cavebot", "route_loaded", f"rota '{name}' carregada ({len(self.route.waypoints)} waypoints)")

    def _new_route(self) -> None:
        name, ok = QInputDialog.getText(self, "Nova rota", "Nome da rota:")
        if not ok or not name.strip():
            return
        try:
            self.routes.create(name.strip())
        except RouteError as exc:
            QMessageBox.warning(self, "Rota", str(exc))
            return
        self._load_routes_combo()
        self.route_combo.setCurrentText(name.strip())
        self._load_route(name.strip())

    def _save_route(self) -> None:
        if self.route is None:
            return
        try:
            self.routes.save(self.route)
        except (OSError, RouteError) as exc:
            QMessageBox.critical(self, "Rota", f"Falha ao salvar: {exc}")
            return
        problems = self.route.validate()
        self.statusBar().showMessage(f"Rota '{self.route.name}' salva." +
                                     (f" Atenção: {len(problems)} problema(s)." if problems else ""))
        for p in problems:
            EVENTS.record("cavebot", "route_problem", p, logging.WARNING)

    def _rename_route(self) -> None:
        if self.route is None:
            return
        name, ok = QInputDialog.getText(self, "Renomear rota", "Novo nome:", text=self.route.name)
        if not ok or not name.strip():
            return
        try:
            self.route = self.routes.rename(self.route.name, name.strip())
        except RouteError as exc:
            QMessageBox.warning(self, "Rota", str(exc))
            return
        self._load_routes_combo()
        self.route_combo.setCurrentText(self.route.name)
        self._load_route(self.route.name)

    def _delete_route(self) -> None:
        if self.route is None:
            return
        if QMessageBox.question(self, "Excluir rota", f"Excluir a rota '{self.route.name}'?") \
                != QMessageBox.StandardButton.Yes:
            return
        try:
            self.routes.delete(self.route.name)
        except RouteError as exc:
            QMessageBox.warning(self, "Rota", str(exc))
        self.route = None
        with self.engine.lock:
            if self.engine.cavebot:
                self.engine.cavebot.stop()
                self.engine.cavebot.route = None
        self._load_routes_combo()
        self._refresh_waypoints()

    def _set_loop(self, on: bool) -> None:
        if self.route is not None and self.route.loop != on:
            self.route.loop = on
            self.routes.save(self.route)

    def _refresh_waypoints(self) -> None:
        wps = self.route.waypoints if self.route else []
        self.wp_table.setRowCount(len(wps))
        for i, wp in enumerate(wps):
            pos = f"{wp.position[0]}, {wp.position[1]}, {wp.position[2]}" if wp.position else "-"
            for c, val in enumerate([str(i + 1), wp.name, WAYPOINT_KINDS[wp.kind], pos, str(wp.radius),
                                     str(len(wp.reference_ids)) if wp.reference_ids else "nenhuma"]):
                self.wp_table.setItem(i, c, QTableWidgetItem(val))

    def _selected_wp(self) -> Optional[int]:
        rows = self.wp_table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _require_route(self) -> bool:
        if self.route is None:
            QMessageBox.information(self, "Rota", "Crie ou carregue uma rota primeiro.")
            return False
        return True

    def _add_waypoint(self) -> None:
        if not self._require_route():
            return
        dlg = WaypointDialog(self, default_radius=self.store.get("cavebot.default_radius", 2))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            wp = dlg.result_waypoint()
            sel = self._selected_wp()
            self.route.add(wp, None if sel is None else sel + 1)
            self.routes.save(self.route)
        except RouteError as exc:
            QMessageBox.warning(self, "Waypoint", str(exc))
            return
        self._refresh_waypoints()

    def _edit_waypoint(self) -> None:
        i = self._selected_wp()
        if self.route is None or i is None:
            return
        dlg = WaypointDialog(self, self.route.waypoints[i])
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.route.replace(i, dlg.result_waypoint())
            self.routes.save(self.route)
        except RouteError as exc:
            QMessageBox.warning(self, "Waypoint", str(exc))
        self._refresh_waypoints()

    def _remove_waypoint(self) -> None:
        i = self._selected_wp()
        if self.route is None or i is None:
            return
        self.route.remove(i)
        self.routes.save(self.route)
        self._refresh_waypoints()

    def _move_waypoint(self, delta: int) -> None:
        i = self._selected_wp()
        if self.route is None or i is None:
            return
        self.route.move(i, i + delta)
        self.routes.save(self.route)
        self._refresh_waypoints()
        self.wp_table.selectRow(max(0, min(i + delta, len(self.route.waypoints) - 1)))

    def _capture_reference(self) -> None:
        i = self._selected_wp()
        if self.route is None or i is None:
            QMessageBox.information(self, "Referência", "Selecione um waypoint na tabela.")
            return
        img = self._latest_image()
        if img is None:
            return
        mini = self.engine.localizer.crop(img)
        if mini is None:
            QMessageBox.warning(self, "Referência", "Defina a região do minimapa na aba Regiões.")
            return
        issues = calibration_report(img, self.engine.localizer.region, self.store.get("cavebot.min_detail", 8.0))
        if issues and QMessageBox.question(self, "Referência", "Possíveis problemas:\n" + "\n".join(issues) +
                                           "\n\nCadastrar assim mesmo?") != QMessageBox.StandardButton.Yes:
            return
        wp = self.route.waypoints[i]
        with self.engine.lock:
            ref = self.engine.localizer.library.add(mini, wp.name, wp.position)
        wp.reference_ids.append(ref.id)
        self.routes.save(self.route)
        self._refresh_waypoints()
        EVENTS.record("cavebot", "reference", f"referência '{ref.id}' cadastrada para o waypoint '{wp.name}'")

    def _clear_references(self) -> None:
        i = self._selected_wp()
        if self.route is None or i is None:
            return
        wp = self.route.waypoints[i]
        used = {r for j, w in enumerate(self.route.waypoints) if j != i for r in w.reference_ids}
        with self.engine.lock:
            for rid in wp.reference_ids:
                if rid not in used:
                    self.engine.localizer.library.remove(rid)
        wp.reference_ids.clear()
        self.routes.save(self.route)
        self._refresh_waypoints()

    # ------------------------------------------------------------ CaveBot
    def _cb_start(self) -> None:
        if not self._require_route():
            return
        sel = self._selected_wp()
        try:
            with self.engine.lock:
                self.engine.localizer.reset()
                self.engine.cavebot.start(self.route, sel or 0)
        except ValueError as exc:
            QMessageBox.warning(self, "CaveBot", str(exc))

    def _cb_pause(self) -> None:
        with self.engine.lock:
            self.engine.cavebot.pause()

    def _cb_resume(self) -> None:
        with self.engine.lock:
            self.engine.cavebot.resume()

    def _cb_stop(self) -> None:
        with self.engine.lock:
            self.engine.cavebot.stop()

    def _cb_set_current(self) -> None:
        i = self._selected_wp()
        if i is not None:
            with self.engine.lock:
                self.engine.cavebot.set_current(i)

    def _observation_test(self) -> None:
        if not self._require_route():
            return
        path = QFileDialog.getExistingDirectory(self, "Pasta com frames gravados (PNG/JPG)")
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Vídeo gravado", "", "Vídeos (*.mp4 *.mkv *.avi *.mov);;Todos (*)")
        if not path:
            return
        cfg = self.store.data
        route = Route.from_dict(self.route.to_dict())
        from cave_navigation import MinimapLocalizer

        localizer = MinimapLocalizer.from_config(cfg, self.engine.localizer.library)

        def job() -> None:
            try:
                rep = run_observation(route, iter_recording(path), localizer,
                                      cfg["cavebot"]["confirm_frames"], cfg["cavebot"]["lost_pause_seconds"],
                                      fps=cfg["capture"].get("playback_fps", 10.0))
                self.bridge.report.emit(rep.summary())
            except Exception as exc:
                log.exception("falha no modo de observação")
                self.bridge.report.emit(f"Falha no teste: {exc}")

        threading.Thread(target=job, name="observation", daemon=True).start()
        self.statusBar().showMessage("Testando a rota com a gravação…")

    def _on_report(self, text: str) -> None:
        EVENTS.record("cavebot", "observation_report", "teste com gravação concluído")
        dlg = QDialog(self)
        dlg.setWindowTitle("Relatório do modo de observação")
        lay = QVBoxLayout(dlg)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        lay.addWidget(view)
        dlg.resize(800, 500)
        dlg.exec()

    # ------------------------------------------------------------ Battle
    def _add_monster_template(self) -> None:
        name, ok = QInputDialog.getText(self, "Referência visual", "Nome do monstro:")
        if ok and name.strip():
            self._begin_select(f"__template:{name.strip()}")

    def _refresh_templates_label(self) -> None:
        names = self.engine.templates.names
        self.templates_label.setText(f"Referências: {len(self.engine.templates.templates)} imagem(ns) — "
                                     + (", ".join(names) if names else "nenhuma"))

    # ------------------------------------------------------------ cura
    def _set_hp_threshold(self, v: float) -> None:
        self._set_cfg("health.threshold_percent", v, None)
        with self.engine.lock:
            self.engine.health.threshold_percent = v

    def _calibrate_bar(self, section: str, region_key: str) -> None:
        img = self._latest_image()
        r = Region.from_any(self.store.get(f"regions.{region_key}"))
        if img is None or r is None:
            QMessageBox.information(self, "Calibração", "Defina a região e conecte a captura.")
            return
        monitor = self.engine.health if section == "health" else self.engine.sio
        try:
            span = monitor.reader.calibrate_full(r.crop(img))
        except ValueError as exc:
            QMessageBox.warning(self, "Calibração", str(exc))
            return
        self.store.set(f"{section}.full_span", span, save=False)
        self.store.set(f"{section}.left_offset", monitor.reader.left_offset)
        EVENTS.record(section, "calibrated", f"barra calibrada: 100% = {span} px")

    # ------------------------------------------------------------ magias
    def _timer_action(self, fn) -> None:
        with self.engine.lock:
            fn()

    def _set_timer_interval(self, timer: SpellTimer, v: float) -> None:
        with self.engine.lock:
            timer.set_interval(v)
        self.store.set("spell_timers.timers", self.engine.timers.to_config())

    def _new_timer(self) -> None:
        name, ok = QInputDialog.getText(self, "Novo temporizador", "Nome da magia:")
        if not ok or not name.strip() or name.strip() in self.engine.timers.timers:
            return
        secs, ok = QInputDialog.getDouble(self, "Novo temporizador", "Intervalo (s):", 30, 1, 3600, 1)
        if not ok:
            return
        with self.engine.lock:
            t = self.engine.timers.add(SpellTimer(name.strip(), secs))
        self._add_timer_widget(t)
        self.store.set("spell_timers.timers", self.engine.timers.to_config())

    # ------------------------------------------------------------ atualização
    def _on_snapshot(self, snap: Snapshot) -> None:
        self.last_snapshot = snap
        if snap.minimap is not None and snap.minimap.size:
            pix = QPixmap.fromImage(to_qimage(snap.minimap)).scaled(
                self.minimap_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation)
            self.minimap_label.setPixmap(pix)
        st = snap.cavebot
        if st is not None:
            self.cb_state.setText(st.state.value)
            color = {BotState.RUNNING: "#2e7d32", BotState.WAITING_POSITION: "#ef6c00",
                     BotState.PAUSED: "#616161", BotState.FINISHED: "#1565c0"}.get(st.state, "#424242")
            self.cb_state.setStyleSheet(f"color:{color}; font-weight:bold;")
            self.cb_current.setText(f"#{st.current_index + 1} {st.current.describe()}" if st.current else "-")
            self.cb_next.setText(st.next.describe() if st.next else "-")
            self.cb_progress.setMaximum(max(1, st.total))
            self.cb_progress.setValue(st.completed)
            self.cb_progress.setFormat(f"{st.completed}/{st.total} (volta {st.lap + 1})")
            self.cb_reason.setText(st.last_reason or "-")
            if st.current_index is not None and st.state != BotState.STOPPED:
                for r in range(self.wp_table.rowCount()):
                    item = self.wp_table.item(r, 0)
                    if item:
                        item.setBackground(QColor("#fff59d") if r == st.current_index else QColor(0, 0, 0, 0))
        if snap.localization is not None:
            loc = snap.localization
            hint = failure_summary(loc.status)
            self.cb_loc.setText(loc.describe() + (f"\nDica: {hint}" if hint else ""))
        self._update_battle(snap)
        self._update_health(snap)
        if snap.errors:
            self.statusBar().showMessage("Erros: " + "; ".join(f"{k}: {v}" for k, v in snap.errors.items()))

    def _update_battle(self, snap: Snapshot) -> None:
        self.battle_summary.setText(f"Battle List: {len(snap.battle_rows)} linha(s) | "
                                    f"detecções visuais: {len(snap.visual_tracks)} | "
                                    f"alvos: {sum(1 for t in snap.targets if t.status != TargetStatus.GONE)}")
        targets = snap.targets
        self.target_table.setRowCount(len(targets))
        colors = {TargetStatus.CONFIRMED: "#c8e6c9", TargetStatus.PROBABLE: "#fff9c4",
                  TargetStatus.UNCERTAIN: "#eeeeee", TargetStatus.GONE: "#bdbdbd"}
        for i, t in enumerate(targets):
            vals = [str(t.target_id), t.status.value, t.name or "?",
                    "-" if t.hp_percent is None else f"{t.hp_percent:.0f}%", f"{t.score:.2f}",
                    " | ".join(t.evidence[-2:])]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setBackground(QColor(colors[t.status]))
                self.target_table.setItem(i, c, item)

    def _update_health(self, snap: Snapshot) -> None:
        hp = self.engine.health.current()
        if hp is not None:
            if hp.percent is None:
                self.hp_value.setText("—")
                self.hp_bar.setValue(0)
            else:
                self.hp_value.setText(f"{hp.percent:.0f}%" + ("" if hp.valid else " ?"))
                self.hp_bar.setValue(round(hp.percent))
            ok = hp.valid and not hp.stale
            low = ok and hp.percent is not None and hp.percent < self.engine.health.threshold_percent
            self.hp_value.setStyleSheet("color:#c62828;" if low else "color:#2e7d32;" if ok else "color:#9e9e9e;")
            self.hp_info.setText(hp.describe())
        sio = self.engine.sio.current()
        if sio is not None:
            self.sio_value.setText("—" if sio.percent is None else f"{sio.percent:.0f}%" + ("" if sio.valid else " ?"))
            self.sio_bar.setValue(round(sio.percent or 0))
            self.sio_info.setText(sio.describe() + ("\n" + "; ".join(sio.reasons) if sio.reasons else ""))

    def _tick_ui(self) -> None:
        with self.engine.lock:
            self.engine.timers.tick()
        for w in self._timer_widgets.values():
            t: SpellTimer = w["timer"]
            w["label"].setText(t.format())
            w["bar"].setValue(int(1000 * t.progress()))
            w["state"].setText(t.state.value)
            warn = t.state == TimerState.EXPIRED or (t.state == TimerState.RUNNING and t.remaining() <= t.warn_seconds)
            w["label"].setStyleSheet("color:#c62828;" if warn else "")

    def _on_event(self, ev: Event) -> None:
        f = self.log_filter.currentText()
        if f == "todos" or f == ev.source:
            self.log_view.appendPlainText(ev.format())
        if ev.source == "cavebot" and ev.kind == "failure":
            self.failures_list.addItem(ev.format())
            self.failures_list.scrollToBottom()
        if ev.level >= logging.CRITICAL:
            self._on_alert("critical", ev.message)

    def _reload_logs(self) -> None:
        f = self.log_filter.currentText()
        self.log_view.setPlainText("\n".join(e.format() for e in EVENTS.events(None if f == "todos" else f)))

    def _wire_alerts(self) -> None:
        self.engine.add_alert_listener(self.bridge.alert.emit)

    def _on_alert(self, level: str, message: str) -> None:
        if level == "critical":
            self.alert_banner.setText("⚠ " + message)
            self.alert_banner.setStyleSheet("background:#c62828; color:white; font-size:16px; padding:6px;")
            self.alert_banner.setVisible(True)
            QApplication.alert(self)
            QTimer.singleShot(4000, lambda: self.alert_banner.setVisible(False))
        else:
            self.alert_banner.setText(message)
            self.alert_banner.setStyleSheet("background:#2e7d32; color:white; font-size:14px; padding:4px;")
            self.alert_banner.setVisible(True)
            QTimer.singleShot(2500, lambda: self.alert_banner.setVisible(False))

    # ------------------------------------------------------------ emergência
    def emergency_stop(self) -> None:
        self.engine.emergency_stop()
        self.release_btn.setEnabled(True)
        self.stop_btn.setText("PARADO — análise interrompida")

    def _release(self) -> None:
        self.engine.release()
        self.release_btn.setEnabled(False)
        self.stop_btn.setText("PARADA DE EMERGÊNCIA (F12)")

    # ------------------------------------------------------------ util
    def _set_cfg(self, path: str, value, module: Optional[str]) -> None:
        self.store.set(path, value)
        if module:
            self.engine.rebuild(module)

    def closeEvent(self, ev):  # noqa: N802
        g = self.geometry()
        self.store.set("ui.geometry", [g.x(), g.y(), g.width(), g.height()])
        EVENTS.unsubscribe(self.bridge.event.emit)
        self.preview_timer.stop()
        self.ui_timer.stop()
        self.worker.stop()
        if self.grabber:
            self.grabber.stop()
        super().closeEvent(ev)


def run_gui(store: Optional[ConfigStore] = None, autoconnect: bool = True) -> int:
    app = QApplication.instance() or QApplication([])
    store = store or ConfigStore()
    win = MainWindow(store)
    win.show()
    if autoconnect:
        QTimer.singleShot(200, win.connect_capture)
    return app.exec()
