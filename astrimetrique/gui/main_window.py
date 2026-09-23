"""
Main Application Window for Astrimetrique with Image Registration, Blinking & Diffing.
"""

from pathlib import Path
from typing import List, Optional
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow,
    QToolBar,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QLabel,
    QDockWidget,
    QWidget,
)

from astrimetrique.core.centroid import CentroidResult
from astrimetrique.core.diff_engine import MovingCandidate
from astrimetrique.core.fits_io import FITSImage, load_fits
from astrimetrique.core.plate_solver import ReferenceStar
from astrimetrique.core.stretch import StretchMode
from astrimetrique.core.synthetic_data import (
    generate_synthetic_fits_field,
    generate_multiepoch_synthetic_pair,
)
from astrimetrique.gui.control_panel import ControlPanel
from astrimetrique.gui.fits_header_dialog import FITSHeaderDialog
from astrimetrique.gui.image_viewer import (
    AstronomicalImageViewer,
    InteractionMode,
    ViewMode,
)
from astrimetrique.gui.star_dialog import StarReferenceDialog
from astrimetrique.gui.styles import OBSIDIAN_THEME
from astrimetrique.utils.coordinates import format_dec, format_ra


class MainWindow(QMainWindow):
    """Astrimetrique Main Window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astrimetrique — Astrometric Reduction & Minor Body Discovery Engine")
        self.resize(1380, 900)

        # Apply Obsidian Space theme
        self.setStyleSheet(OBSIDIAN_THEME)

        self._init_ui()
        self._init_menus_and_toolbar()
        self._init_statusbar()

        # Load initial multi-epoch synthetic asteroid field for instant discovery demonstration!
        self.load_synthetic_multiepoch_pair()

    def _init_ui(self):
        # 1. Central Viewport
        self.viewer = AstronomicalImageViewer(self)
        self.setCentralWidget(self.viewer)

        self.viewer.cursorMoved.connect(self._on_cursor_moved)
        self.viewer.starPicked.connect(self._on_star_picked)
        self.viewer.targetPicked.connect(self._on_target_picked)
        self.viewer.activeFrameChanged.connect(self._on_active_frame_changed)

        # 2. Right Control Panel Dock
        self.dock = QDockWidget("Astrometric Control & Discovery", self)
        self.dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)

        self.control_panel = ControlPanel(self)
        self.dock.setWidget(self.control_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)

        # Connect Control Panel signals to Viewport
        self.control_panel.starsChanged.connect(self._on_stars_changed)
        self.control_panel.targetUpdated.connect(self.viewer.set_target_marker)
        self.control_panel.viewModeRequested.connect(self.viewer.set_view_mode)
        self.control_panel.blinkSpeedChanged.connect(self.viewer.set_blink_speed)
        self.control_panel.warpedImageReady.connect(self.viewer.set_warped_target_image)
        self.control_panel.diffImageReady.connect(self.viewer.set_difference_image)
        self.control_panel.candidatesFound.connect(self.viewer.update_candidate_markers)
        self.control_panel.candidateSelected.connect(self._on_candidate_selected)

    def _init_menus_and_toolbar(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")

        act_open_ref = QAction("📂 Open Reference Image (Frame A)...", self)
        act_open_ref.setShortcut("Ctrl+O")
        act_open_ref.triggered.connect(self.open_reference_fits_dialog)
        file_menu.addAction(act_open_ref)

        act_open_target = QAction("📂 Open Target Image (Frame B)...", self)
        act_open_target.setShortcut("Ctrl+Shift+O")
        act_open_target.triggered.connect(self.open_target_fits_dialog)
        file_menu.addAction(act_open_target)

        file_menu.addSeparator()

        act_synth_pair = QAction("✨ Generate Multi-Epoch Discovery Pair (Moving Asteroid)", self)
        act_synth_pair.setShortcut("Ctrl+M")
        act_synth_pair.triggered.connect(self.load_synthetic_multiepoch_pair)
        file_menu.addAction(act_synth_pair)

        act_synth_single = QAction("✨ Generate Single Test Field", self)
        act_synth_single.triggered.connect(self.load_synthetic_single_field)
        file_menu.addAction(act_synth_single)

        file_menu.addSeparator()

        act_headers = QAction("🔍 Inspect FITS Headers...", self)
        act_headers.setShortcut("Ctrl+H")
        act_headers.triggered.connect(self.inspect_fits_headers)
        file_menu.addAction(act_headers)

        act_export_mpc = QAction("📄 Export MPC 80-Column Report...", self)
        act_export_mpc.setShortcut("Ctrl+E")
        act_export_mpc.triggered.connect(self.control_panel._open_mpc_export_dialog)
        file_menu.addAction(act_export_mpc)

        file_menu.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Discovery Menu
        discovery_menu = menu_bar.addMenu("&Discovery")

        act_align = QAction("🔄 Align Frame B to Frame A", self)
        act_align.setShortcut("Ctrl+A")
        act_align.triggered.connect(self.control_panel._run_alignment)
        discovery_menu.addAction(act_align)

        act_diff = QAction("🔍 Compute Difference Map", self)
        act_diff.setShortcut("Ctrl+D")
        act_diff.triggered.connect(self.control_panel._run_diffing)
        discovery_menu.addAction(act_diff)

        # Toolbar
        self.toolbar = QToolBar("Main Operations", self)
        self.toolbar.setIconSize(QSize(18, 18))
        self.toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        self.toolbar.addAction(act_open_ref)
        self.toolbar.addAction(act_open_target)
        self.toolbar.addAction(act_synth_pair)
        self.toolbar.addSeparator()

        # Registration & Diff Actions in Toolbar
        self.toolbar.addAction(act_align)
        self.toolbar.addAction(act_diff)
        self.toolbar.addSeparator()

        # Stretch selector
        lbl_stretch = QLabel(" Stretch: ")
        lbl_stretch.setStyleSheet("color: #94A3B8; font-weight: bold;")
        self.toolbar.addWidget(lbl_stretch)

        self.combo_stretch = QComboBox()
        for mode in StretchMode:
            self.combo_stretch.addItem(mode.value, mode)
        self.combo_stretch.setCurrentText(StretchMode.ZSCALE.value)
        self.combo_stretch.currentIndexChanged.connect(self._on_stretch_changed)
        self.toolbar.addWidget(self.combo_stretch)

        # Invert Colors Action
        self.act_invert = QAction("🌓 Invert (Negative)", self)
        self.act_invert.setCheckable(True)
        self.act_invert.toggled.connect(self.viewer.set_invert_colors)
        self.toolbar.addAction(self.act_invert)

        self.toolbar.addSeparator()

        # Interaction Mode Group
        mode_group = QActionGroup(self)

        self.act_mode_pick_star = QAction("⭐ Pick Star Mode", self)
        self.act_mode_pick_star.setCheckable(True)
        self.act_mode_pick_star.setChecked(True)
        self.act_mode_pick_star.triggered.connect(lambda: self.viewer.set_interaction_mode(InteractionMode.PICK_STAR))
        mode_group.addAction(self.act_mode_pick_star)
        self.toolbar.addAction(self.act_mode_pick_star)

        self.act_mode_pick_target = QAction("🎯 Pick Target Mode", self)
        self.act_mode_pick_target.setCheckable(True)
        self.act_mode_pick_target.triggered.connect(lambda: self.viewer.set_interaction_mode(InteractionMode.PICK_TARGET))
        mode_group.addAction(self.act_mode_pick_target)
        self.toolbar.addAction(self.act_mode_pick_target)

        self.act_mode_pan = QAction("✋ Pan / Zoom", self)
        self.act_mode_pan.setCheckable(True)
        self.act_mode_pan.triggered.connect(lambda: self.viewer.set_interaction_mode(InteractionMode.PAN_ZOOM))
        mode_group.addAction(self.act_mode_pan)
        self.toolbar.addAction(self.act_mode_pan)

        self.toolbar.addSeparator()
        self.toolbar.addAction(act_headers)
        self.toolbar.addAction(act_export_mpc)

    def _init_statusbar(self):
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)

        self.lbl_status_pos = QLabel("X: --- px  Y: --- px")
        self.lbl_status_pos.setStyleSheet("color: #00F2FE; font-weight: bold; margin-right: 12px;")
        self.statusbar.addPermanentWidget(self.lbl_status_pos)

        self.lbl_status_adu = QLabel("ADU: ---")
        self.lbl_status_adu.setStyleSheet("color: #E2E8F0; margin-right: 12px;")
        self.statusbar.addPermanentWidget(self.lbl_status_adu)

        self.lbl_status_sky = QLabel("Sky: ---")
        self.lbl_status_sky.setStyleSheet("color: #00E676; margin-right: 12px;")
        self.statusbar.addPermanentWidget(self.lbl_status_sky)

        self.lbl_status_info = QLabel("Ready")
        self.statusbar.addWidget(self.lbl_status_info)

    def open_reference_fits_dialog(self):
        """Open Reference Image A."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Reference Astronomical FITS Image (Frame A)",
            "",
            "FITS Images (*.fits *.fit *.fts);;All Files (*)",
        )
        if file_path:
            try:
                fits_img = load_fits(file_path)
                self.viewer.load_reference_image(fits_img)
                self.control_panel.set_fits_image(fits_img)
                self.lbl_status_info.setText(f"Loaded Reference Image A: {fits_img.metadata.filename}")
            except Exception as e:
                QMessageBox.critical(self, "FITS Load Error", f"Failed to load Reference Image:\n{e}")

    def open_target_fits_dialog(self):
        """Open Target Image B."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Target Astronomical FITS Image (Frame B)",
            "",
            "FITS Images (*.fits *.fit *.fts);;All Files (*)",
        )
        if file_path:
            try:
                fits_img = load_fits(file_path)
                self.viewer.load_target_image(fits_img)
                self.control_panel.set_target_fits_image(fits_img)
                self.lbl_status_info.setText(f"Loaded Target Image B: {fits_img.metadata.filename}")
            except Exception as e:
                QMessageBox.critical(self, "FITS Load Error", f"Failed to load Target Image:\n{e}")

    def load_synthetic_multiepoch_pair(self):
        """Generate and load synthetic multi-epoch pair with moving minor planet."""
        img_a, img_b, stars_a, stars_b, ast_a, ast_b = generate_multiepoch_synthetic_pair(
            width=1024,
            height=1024,
            telescope_shift_x=12.4,
            telescope_shift_y=-7.8,
            telescope_rot_deg=0.75,
            asteroid_motion_x=32.0,
            asteroid_motion_y=16.0,
            asteroid_mag=14.8,
        )

        self.viewer.load_reference_image(img_a)
        self.viewer.load_target_image(img_b)

        self.control_panel.set_fits_image(img_a)
        self.control_panel.set_target_fits_image(img_b, stars_b=stars_b)
        self.control_panel.set_reference_stars(stars_a)

        # Set target asteroid
        target_centroid = CentroidResult(
            x=ast_a.x,
            y=ast_a.y,
            fwhm=3.1,
            snr=24.5,
            peak_flux=210.0,
            success=True,
        )
        self.control_panel.set_target_centroid(target_centroid)

        self.lbl_status_info.setText(
            "Loaded Multi-Epoch Discovery Pair (Epoch 1 & Epoch 2 with moving asteroid 2024 AB). Ready to Align & Blink!"
        )

    def load_synthetic_single_field(self):
        """Generate single test field."""
        fits_img, ref_stars, target_star = generate_synthetic_fits_field(
            width=1024,
            height=1024,
            center_ra_deg=180.0,
            center_dec_deg=25.0,
            num_stars=14,
            include_asteroid=True,
        )
        self.viewer.load_reference_image(fits_img)
        self.control_panel.set_fits_image(fits_img)
        self.control_panel.set_reference_stars(ref_stars)

        if target_star:
            target_centroid = CentroidResult(
                x=target_star.x,
                y=target_star.y,
                fwhm=3.1,
                snr=22.4,
                peak_flux=180.0,
                success=True,
            )
            self.control_panel.set_target_centroid(target_centroid)

        self.lbl_status_info.setText("Generated Single Synthetic Test Field")

    def inspect_fits_headers(self):
        """Open FITS Header Inspector dialog."""
        img = self.viewer.fits_image_a or self.viewer.fits_image_b
        if img is None:
            QMessageBox.warning(self, "No FITS Loaded", "Please load a FITS image first.")
            return

        dlg = FITSHeaderDialog(img.metadata.headers, parent=self)
        dlg.exec()

    def _on_stretch_changed(self, index: int):
        mode = self.combo_stretch.currentData()
        if mode:
            self.viewer.set_stretch_mode(mode)

    def _on_active_frame_changed(self, frame_name: str):
        pass

    def _on_candidate_selected(self, x: float, y: float):
        """Center viewport around candidate coordinates."""
        self.viewer.plot_item.setXRange(x - 60, x + 60, padding=0)
        self.viewer.plot_item.setYRange(y - 60, y + 60, padding=0)

    def _on_cursor_moved(self, x: float, y: float, adu: float):
        self.lbl_status_pos.setText(f"X: {x:6.2f} px  Y: {y:6.2f} px")
        self.lbl_status_adu.setText(f"ADU: {adu:7.1f}")

        if self.control_panel.solver.solution is not None:
            try:
                ra_deg, dec_deg = self.control_panel.solver.pixel_to_sky(x, y)
                self.lbl_status_sky.setText(f"Sky (J2000): {format_ra(ra_deg)} {format_dec(dec_deg)}")
            except Exception:
                self.lbl_status_sky.setText("Sky: ---")
        elif self.viewer.fits_image_a and self.viewer.fits_image_a.wcs and self.viewer.fits_image_a.wcs.has_celestial:
            try:
                sky = self.viewer.fits_image_a.wcs.pixel_to_world(x, y)
                self.lbl_status_sky.setText(f"Sky (WCS): {format_ra(sky.ra.deg)} {format_dec(sky.dec.deg)}")
            except Exception:
                self.lbl_status_sky.setText("Sky: ---")
        else:
            self.lbl_status_sky.setText("Sky: ---")

    def _on_star_picked(self, centroid: CentroidResult):
        """Handle star pick event."""
        approx_ra = None
        approx_dec = None
        if self.control_panel.solver.solution is not None:
            approx_ra, approx_dec = self.control_panel.solver.pixel_to_sky(centroid.x, centroid.y)
        elif self.viewer.fits_image_a and self.viewer.fits_image_a.wcs and self.viewer.fits_image_a.wcs.has_celestial:
            sky = self.viewer.fits_image_a.wcs.pixel_to_world(centroid.x, centroid.y)
            approx_ra = float(sky.ra.deg)
            approx_dec = float(sky.dec.deg)
        elif self.viewer.fits_image_a:
            approx_ra = self.viewer.fits_image_a.metadata.ra_approx
            approx_dec = self.viewer.fits_image_a.metadata.dec_approx

        star_idx = len(self.control_panel.reference_stars) + 1
        dlg = StarReferenceDialog(
            centroid=centroid,
            default_id=f"REF-{star_idx:02d}",
            approx_ra=approx_ra,
            approx_dec=approx_dec,
            parent=self,
        )
        if dlg.exec() and dlg.result_star:
            self.control_panel.add_reference_star(dlg.result_star)

    def _on_target_picked(self, centroid: CentroidResult):
        """Handle target pick event."""
        self.control_panel.set_target_centroid(centroid)
        self.lbl_status_info.setText(f"Target selected at ({centroid.x:.2f}, {centroid.y:.2f})")

    def _on_stars_changed(self):
        """Sync star markers."""
        self.viewer.update_star_markers(self.control_panel.reference_stars)
