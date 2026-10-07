"""
Astrometrica-Style High-Density Scientific Control Center.
Presents stacked industrial panels: Reference Star Table, Target Object Panel,
Solution Quality Box, MPC Settings, and Discovery/Blinking Engine.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QHeaderView,
    QMessageBox,
    QProgressBar,
    QSlider,
    QScrollArea,
    QFrame,
)

from astrimetrique.core.centroid import CentroidResult
from astrimetrique.core.diff_engine import DiffResult, MovingCandidate, compute_difference_map
from astrimetrique.core.fits_io import FITSImage
from astrimetrique.core.gaia_catalog import GaiaStar, query_gaia_cone
from astrimetrique.core.mpc_exporter import MPCObservation, MPCObservatoryHeader
from astrimetrique.core.plate_solver import LeastSquaresPlateSolver, PlateSolution, ReferenceStar
from astrimetrique.core.registration import (
    RegistrationTransform,
    compute_affine_from_points,
    warp_image,
)
from astrimetrique.gui.export_dialog import MPCExportDialog
from astrimetrique.gui.image_viewer import ViewMode
from astrimetrique.gui.star_dialog import StarReferenceDialog
from astrimetrique.utils.coordinates import format_dec, format_ra, parse_dec, parse_ra


class AlignmentWorker(QThread):
    """Background worker for sub-pixel image warping & registration."""
    finished = pyqtSignal(object, np.ndarray)  # (RegistrationTransform, warped_array)
    error = pyqtSignal(str)

    def __init__(
        self,
        fits_a: FITSImage,
        fits_b: FITSImage,
        stars_a: List[ReferenceStar],
        stars_b: Optional[List[ReferenceStar]] = None,
        sol_a: Optional[PlateSolution] = None,
        sol_b: Optional[PlateSolution] = None,
    ):
        super().__init__()
        self.fits_a = fits_a
        self.fits_b = fits_b
        self.stars_a = stars_a
        self.stars_b = stars_b or []
        self.sol_a = sol_a
        self.sol_b = sol_b

    def run(self):
        try:
            from astrimetrique.core.registration import align_images_pipeline
            transform, warped_b = align_images_pipeline(
                self.fits_a,
                self.fits_b,
                self.stars_a,
                self.stars_b,
                self.sol_a,
                self.sol_b,
            )
            self.finished.emit(transform, warped_b)
        except Exception as e:
            self.error.emit(str(e))


class DiffWorker(QThread):
    """Background worker for photometric difference map computation."""
    finished = pyqtSignal(object)  # DiffResult
    error = pyqtSignal(str)

    def __init__(self, img_a: np.ndarray, img_b_warped: np.ndarray, threshold_sigma: float = 3.5):
        super().__init__()
        self.img_a = img_a
        self.img_b_warped = img_b_warped
        self.threshold_sigma = threshold_sigma

    def run(self):
        try:
            diff_res = compute_difference_map(self.img_a, self.img_b_warped, threshold_sigma=self.threshold_sigma)
            self.finished.emit(diff_res)
        except Exception as e:
            self.error.emit(str(e))


class FieldGaiaWorker(QThread):
    """Background worker for querying Gaia DR3 across the entire field."""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, ra_center: float, dec_center: float, radius_arcmin: float):
        super().__init__()
        self.ra_center = ra_center
        self.dec_center = dec_center
        self.radius_arcmin = radius_arcmin

    def run(self):
        try:
            stars = query_gaia_cone(
                self.ra_center,
                self.dec_center,
                radius_arcmin=self.radius_arcmin,
                mag_limit=17.0,
                row_limit=100,
            )
            self.finished.emit(stars)
        except Exception as e:
            self.error.emit(str(e))


class ControlPanel(QWidget):
    """
    High-density Astrometrica scientific control center with stacked panels.
    """

    solveRequested = pyqtSignal()
    starsChanged = pyqtSignal()
    targetUpdated = pyqtSignal(float, float, str)  # (x, y, label)
    viewModeRequested = pyqtSignal(object)  # ViewMode
    blinkSpeedChanged = pyqtSignal(int)  # ms
    warpedImageReady = pyqtSignal(np.ndarray)
    diffImageReady = pyqtSignal(np.ndarray)
    candidatesFound = pyqtSignal(list)
    candidateSelected = pyqtSignal(float, float)  # (x, y)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.fits_image_a: Optional[FITSImage] = None
        self.fits_image_b: Optional[FITSImage] = None
        self.reference_stars: List[ReferenceStar] = []
        self.reference_stars_b: List[ReferenceStar] = []
        self.target_centroid: Optional[CentroidResult] = None
        self.target_desig: str = "2024 AB"
        self.target_mag: Optional[float] = None
        self.target_ra_deg: Optional[float] = None
        self.target_dec_deg: Optional[float] = None

        self.solver = LeastSquaresPlateSolver()
        self.plate_solution: Optional[PlateSolution] = None
        self.reg_transform: Optional[RegistrationTransform] = None
        self.diff_result: Optional[DiffResult] = None
        self.warped_b_array: Optional[np.ndarray] = None

        self.align_worker: Optional[AlignmentWorker] = None
        self.diff_worker: Optional[DiffWorker] = None
        self.gaia_worker: Optional[FieldGaiaWorker] = None

        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(2, 2, 2, 2)
        root_layout.setSpacing(0)

        # Scroll area containing the stacked industrial panels
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        mono_font = QFont("Courier New", 10)

        # =========================================================================
        # Panel 1: Reference Star Table (Dense Grid: ID | X | Y | RA | Dec | Res)
        # =========================================================================
        group_stars = QGroupBox("Reference Star Table (LSPC)")
        stars_layout = QVBoxLayout(group_stars)
        stars_layout.setContentsMargins(4, 6, 4, 4)
        stars_layout.setSpacing(4)

        self.star_table = QTableWidget()
        self.star_table.setColumnCount(6)
        self.star_table.setHorizontalHeaderLabels([
            "ID", "X (px)", "Y (px)", "RA (J2000)", "Dec (J2000)", "Res (\")"
        ])
        self.star_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.star_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.star_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.star_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.star_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.star_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.star_table.verticalHeader().setVisible(False)
        self.star_table.verticalHeader().setDefaultSectionSize(18)
        self.star_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.star_table.itemDoubleClicked.connect(self._on_star_row_double_clicked)
        self.star_table.setMinimumHeight(150)
        stars_layout.addWidget(self.star_table)

        self.gaia_progress = QProgressBar()
        self.gaia_progress.setRange(0, 0)
        self.gaia_progress.hide()
        stars_layout.addWidget(self.gaia_progress)

        star_btns_layout = QHBoxLayout()
        star_btns_layout.setSpacing(4)

        self.btn_solve = QPushButton("Solve Plate")
        self.btn_solve.setObjectName("solveButton")
        self.btn_solve.clicked.connect(self.solve_plate)
        star_btns_layout.addWidget(self.btn_solve)

        self.btn_auto_gaia = QPushButton("Gaia DR3 Field")
        self.btn_auto_gaia.clicked.connect(self._fetch_gaia_field)
        star_btns_layout.addWidget(self.btn_auto_gaia)

        self.btn_clear_stars = QPushButton("Clear")
        self.btn_clear_stars.clicked.connect(self._clear_all_stars)
        star_btns_layout.addWidget(self.btn_clear_stars)
        stars_layout.addLayout(star_btns_layout)

        layout.addWidget(group_stars)

        # =========================================================================
        # Panel 2: Target Object Panel (Designation & Solved Coordinates)
        # =========================================================================
        group_target = QGroupBox("Target Object Panel")
        target_layout = QFormLayout(group_target)
        target_layout.setContentsMargins(4, 6, 4, 4)
        target_layout.setSpacing(3)

        target_row1 = QHBoxLayout()
        target_row1.setSpacing(4)
        self.txt_target_desig = QLineEdit(self.target_desig)
        self.txt_target_desig.setPlaceholderText("e.g. 2024 AB")
        self.txt_target_desig.textChanged.connect(self._on_target_desig_changed)
        target_row1.addWidget(self.txt_target_desig)

        self.chk_discovery = QCheckBox("Disc (*)")
        target_row1.addWidget(self.chk_discovery)
        target_layout.addRow("Designation:", target_row1)

        self.lbl_target_xy = QLabel("-")
        self.lbl_target_xy.setFont(mono_font)
        target_layout.addRow("Centroid (x,y):", self.lbl_target_xy)

        self.lbl_target_ra = QLabel("-")
        self.lbl_target_ra.setFont(mono_font)
        self.lbl_target_ra.setStyleSheet("color: #000080; font-weight: bold;")
        target_layout.addRow("Solved RA:", self.lbl_target_ra)

        self.lbl_target_dec = QLabel("-")
        self.lbl_target_dec.setFont(mono_font)
        self.lbl_target_dec.setStyleSheet("color: #000080; font-weight: bold;")
        target_layout.addRow("Solved Dec:", self.lbl_target_dec)

        target_row2 = QHBoxLayout()
        target_row2.setSpacing(4)
        self.txt_target_mag = QLineEdit("15.5")
        self.txt_target_mag.setPlaceholderText("Mag")
        target_row2.addWidget(self.txt_target_mag)

        self.lbl_target_fwhm = QLabel("SNR: -")
        self.lbl_target_fwhm.setFont(mono_font)
        target_row2.addWidget(self.lbl_target_fwhm)
        target_layout.addRow("Mag / SNR:", target_row2)

        layout.addWidget(group_target)

        # =========================================================================
        # Panel 3: Solution Quality Box (Scale, Rotation, Total RMS)
        # =========================================================================
        group_solution = QGroupBox("Solution Quality Box")
        sol_layout = QFormLayout(group_solution)
        sol_layout.setContentsMargins(4, 6, 4, 4)
        sol_layout.setSpacing(3)

        self.lbl_sol_scale = QLabel("-")
        self.lbl_sol_scale.setFont(mono_font)
        sol_layout.addRow("Pixel Scale:", self.lbl_sol_scale)

        self.lbl_sol_rotation = QLabel("-")
        self.lbl_sol_rotation.setFont(mono_font)
        sol_layout.addRow("Field Rotation:", self.lbl_sol_rotation)

        self.lbl_sol_rms_tot = QLabel("-")
        self.lbl_sol_rms_tot.setFont(mono_font)
        self.lbl_sol_rms_tot.setStyleSheet("color: #006000; font-weight: bold;")
        sol_layout.addRow("Total RMS Error:", self.lbl_sol_rms_tot)

        self.lbl_sol_stars = QLabel("0 stars used")
        self.lbl_sol_stars.setFont(mono_font)
        sol_layout.addRow("Residuals:", self.lbl_sol_stars)

        layout.addWidget(group_solution)

        # =========================================================================
        # Panel 4: MPC Settings (Obs Code, Date, Filter)
        # =========================================================================
        group_mpc = QGroupBox("MPC Settings & Export")
        mpc_layout = QFormLayout(group_mpc)
        mpc_layout.setContentsMargins(4, 6, 4, 4)
        mpc_layout.setSpacing(3)

        mpc_fields_row = QHBoxLayout()
        mpc_fields_row.setSpacing(4)
        self.txt_obs_code = QLineEdit("500")
        self.txt_obs_code.setMaximumWidth(45)
        self.txt_obs_code.setMaxLength(3)
        mpc_fields_row.addWidget(self.txt_obs_code)

        self.txt_filter_band = QLineEdit("R")
        self.txt_filter_band.setMaximumWidth(30)
        self.txt_filter_band.setMaxLength(1)
        mpc_fields_row.addWidget(self.txt_filter_band)

        self.txt_observer = QLineEdit("Observer")
        mpc_fields_row.addWidget(self.txt_observer)
        mpc_layout.addRow("Code/Filt/Obs:", mpc_fields_row)

        self.txt_obs_date = QLineEdit("")
        self.txt_obs_date.setPlaceholderText("YYYY-MM-DDTHH:MM:SS.sss (UTC)")
        mpc_layout.addRow("Date (UTC):", self.txt_obs_date)

        self.btn_export_mpc = QPushButton("Export MPC 80-Col")
        self.btn_export_mpc.setObjectName("primaryButton")
        self.btn_export_mpc.clicked.connect(self._open_mpc_export_dialog)
        mpc_layout.addRow("", self.btn_export_mpc)

        layout.addWidget(group_mpc)

        # =========================================================================
        # Panel 5: Multi-Epoch Registration & Discovery Engine
        # =========================================================================
        group_discovery = QGroupBox("Registration, Blinking & Diffing")
        disc_layout = QVBoxLayout(group_discovery)
        disc_layout.setContentsMargins(4, 6, 4, 4)
        disc_layout.setSpacing(4)

        self.lbl_align_status = QLabel("Target: Not aligned")
        self.lbl_align_status.setFont(mono_font)
        self.lbl_align_status.setStyleSheet("color: #000080; font-weight: bold;")
        disc_layout.addWidget(self.lbl_align_status)

        reg_btn_row = QHBoxLayout()
        reg_btn_row.setSpacing(4)
        self.btn_align = QPushButton("Align B -> A")
        self.btn_align.clicked.connect(self._run_alignment)
        reg_btn_row.addWidget(self.btn_align)

        self.btn_toggle_blink = QPushButton("▶ Blink")
        self.btn_toggle_blink.setCheckable(True)
        self.btn_toggle_blink.clicked.connect(self._on_blink_toggled)
        reg_btn_row.addWidget(self.btn_toggle_blink)
        disc_layout.addLayout(reg_btn_row)

        self.align_progress = QProgressBar()
        self.align_progress.setRange(0, 0)
        self.align_progress.hide()
        disc_layout.addWidget(self.align_progress)

        diff_btn_row = QHBoxLayout()
        diff_btn_row.setSpacing(4)
        self.btn_generate_diff = QPushButton("Detect Moving")
        self.btn_generate_diff.clicked.connect(self._run_diffing)
        diff_btn_row.addWidget(self.btn_generate_diff)

        self.lbl_speed_val = QLabel("500ms")
        self.lbl_speed_val.setFont(mono_font)
        diff_btn_row.addWidget(self.lbl_speed_val)
        disc_layout.addLayout(diff_btn_row)

        self.diff_progress = QProgressBar()
        self.diff_progress.setRange(0, 0)
        self.diff_progress.hide()
        disc_layout.addWidget(self.diff_progress)

        self.candidate_table = QTableWidget()
        self.candidate_table.setColumnCount(5)
        self.candidate_table.setHorizontalHeaderLabels(["ID", "X, Y", "S/N", "Scale", "Conf"])
        self.candidate_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.candidate_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.candidate_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.candidate_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.candidate_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.candidate_table.verticalHeader().setVisible(False)
        self.candidate_table.verticalHeader().setDefaultSectionSize(18)
        self.candidate_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.candidate_table.itemDoubleClicked.connect(self._on_candidate_double_clicked)
        self.candidate_table.setMinimumHeight(100)
        disc_layout.addWidget(self.candidate_table)

        layout.addWidget(group_discovery)

        layout.addStretch()

        scroll_area.setWidget(container)
        root_layout.addWidget(scroll_area)

    def set_fits_image(self, fits_img: FITSImage):
        """Set loaded Reference FITS image (Frame A)."""
        self.fits_image_a = fits_img
        if fits_img.metadata.date_obs:
            self.txt_obs_date.setText(fits_img.metadata.date_obs.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3])
        if fits_img.metadata.filter_name:
            self.txt_filter_band.setText(fits_img.metadata.filter_name[0].upper())
        if fits_img.metadata.object_name:
            self.txt_target_desig.setText(fits_img.metadata.object_name)

    def set_target_fits_image(self, fits_img: FITSImage, stars_b: Optional[List[ReferenceStar]] = None):
        """Set loaded Target FITS image (Frame B)."""
        self.fits_image_b = fits_img
        self.reference_stars_b = stars_b or []
        self.lbl_align_status.setText("Frame B loaded (Click Align)")
        self.lbl_align_status.setStyleSheet("color: #804000; font-weight: bold;")

    def add_reference_star(self, star: ReferenceStar):
        """Add a reference star."""
        self.reference_stars.append(star)
        self._refresh_star_table()
        self.starsChanged.emit()

    def set_reference_stars(self, stars: List[ReferenceStar]):
        """Replace reference stars with new list."""
        self.reference_stars = list(stars)
        self._refresh_star_table()
        self.starsChanged.emit()

    def set_target_centroid(self, centroid: CentroidResult):
        """Record target object centroid."""
        self.target_centroid = centroid
        self.lbl_target_xy.setText(f"{centroid.x:.2f}, {centroid.y:.2f}")
        self.lbl_target_fwhm.setText(f"SNR: {centroid.snr:.1f}")

        if self.solver.solution is not None:
            self._update_target_astrometry()

        self.targetUpdated.emit(centroid.x, centroid.y, self.target_desig)

    def _update_target_astrometry(self):
        if self.target_centroid is None or self.solver.solution is None:
            return

        ra_deg, dec_deg = self.solver.pixel_to_sky(self.target_centroid.x, self.target_centroid.y)
        self.target_ra_deg = ra_deg
        self.target_dec_deg = dec_deg

        self.lbl_target_ra.setText(f"{format_ra(ra_deg)}")
        self.lbl_target_dec.setText(f"{format_dec(dec_deg)}")

    def _on_target_desig_changed(self, text: str):
        self.target_desig = text.strip() or "TARGET"
        if self.target_centroid:
            self.targetUpdated.emit(self.target_centroid.x, self.target_centroid.y, self.target_desig)

    def _on_blink_toggled(self, checked: bool):
        if checked:
            self.btn_toggle_blink.setText("⏸ Pause")
            self.viewModeRequested.emit(ViewMode.BLINK)
        else:
            self.btn_toggle_blink.setText("▶ Blink")
            self.viewModeRequested.emit(ViewMode.FRAME_A)

    def _run_alignment(self):
        """Execute image registration in background thread."""
        if self.fits_image_a is None or self.fits_image_b is None:
            QMessageBox.warning(self, "Alignment", "Please load both Reference (A) and Target (B) images.")
            return

        active_stars_a = [s for s in self.reference_stars if s.enabled]
        active_stars_b = [s for s in self.reference_stars_b if s.enabled] if self.reference_stars_b else []

        if len(active_stars_a) < 3 and not (self.fits_image_a.wcs and self.fits_image_b.wcs):
            QMessageBox.warning(
                self,
                "Reference Stars Required",
                "At least 3 reference stars or FITS WCS headers are required to register and align the images.\n"
                "Please pick reference stars on Frame A or click 'Gaia DR3 Field'.",
            )
            return

        self.btn_align.setEnabled(False)
        self.align_progress.show()
        self.lbl_align_status.setText("Aligning...")

        self.align_worker = AlignmentWorker(
            fits_a=self.fits_image_a,
            fits_b=self.fits_image_b,
            stars_a=active_stars_a,
            stars_b=active_stars_b,
            sol_a=self.plate_solution,
            sol_b=None,
        )
        self.align_worker.finished.connect(self._on_alignment_finished)
        self.align_worker.error.connect(self._on_alignment_error)
        self.align_worker.start()

    def _on_alignment_finished(self, transform: RegistrationTransform, warped_b: np.ndarray):
        self.align_progress.hide()
        self.btn_align.setEnabled(True)
        self.reg_transform = transform
        self.warped_b_array = warped_b

        self.lbl_align_status.setText(
            f"ALIGNED (ΔX:{transform.translation_x:+.1f} ΔY:{transform.translation_y:+.1f} RMS:{transform.alignment_rms_px:.2f}px)"
        )
        self.lbl_align_status.setStyleSheet("color: #006000; font-weight: bold;")

        # Send warped array to viewer
        self.warpedImageReady.emit(warped_b)
        self.viewModeRequested.emit(ViewMode.FRAME_B)

        QMessageBox.information(
            self,
            "Registration Complete",
            f"Image B aligned to Image A successfully!\n\n"
            f"• Translation Shift: ΔX = {transform.translation_x:+.2f} px, ΔY = {transform.translation_y:+.2f} px\n"
            f"• Relative Rotation: {transform.rotation_deg:.2f}°\n"
            f"• Alignment RMS: {transform.alignment_rms_px:.3f} px (Sub-pixel)\n\n"
            f"View Mode switched to 'Frame B (Aligned)'. You can now toggle Blink Mode or Detect Moving Bodies.",
        )

    def _on_alignment_error(self, err_msg: str):
        self.align_progress.hide()
        self.btn_align.setEnabled(True)
        self.lbl_align_status.setText(f"Failed: {err_msg}")
        self.lbl_align_status.setStyleSheet("color: #800000; font-weight: bold;")
        QMessageBox.critical(self, "Registration Error", f"Failed to register images:\n{err_msg}")

    def _run_diffing(self):
        """Execute photometric subtraction and candidate detection in background thread."""
        if self.fits_image_a is None:
            QMessageBox.warning(self, "No Reference Image", "Please load Reference Image A.")
            return

        target_array = self.warped_b_array if self.warped_b_array is not None else (self.fits_image_b.data if self.fits_image_b else None)
        if target_array is None:
            QMessageBox.warning(self, "No Target Image", "Please load and align Target Image B.")
            return

        self.btn_generate_diff.setEnabled(False)
        self.diff_progress.show()

        self.diff_worker = DiffWorker(self.fits_image_a.data, target_array, threshold_sigma=3.5)
        self.diff_worker.finished.connect(self._on_diff_finished)
        self.diff_worker.error.connect(self._on_diff_error)
        self.diff_worker.start()

    def _on_diff_finished(self, result: DiffResult):
        self.diff_progress.hide()
        self.btn_generate_diff.setEnabled(True)
        self.diff_result = result

        self.diffImageReady.emit(result.diff_abs)
        self.candidatesFound.emit(result.candidates)

        self.candidate_table.setRowCount(len(result.candidates))
        mono_font = QFont("Courier New", 10)

        for row, cand in enumerate(result.candidates):
            item_id = QTableWidgetItem(cand.candidate_id)
            item_id.setFont(mono_font)
            if cand.is_priority:
                item_id.setForeground(QColor("#000080"))
            item_id.setFlags(item_id.flags() ^ Qt.ItemFlag.ItemIsEditable)

            item_xy = QTableWidgetItem(f"{cand.x:.1f},{cand.y:.1f}")
            item_xy.setFont(mono_font)
            item_xy.setFlags(item_xy.flags() ^ Qt.ItemFlag.ItemIsEditable)

            item_snr = QTableWidgetItem(f"{cand.snr:.1f}s")
            item_snr.setFont(mono_font)
            item_snr.setForeground(QColor("#006000" if cand.snr >= 8.0 else "#804000"))
            item_snr.setFlags(item_snr.flags() ^ Qt.ItemFlag.ItemIsEditable)

            item_scale = QTableWidgetItem(cand.scale)
            item_scale.setFont(mono_font)
            item_scale.setFlags(item_scale.flags() ^ Qt.ItemFlag.ItemIsEditable)

            item_conf = QTableWidgetItem(f"{cand.confidence_score * 100:.0f}%")
            item_conf.setFont(mono_font)
            item_conf.setFlags(item_conf.flags() ^ Qt.ItemFlag.ItemIsEditable)

            self.candidate_table.setItem(row, 0, item_id)
            self.candidate_table.setItem(row, 1, item_xy)
            self.candidate_table.setItem(row, 2, item_snr)
            self.candidate_table.setItem(row, 3, item_scale)
            self.candidate_table.setItem(row, 4, item_conf)

        QMessageBox.information(
            self,
            "Diffing Complete",
            f"Difference map computed!\n\n"
            f"• Background Noise: {result.noise_sigma:.2f} ADU\n"
            f"• Moving Candidates Detected: {len(result.candidates)}\n\n"
            f"Double-click any candidate in the list to inspect.",
        )

    def _on_diff_error(self, err_msg: str):
        self.diff_progress.hide()
        self.btn_generate_diff.setEnabled(True)
        QMessageBox.critical(self, "Diffing Error", f"Failed to compute difference map:\n{err_msg}")

    def _on_candidate_double_clicked(self, item: QTableWidgetItem):
        row = item.row()
        if self.diff_result and 0 <= row < len(self.diff_result.candidates):
            cand = self.diff_result.candidates[row]
            self.candidateSelected.emit(cand.x, cand.y)
            target_centroid = CentroidResult(
                x=cand.x,
                y=cand.y,
                fwhm=3.0,
                peak_flux=cand.peak_diff,
                snr=cand.snr,
                success=True,
            )
            self.set_target_centroid(target_centroid)

    def solve_plate(self):
        """Perform rigorous Least-Squares Plate-Constants (LSPC) solution."""
        active_stars = [s for s in self.reference_stars if s.enabled]
        if len(active_stars) < 3:
            QMessageBox.warning(
                self,
                "Insufficient Reference Stars",
                f"Astrometric plate reduction requires at least 3 reference stars.\n"
                f"Currently available: {len(active_stars)} active stars.",
            )
            return

        try:
            sol = self.solver.solve(self.reference_stars)
            self.plate_solution = sol

            self.lbl_sol_scale.setText(f"{sol.pixel_scale_avg_arcsec:.3f} \"/px")
            self.lbl_sol_rotation.setText(f"{sol.rotation_deg:.2f}°")
            self.lbl_sol_rms_tot.setText(f"{sol.rms_total_arcsec:.3f}\" (RMS)")
            self.lbl_sol_stars.setText(f"{sol.num_stars_used}/{sol.num_stars_total} stars (Max: {sol.max_residual_arcsec:.2f}\")")

            self._refresh_star_table()
            self.starsChanged.emit()

            if self.target_centroid:
                self._update_target_astrometry()

            QMessageBox.information(
                self,
                "Plate Solved",
                f"Plate solution converged successfully!\n\n"
                f"• Reference Stars: {sol.num_stars_used}\n"
                f"• Pixel Scale: {sol.pixel_scale_avg_arcsec:.3f} \"/pixel\n"
                f"• Field Rotation: {sol.rotation_deg:.2f}°\n"
                f"• Total RMS Error: {sol.rms_total_arcsec:.3f} arcsec",
            )

        except Exception as e:
            QMessageBox.critical(self, "Plate Solve Error", f"Failed to solve astrometric plate:\n{e}")

    def _refresh_star_table(self):
        self.star_table.blockSignals(True)
        self.star_table.setRowCount(len(self.reference_stars))
        mono_font = QFont("Courier New", 10)

        for row, star in enumerate(self.reference_stars):
            item_id = QTableWidgetItem(star.star_id)
            item_id.setFont(mono_font)
            item_id.setFlags(item_id.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.star_table.setItem(row, 0, item_id)

            item_x = QTableWidgetItem(f"{star.x:.2f}")
            item_x.setFont(mono_font)
            item_x.setFlags(item_x.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.star_table.setItem(row, 1, item_x)

            item_y = QTableWidgetItem(f"{star.y:.2f}")
            item_y.setFont(mono_font)
            item_y.setFlags(item_y.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.star_table.setItem(row, 2, item_y)

            item_ra = QTableWidgetItem(format_ra(star.ra_deg))
            item_ra.setFont(mono_font)
            item_ra.setFlags(item_ra.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.star_table.setItem(row, 3, item_ra)

            item_dec = QTableWidgetItem(format_dec(star.dec_deg))
            item_dec.setFont(mono_font)
            item_dec.setFlags(item_dec.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.star_table.setItem(row, 4, item_dec)

            res_str = f"{star.res_total_arcsec:.2f}\"" if star.solved_ra_deg is not None else "-"
            item_res = QTableWidgetItem(res_str)
            item_res.setFont(mono_font)
            item_res.setFlags(item_res.flags() ^ Qt.ItemFlag.ItemIsEditable)
            if star.solved_ra_deg is not None:
                if star.res_total_arcsec < 0.5:
                    item_res.setForeground(QColor("#006000"))
                elif star.res_total_arcsec < 1.0:
                    item_res.setForeground(QColor("#000080"))
                else:
                    item_res.setForeground(QColor("#804000"))
            self.star_table.setItem(row, 5, item_res)

        self.star_table.blockSignals(False)

    def _on_star_row_double_clicked(self, item: QTableWidgetItem):
        row = item.row()
        if not (0 <= row < len(self.reference_stars)):
            return

        star = self.reference_stars[row]
        centroid = CentroidResult(x=star.x, y=star.y, fwhm=3.0, success=True)
        dlg = StarReferenceDialog(
            centroid=centroid,
            default_id=star.star_id,
            existing_star=star,
            parent=self,
        )
        if dlg.exec() and dlg.result_star:
            self.reference_stars[row] = dlg.result_star
            self._refresh_star_table()
            self.starsChanged.emit()

    def _clear_all_stars(self):
        if not self.reference_stars:
            return
        reply = QMessageBox.question(
            self,
            "Clear Reference Stars",
            "Are you sure you want to remove all reference stars?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.reference_stars.clear()
            self._refresh_star_table()
            self.starsChanged.emit()

    def _fetch_gaia_field(self):
        """Query Gaia DR3 stars across the entire image field."""
        if self.fits_image_a is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load a FITS image first.")
            return

        ra_hint = self.fits_image_a.metadata.ra_approx
        dec_hint = self.fits_image_a.metadata.dec_approx

        if ra_hint is None or dec_hint is None:
            QMessageBox.warning(
                self,
                "Missing Field Coordinates",
                "FITS image does not specify approximate celestial coordinates (RA/DEC in header).",
            )
            return

        scale = self.fits_image_a.metadata.pixel_scale_approx or 1.5
        radius_arcmin = (max(self.fits_image_a.width, self.fits_image_a.height) * scale) / 120.0

        self.btn_auto_gaia.setEnabled(False)
        self.gaia_progress.show()

        self.gaia_worker = FieldGaiaWorker(ra_hint, dec_hint, radius_arcmin=radius_arcmin)
        self.gaia_worker.finished.connect(self._on_gaia_field_finished)
        self.gaia_worker.error.connect(self._on_gaia_field_error)
        self.gaia_worker.start()

    def _on_gaia_field_finished(self, stars: list):
        self.gaia_progress.hide()
        self.btn_auto_gaia.setEnabled(True)

        if not stars:
            QMessageBox.information(self, "Gaia Query", "No Gaia DR3 stars found in the specified field.")
            return

        QMessageBox.information(
            self,
            "Gaia DR3 Field Fetched",
            f"Successfully retrieved {len(stars)} Gaia DR3 catalog stars!",
        )

    def _on_gaia_field_error(self, err_msg: str):
        self.gaia_progress.hide()
        self.btn_auto_gaia.setEnabled(True)
        QMessageBox.critical(self, "Gaia Query Failed", f"Error querying Gaia DR3:\n{err_msg}")

    def _open_mpc_export_dialog(self):
        """Build observation and open MPC 80-column export window."""
        if self.target_ra_deg is None or self.target_dec_deg is None:
            QMessageBox.warning(
                self,
                "Unsolved Target",
                "Please solve the plate and pick a target object to obtain J2000 celestial coordinates.",
            )
            return

        try:
            obs_dt = datetime.fromisoformat(self.txt_obs_date.text().strip())
        except Exception:
            obs_dt = datetime.now(timezone.utc)

        try:
            mag = float(self.txt_target_mag.text().strip()) if self.txt_target_mag.text().strip() else None
        except ValueError:
            mag = None

        obs = MPCObservation(
            designation=self.target_desig,
            utc_time=obs_dt,
            ra_deg=self.target_ra_deg,
            dec_deg=self.target_dec_deg,
            mag=mag,
            band=self.txt_filter_band.text().strip() or "R",
            obs_code=self.txt_obs_code.text().strip() or "500",
            is_discovery=self.chk_discovery.isChecked(),
            note1=" ",
            note2="C",
            catalog_code="Gaia3",
        )

        hdr = MPCObservatoryHeader(
            cod=self.txt_obs_code.text().strip() or "500",
            con=self.txt_observer.text().strip(),
            obs=self.txt_observer.text().strip(),
            mea=self.txt_observer.text().strip(),
            tel="0.4m f/8 Reflector + CCD",
            net="Gaia-DR3",
            ack="Astrimetrique v0.1.0",
        )

        dlg = MPCExportDialog(observations=[obs], header=hdr, parent=self)
        dlg.exec()
