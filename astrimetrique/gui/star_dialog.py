"""
Star Reference Entry & Gaia DR3 Matching Dialog.
Enables manual coordinate input and 1-click Gaia DR3 catalog query auto-fill.
"""

from typing import Optional
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QProgressBar,
    QMessageBox,
    QWidget,
)

from astrimetrique.core.centroid import CentroidResult
from astrimetrique.core.gaia_catalog import GaiaStar, query_gaia_cone, find_nearest_gaia_star
from astrimetrique.core.plate_solver import ReferenceStar
from astrimetrique.utils.coordinates import format_dec, format_ra, parse_dec, parse_ra


class GaiaFetchWorker(QThread):
    """Background worker for querying Gaia DR3 without freezing the UI."""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, ra_hint: float, dec_hint: float, radius_arcmin: float = 5.0):
        super().__init__()
        self.ra_hint = ra_hint
        self.dec_hint = dec_hint
        self.radius_arcmin = radius_arcmin

    def run(self):
        try:
            stars = query_gaia_cone(
                self.ra_hint,
                self.dec_hint,
                radius_arcmin=self.radius_arcmin,
                mag_limit=18.5,
                row_limit=50,
            )
            self.finished.emit(stars)
        except Exception as e:
            self.error.emit(str(e))


class StarReferenceDialog(QDialog):
    """
    Dialog for inspecting centroid measurements and assigning catalog coordinates.
    """

    def __init__(
        self,
        centroid: CentroidResult,
        default_id: str = "REF-01",
        approx_ra: Optional[float] = None,
        approx_dec: Optional[float] = None,
        existing_star: Optional[ReferenceStar] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.centroid = centroid
        self.approx_ra = approx_ra
        self.approx_dec = approx_dec
        self.existing_star = existing_star
        self.result_star: Optional[ReferenceStar] = None
        self.worker: Optional[GaiaFetchWorker] = None

        self.setWindowTitle("Reference Star Measurement")
        self.resize(440, 480)
        self.setModal(True)

        self._init_ui(default_id)

    def _init_ui(self, default_id: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Centroid Measurement Summary Box
        meas_group = QGroupBox("Sub-Pixel Centroid Fit")
        meas_layout = QFormLayout(meas_group)

        mono_font = QFont("SF Mono", 11)

        lbl_xy = QLabel(f"X: {self.centroid.x:.3f} px  |  Y: {self.centroid.y:.3f} px")
        lbl_xy.setFont(mono_font)
        lbl_xy.setStyleSheet("color: #00F2FE; font-weight: bold;")
        meas_layout.addRow("Centroid (x, y):", lbl_xy)

        lbl_fwhm = QLabel(f"{self.centroid.fwhm:.2f} px  (Method: {self.centroid.fit_method})")
        lbl_fwhm.setFont(mono_font)
        meas_layout.addRow("FWHM:", lbl_fwhm)

        lbl_snr = QLabel(f"{self.centroid.snr:.1f}  (Peak: {self.centroid.peak_flux:.1f} ADU)")
        lbl_snr.setFont(mono_font)
        meas_layout.addRow("SNR / Peak:", lbl_snr)

        layout.addWidget(meas_group)

        # Catalog Coordinates Box
        cat_group = QGroupBox("Catalog Coordinates (J2000)")
        cat_layout = QFormLayout(cat_group)

        self.txt_id = QLineEdit(self.existing_star.star_id if self.existing_star else default_id)
        cat_layout.addRow("Star ID / Designation:", self.txt_id)

        # RA Input
        initial_ra_str = ""
        if self.existing_star:
            initial_ra_str = format_ra(self.existing_star.ra_deg)
        elif self.approx_ra is not None:
            initial_ra_str = format_ra(self.approx_ra)

        self.txt_ra = QLineEdit(initial_ra_str)
        self.txt_ra.setPlaceholderText("HH MM SS.ss or decimal deg (e.g. 12 34 56.78)")
        cat_layout.addRow("Right Ascension (RA):", self.txt_ra)

        # Dec Input
        initial_dec_str = ""
        if self.existing_star:
            initial_dec_str = format_dec(self.existing_star.dec_deg)
        elif self.approx_dec is not None:
            initial_dec_str = format_dec(self.approx_dec)

        self.txt_dec = QLineEdit(initial_dec_str)
        self.txt_dec.setPlaceholderText("+DD MM SS.s or decimal deg (e.g. +24 15 30.0)")
        cat_layout.addRow("Declination (Dec):", self.txt_dec)

        # Mag Input
        initial_mag_str = f"{self.existing_star.mag:.1f}" if (self.existing_star and self.existing_star.mag) else ""
        self.txt_mag = QLineEdit(initial_mag_str)
        self.txt_mag.setPlaceholderText("e.g. 14.2 (optional)")
        cat_layout.addRow("Magnitude (G / V):", self.txt_mag)

        layout.addWidget(cat_group)

        # Gaia DR3 Auto-Fetch Box
        gaia_group = QGroupBox("Gaia DR3 Auto-Match")
        gaia_layout = QVBoxLayout(gaia_group)

        gaia_desc = QLabel("Automatically fetch precise catalog coordinates from ESA Gaia DR3 via astroquery:")
        gaia_desc.setWordWrap(True)
        gaia_desc.setStyleSheet("color: #94A3B8; font-size: 11px;")
        gaia_layout.addWidget(gaia_desc)

        btn_gaia_layout = QHBoxLayout()
        self.btn_gaia_fetch = QPushButton("✨ Fetch from Gaia DR3")
        self.btn_gaia_fetch.setObjectName("primaryButton")
        self.btn_gaia_fetch.clicked.connect(self._fetch_from_gaia)
        btn_gaia_layout.addWidget(self.btn_gaia_fetch)
        gaia_layout.addLayout(btn_gaia_layout)

        self.gaia_progress = QProgressBar()
        self.gaia_progress.setRange(0, 0)
        self.gaia_progress.hide()
        gaia_layout.addWidget(self.gaia_progress)

        self.lbl_gaia_status = QLabel("")
        self.lbl_gaia_status.setStyleSheet("color: #00F2FE; font-size: 11px;")
        gaia_layout.addWidget(self.lbl_gaia_status)

        layout.addWidget(gaia_group)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton("Add Reference Star" if not self.existing_star else "Save Changes")
        self.btn_ok.setObjectName("primaryButton")
        self.btn_ok.clicked.connect(self._on_save)
        btn_layout.addWidget(self.btn_ok)

        layout.addLayout(btn_layout)

    def _fetch_from_gaia(self):
        """Trigger background query to Gaia DR3."""
        # Check if we have approximate RA/Dec from current inputs or header hint
        try:
            ra_hint = parse_ra(self.txt_ra.text()) if self.txt_ra.text().strip() else self.approx_ra
            dec_hint = parse_dec(self.txt_dec.text()) if self.txt_dec.text().strip() else self.approx_dec
        except Exception:
            ra_hint = self.approx_ra
            dec_hint = self.approx_dec

        if ra_hint is None or dec_hint is None:
            QMessageBox.warning(
                self,
                "Gaia Query",
                "Approximate field coordinates are required to query Gaia DR3. "
                "Please ensure the FITS header contains RA/Dec or enter approximate coordinates.",
            )
            return

        self.btn_gaia_fetch.setEnabled(False)
        self.gaia_progress.show()
        self.lbl_gaia_status.setText("Querying Gaia DR3 online catalog...")

        self.worker = GaiaFetchWorker(ra_hint, dec_hint, radius_arcmin=3.0)
        self.worker.finished.connect(self._on_gaia_finished)
        self.worker.error.connect(self._on_gaia_error)
        self.worker.start()

    def _on_gaia_finished(self, stars: list):
        self.gaia_progress.hide()
        self.btn_gaia_fetch.setEnabled(True)

        if not stars:
            self.lbl_gaia_status.setText("No Gaia DR3 stars found in the immediate vicinity.")
            return

        # Find nearest star
        try:
            target_ra = parse_ra(self.txt_ra.text()) if self.txt_ra.text().strip() else self.approx_ra
            target_dec = parse_dec(self.txt_dec.text()) if self.txt_dec.text().strip() else self.approx_dec
            nearest = find_nearest_gaia_star(target_ra, target_dec, stars, max_separation_arcsec=30.0) or stars[0]
        except Exception:
            nearest = stars[0]

        # Auto-fill fields
        self.txt_id.setText(nearest.source_id)
        self.txt_ra.setText(format_ra(nearest.ra_deg))
        self.txt_dec.setText(format_dec(nearest.dec_deg))
        if nearest.phot_g_mean_mag is not None:
            self.txt_mag.setText(f"{nearest.phot_g_mean_mag:.2f}")

        self.lbl_gaia_status.setText(
            f"Matched {nearest.source_id} (G={nearest.phot_g_mean_mag:.2f}, sep={nearest.separation_arcsec:.2f}\")"
            if nearest.separation_arcsec is not None
            else f"Matched {nearest.source_id}"
        )

    def _on_gaia_error(self, err_msg: str):
        self.gaia_progress.hide()
        self.btn_gaia_fetch.setEnabled(True)
        self.lbl_gaia_status.setText(f"Gaia query failed: {err_msg}")

    def _on_save(self):
        """Validate coordinates and accept dialog."""
        star_id = self.txt_id.text().strip() or "REF"
        try:
            ra_deg = parse_ra(self.txt_ra.text())
            dec_deg = parse_dec(self.txt_dec.text())
        except Exception as e:
            QMessageBox.critical(self, "Invalid Coordinates", f"Could not parse RA or Dec: {e}")
            return

        mag = None
        if self.txt_mag.text().strip():
            try:
                mag = float(self.txt_mag.text())
            except ValueError:
                pass

        self.result_star = ReferenceStar(
            star_id=star_id,
            x=self.centroid.x,
            y=self.centroid.y,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            mag=mag,
            enabled=True,
        )
        self.accept()
