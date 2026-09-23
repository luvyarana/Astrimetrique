"""
MPC 80-Column Export & Verification Dialog.
"""

from typing import List, Optional
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QWidget,
)

from astrimetrique.core.mpc_exporter import (
    MPCObservation,
    MPCObservatoryHeader,
    generate_mpc_report,
    format_mpc_80_col,
)


class MPCExportDialog(QDialog):
    """Dialog displaying validated MPC 80-column observation records."""

    def __init__(
        self,
        observations: List[MPCObservation],
        header: Optional[MPCObservatoryHeader] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.observations = observations
        self.header = header or MPCObservatoryHeader()
        self.setWindowTitle("MPC 80-Column Observation Report")
        self.resize(780, 520)

        layout = QVBoxLayout(self)

        # Header Info Banner
        banner = QLabel(
            "Minor Planet Center (MPC) 80-Column Standard Format\n"
            "Strict fixed-width record with fractional-day UTC, J2000 celestial coordinates, and observatory code."
        )
        banner.setStyleSheet("color: #00F2FE; font-weight: bold; padding: 4px;")
        layout.addWidget(banner)

        # 80-Column Visual Ruler
        ruler_lbl = QLabel("1...5....10...15...20...25...30...35...40...45...50...55...60...65...70...75...80")
        ruler_lbl.setFont(QFont("SF Mono", 11))
        ruler_lbl.setStyleSheet("color: #64748B; background-color: #121620; padding: 3px 6px; border: 1px solid #1E2738;")
        layout.addWidget(ruler_lbl)

        # Text Area
        self.text_area = QPlainTextEdit()
        self.text_area.setFont(QFont("SF Mono", 12))
        self.text_area.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text_area.setStyleSheet(
            "background-color: #0E121A; color: #00F2FE; border: 1px solid #1E2738; padding: 8px;"
        )
        layout.addWidget(self.text_area)

        # Action Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_copy = QPushButton("📋 Copy to Clipboard")
        self.btn_copy.setObjectName("primaryButton")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_layout.addWidget(self.btn_copy)

        self.btn_save = QPushButton("💾 Save to File...")
        self.btn_save.clicked.connect(self._save_to_file)
        btn_layout.addWidget(self.btn_save)

        btn_layout.addStretch()

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

        self._populate_text()

    def _populate_text(self):
        report = generate_mpc_report(self.observations, self.header)
        self.text_area.setPlainText(report)

    def _copy_to_clipboard(self):
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.text_area.toPlainText())
        QMessageBox.information(self, "Copied", "MPC report copied to clipboard.")

    def _save_to_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save MPC Report",
            "mpc_observations.txt",
            "Text Files (*.txt *.mpc);;All Files (*)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text_area.toPlainText())
            QMessageBox.information(self, "Saved", f"MPC report saved to:\n{path}")
