"""
FITS Header Inspector Dialog.
"""

from typing import Dict, Any
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHeaderView,
    QWidget,
)


class FITSHeaderDialog(QDialog):
    """Dialog to search and view all raw FITS header cards."""

    def __init__(self, headers: Dict[str, Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.headers = headers
        self.setWindowTitle("FITS Header Inspector")
        self.resize(600, 500)

        layout = QVBoxLayout(self)

        # Search Bar
        search_layout = QHBoxLayout()
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("Filter keywords or values (e.g. DATE, EXPTIME, WCS)...")
        self.txt_filter.textChanged.connect(self._apply_filter)
        search_layout.addWidget(self.txt_filter)
        layout.addLayout(search_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Keyword", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        # Close Button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        self._populate_table()

    def _populate_table(self, filter_text: str = ""):
        filter_lower = filter_text.strip().lower()
        items = []
        for k, v in self.headers.items():
            k_str, v_str = str(k), str(v)
            if not filter_lower or filter_lower in k_str.lower() or filter_lower in v_str.lower():
                items.append((k_str, v_str))

        self.table.setRowCount(len(items))
        for row, (k, v) in enumerate(items):
            item_k = QTableWidgetItem(k)
            item_k.setFlags(item_k.flags() ^ Qt.ItemFlag.ItemIsEditable)
            item_v = QTableWidgetItem(v)
            item_v.setFlags(item_v.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item_k)
            self.table.setItem(row, 1, item_v)

    def _apply_filter(self, text: str):
        self._populate_table(text)
