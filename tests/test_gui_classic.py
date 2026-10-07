"""
Unit test for Classic Astrometrica GUI layout and styling.
"""

import os
import pytest
from PyQt6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_classic_astrometrica_main_window(qapp):
    from astrimetrique.gui.main_window import MainWindow
    from astrimetrique.gui.styles import CLASSIC_ASTROMETRICA_THEME

    win = MainWindow()
    assert win is not None
    assert "Classic Astrometrica" in win.windowTitle()
    assert win.styleSheet() == CLASSIC_ASTROMETRICA_THEME

    # Verify top toolbar buttons
    action_texts = [a.text() for a in win.toolbar.actions() if a.text()]
    assert "Load Reference" in action_texts
    assert "Load Target" in action_texts
    assert "Align" in action_texts
    assert "Solve Plate" in action_texts
    assert "Export MPC" in action_texts

    # Verify Astrometrica stacked panels
    cp = win.control_panel
    assert cp.star_table.columnCount() == 6
    assert cp.star_table.horizontalHeaderItem(0).text() == "ID"
    assert cp.star_table.horizontalHeaderItem(5).text() == 'Res (")'
    assert cp.txt_target_desig is not None
    assert cp.lbl_sol_scale is not None
    assert cp.txt_obs_code.text() == "500"

    # Verify image viewer crosshair & overlay
    viewer = win.viewer
    assert viewer.crosshair_v is not None
    assert viewer.crosshair_h is not None
    assert viewer.coord_overlay is not None
    assert "X:" in viewer.coord_overlay.text()
