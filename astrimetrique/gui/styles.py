"""
Classic Astrometrica & Windows Classic Theme - Astrimetrique Design System.
Provides high-density, industrial scientific layout inspired by Astrometrica and Windows 95/XP.
Pure white data tables, classic gray chrome (#D4D0C8), navy accents (#000080), beveled borders,
and 0px border-radius throughout.
"""

CLASSIC_ASTROMETRICA_THEME = """
/* Global Application Styles - Windows Classic / Astrometrica */
QWidget {
    background-color: #D4D0C8;
    color: #000000;
    font-family: "MS Sans Serif", Tahoma, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    font-size: 11px;
    selection-background-color: #000080;
    selection-color: #FFFFFF;
}

/* Main Window */
QMainWindow {
    background-color: #D4D0C8;
}

QMainWindow::separator {
    background-color: #808080;
    width: 3px;
    height: 3px;
}

/* Menu Bar */
QMenuBar {
    background-color: #D4D0C8;
    border-bottom: 1px solid #808080;
    font-size: 11px;
}

QMenuBar::item {
    background: transparent;
    padding: 3px 8px;
}

QMenuBar::item:selected {
    background-color: #000080;
    color: #FFFFFF;
}

QMenu {
    background-color: #D4D0C8;
    border: 2px outset #FFFFFF;
    padding: 2px;
}

QMenu::item {
    padding: 3px 20px 3px 15px;
}

QMenu::item:selected {
    background-color: #000080;
    color: #FFFFFF;
}

QMenu::separator {
    height: 1px;
    background-color: #808080;
    margin: 3px 2px;
}

/* ToolBar */
QToolBar {
    background-color: #D4D0C8;
    border-top: 1px solid #FFFFFF;
    border-bottom: 1px solid #808080;
    padding: 2px;
    spacing: 3px;
}

QToolButton {
    background-color: #D4D0C8;
    border-top: 1px solid #FFFFFF;
    border-left: 1px solid #FFFFFF;
    border-right: 1px solid #808080;
    border-bottom: 1px solid #808080;
    border-radius: 0px;
    padding: 3px 6px;
    color: #000000;
    font-weight: normal;
    font-size: 11px;
}

QToolButton:hover {
    background-color: #E0DDD5;
}

QToolButton:pressed {
    border-top: 1px solid #808080;
    border-left: 1px solid #808080;
    border-right: 1px solid #FFFFFF;
    border-bottom: 1px solid #FFFFFF;
    background-color: #C0BCB4;
    padding: 4px 5px 2px 7px;
}

QToolButton:checked {
    border-top: 1px solid #808080;
    border-left: 1px solid #808080;
    border-right: 1px solid #FFFFFF;
    border-bottom: 1px solid #FFFFFF;
    background-color: #DFDFDF;
    font-weight: bold;
}

/* Dock Widgets */
QDockWidget {
    color: #000000;
    font-weight: bold;
    font-size: 11px;
}

QDockWidget::title {
    background-color: #000080;
    color: #FFFFFF;
    padding: 3px 6px;
    text-align: left;
    font-weight: bold;
    font-size: 11px;
    border: 1px solid #808080;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #808080;
    background-color: #D4D0C8;
    border-radius: 0px;
    top: -1px;
}

QTabBar::tab {
    background-color: #C0C0C0;
    color: #000000;
    border-top: 1px solid #FFFFFF;
    border-left: 1px solid #FFFFFF;
    border-right: 1px solid #808080;
    border-bottom: 1px solid #808080;
    padding: 3px 10px;
    margin-right: 1px;
    border-radius: 0px;
    font-weight: normal;
    font-size: 11px;
}

QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #000000;
    border-bottom: 1px solid #FFFFFF;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #D4D0C8;
}

/* Tables - Ultra Dense Monospace White Background */
QTableWidget {
    background-color: #FFFFFF;
    color: #000000;
    gridline-color: #B0B0B0;
    border-top: 1px solid #808080;
    border-left: 1px solid #808080;
    border-right: 1px solid #FFFFFF;
    border-bottom: 1px solid #FFFFFF;
    border-radius: 0px;
    font-family: "Courier New", Consolas, "SF Mono", monospace;
    font-size: 11px;
    selection-background-color: #000080;
    selection-color: #FFFFFF;
}

QTableWidget::item {
    padding: 1px 3px;
    border: none;
}

QTableWidget::item:selected {
    background-color: #000080;
    color: #FFFFFF;
}

QHeaderView::section {
    background-color: #D4D0C8;
    color: #000000;
    padding: 2px 4px;
    border-top: 1px solid #FFFFFF;
    border-left: 1px solid #FFFFFF;
    border-right: 1px solid #808080;
    border-bottom: 1px solid #808080;
    font-weight: bold;
    font-size: 10px;
    text-transform: none;
    font-family: "MS Sans Serif", Tahoma, Arial, sans-serif;
}

/* Buttons - Classic Gray Beveled */
QPushButton {
    background-color: #D4D0C8;
    border-top: 1px solid #FFFFFF;
    border-left: 1px solid #FFFFFF;
    border-right: 1px solid #808080;
    border-bottom: 1px solid #808080;
    border-radius: 0px;
    padding: 3px 10px;
    color: #000000;
    font-size: 11px;
    font-weight: normal;
}

QPushButton:hover {
    background-color: #E2DFD8;
}

QPushButton:pressed {
    border-top: 1px solid #808080;
    border-left: 1px solid #808080;
    border-right: 1px solid #FFFFFF;
    border-bottom: 1px solid #FFFFFF;
    background-color: #C0BCB4;
    padding: 4px 9px 2px 11px;
}

QPushButton:disabled {
    background-color: #D4D0C8;
    border: 1px solid #A0A0A0;
    color: #808080;
}

/* Primary / Solve Buttons - Industrial Classic Styling */
QPushButton#primaryButton {
    background-color: #D4D0C8;
    border-top: 1px solid #FFFFFF;
    border-left: 1px solid #FFFFFF;
    border-right: 1px solid #808080;
    border-bottom: 1px solid #808080;
    color: #000080;
    font-weight: bold;
}

QPushButton#primaryButton:hover {
    background-color: #E8E6E0;
}

QPushButton#solveButton {
    background-color: #D4D0C8;
    border-top: 1px solid #FFFFFF;
    border-left: 1px solid #FFFFFF;
    border-right: 1px solid #808080;
    border-bottom: 1px solid #808080;
    color: #006000;
    font-weight: bold;
}

QPushButton#solveButton:hover {
    background-color: #E8E6E0;
}

/* Inputs & Combos - Inset White/Light Gray Fields */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #EBEBEB;
    color: #000000;
    border-top: 1px solid #808080;
    border-left: 1px solid #808080;
    border-right: 1px solid #FFFFFF;
    border-bottom: 1px solid #FFFFFF;
    border-radius: 0px;
    padding: 2px 4px;
    font-family: "Courier New", Consolas, "SF Mono", monospace;
    font-size: 11px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    background-color: #FFFFFF;
    border-top: 1px solid #000080;
    border-left: 1px solid #000080;
    border-right: 1px solid #808080;
    border-bottom: 1px solid #808080;
}

QComboBox::drop-down {
    border-left: 1px solid #808080;
    background-color: #D4D0C8;
    width: 16px;
}

QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #808080;
    selection-background-color: #000080;
    selection-color: #FFFFFF;
    color: #000000;
    font-family: "Courier New", Consolas, "SF Mono", monospace;
    font-size: 11px;
}

/* Group Boxes - Classic Astrometrica Panel Frames */
QGroupBox {
    border: 1px solid #808080;
    border-radius: 0px;
    margin-top: 8px;
    padding-top: 8px;
    padding-left: 4px;
    padding-right: 4px;
    padding-bottom: 4px;
    font-weight: bold;
    color: #000080;
    font-size: 11px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    left: 8px;
    background-color: #D4D0C8;
}

/* Labels */
QLabel {
    color: #000000;
    font-size: 11px;
}

QLabel#metricValue {
    font-family: "Courier New", Consolas, "SF Mono", monospace;
    font-weight: bold;
    color: #000080;
    font-size: 11px;
}

QLabel#metricLabel {
    color: #303030;
    font-size: 10px;
}

/* Status Bar */
QStatusBar {
    background-color: #D4D0C8;
    border-top: 1px solid #808080;
    color: #000000;
    font-family: "Courier New", Consolas, "SF Mono", monospace;
    font-size: 11px;
    padding: 2px;
}

QStatusBar::item {
    border: 1px inset #FFFFFF;
    padding: 1px 4px;
}

/* ScrollBars - Classic Thin Windows Gray */
QScrollBar:vertical {
    background-color: #D4D0C8;
    border: 1px solid #808080;
    width: 14px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #D4D0C8;
    border-top: 1px solid #FFFFFF;
    border-left: 1px solid #FFFFFF;
    border-right: 1px solid #808080;
    border-bottom: 1px solid #808080;
    min-height: 16px;
    border-radius: 0px;
}

QScrollBar::handle:vertical:hover {
    background-color: #E2DFD8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #D4D0C8;
    border: 1px solid #808080;
    height: 14px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #D4D0C8;
    border-top: 1px solid #FFFFFF;
    border-left: 1px solid #FFFFFF;
    border-right: 1px solid #808080;
    border-bottom: 1px solid #808080;
    min-width: 16px;
    border-radius: 0px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #E2DFD8;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* CheckBox & Radio */
QCheckBox, QRadioButton {
    spacing: 4px;
    font-size: 11px;
}

/* Progress Bar */
QProgressBar {
    border: 1px inset #808080;
    background-color: #FFFFFF;
    text-align: center;
    color: #000000;
    font-family: "Courier New", Consolas, monospace;
    font-size: 10px;
    height: 14px;
}

QProgressBar::chunk {
    background-color: #000080;
}
"""

# Alias for backward compatibility
OBSIDIAN_THEME = CLASSIC_ASTROMETRICA_THEME
