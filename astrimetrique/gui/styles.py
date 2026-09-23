"""
Obsidian Space Theme - Astrimetrique Design System.
Provides deep charcoal dark palettes, neon-cyan accents, and crisp typography.
"""

OBSIDIAN_THEME = """
/* Global Application Styles */
QWidget {
    background-color: #0B0E14;
    color: #E2E8F0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
    selection-background-color: #0088CC;
    selection-color: #FFFFFF;
}

/* Main Window */
QMainWindow {
    background-color: #0B0E14;
}

QMainWindow::separator {
    background-color: #1A2130;
    width: 2px;
    height: 2px;
}

/* ToolBar */
QToolBar {
    background-color: #121620;
    border-bottom: 1px solid #1E2738;
    padding: 4px;
    spacing: 6px;
}

QToolButton {
    background-color: #181E2C;
    border: 1px solid #243046;
    border-radius: 5px;
    padding: 5px 10px;
    color: #E2E8F0;
    font-weight: 500;
}

QToolButton:hover {
    background-color: #202A3D;
    border-color: #00F2FE;
    color: #00F2FE;
}

QToolButton:pressed {
    background-color: #0F1522;
}

QToolButton:checked {
    background-color: #004D73;
    border-color: #00F2FE;
    color: #FFFFFF;
    font-weight: bold;
}

/* Dock Widgets */
QDockWidget {
    color: #E2E8F0;
    font-weight: bold;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget::title {
    background-color: #121620;
    border-bottom: 1px solid #1E2738;
    padding: 6px 10px;
    text-align: left;
    color: #00F2FE;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #1E2738;
    background-color: #0E121A;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #121620;
    color: #94A3B8;
    border: 1px solid #1E2738;
    border-bottom: none;
    padding: 6px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #0E121A;
    color: #00F2FE;
    border-top: 2px solid #00F2FE;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background-color: #181E2C;
    color: #E2E8F0;
}

/* Tables */
QTableWidget {
    background-color: #0E121A;
    gridline-color: #1A2232;
    border: 1px solid #1E2738;
    border-radius: 4px;
    font-family: "SF Mono", "Menlo", "Courier New", monospace;
    font-size: 12px;
}

QTableWidget::item {
    padding: 4px 6px;
    border-bottom: 1px solid #141B26;
}

QTableWidget::item:selected {
    background-color: #003B5C;
    color: #00F2FE;
}

QHeaderView::section {
    background-color: #141924;
    color: #94A3B8;
    padding: 5px;
    border: 1px solid #1E2738;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}

/* Buttons */
QPushButton {
    background-color: #181E2C;
    border: 1px solid #28364F;
    border-radius: 5px;
    padding: 6px 14px;
    color: #E2E8F0;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #222B3F;
    border-color: #00F2FE;
    color: #00F2FE;
}

QPushButton:pressed {
    background-color: #101520;
}

QPushButton:disabled {
    background-color: #12151C;
    border-color: #1A202C;
    color: #4A5568;
}

/* Accent Cyan Button */
QPushButton#primaryButton {
    background-color: #0077A8;
    border: 1px solid #00F2FE;
    color: #FFFFFF;
}

QPushButton#primaryButton:hover {
    background-color: #0099D6;
    border-color: #FFFFFF;
}

/* Accent Green Solve Button */
QPushButton#solveButton {
    background-color: #00875A;
    border: 1px solid #00E676;
    color: #FFFFFF;
}

QPushButton#solveButton:hover {
    background-color: #00A870;
    border-color: #FFFFFF;
}

/* Inputs & Combos */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #121620;
    border: 1px solid #243046;
    border-radius: 4px;
    padding: 5px 8px;
    color: #F1F5F9;
    font-family: "SF Mono", "Menlo", "Courier New", monospace;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #00F2FE;
    background-color: #161D2C;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #121620;
    border: 1px solid #243046;
    selection-background-color: #005A87;
    color: #F1F5F9;
}

/* Group Boxes */
QGroupBox {
    border: 1px solid #1E2738;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: 600;
    color: #00F2FE;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 10px;
}

/* Labels */
QLabel {
    color: #CBD5E1;
}

QLabel#metricValue {
    font-family: "SF Mono", "Menlo", "Courier New", monospace;
    font-weight: bold;
    color: #00F2FE;
}

QLabel#metricLabel {
    color: #94A3B8;
    font-size: 11px;
}

/* Status Bar */
QStatusBar {
    background-color: #0E121A;
    border-top: 1px solid #1A2232;
    color: #94A3B8;
    font-family: "SF Mono", "Menlo", "Courier New", monospace;
    font-size: 12px;
}

/* ScrollBars */
QScrollBar:vertical {
    background-color: #0E121A;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #243046;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background-color: #00F2FE;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #0E121A;
    height: 10px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #243046;
    min-width: 20px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #00F2FE;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""
