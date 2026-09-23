"""
High-Performance Astronomical Image Viewer with Multi-Frame Blinking & Diffing.
Features hardware-accelerated PyQtGraph rendering, locked uniform ZScale dynamic range,
and rapid sub-pixel blinking between aligned astronomical exposures.
"""

from enum import Enum
from typing import Callable, List, Optional, Tuple
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPen
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsSimpleTextItem,
    QGraphicsRectItem,
    QWidget,
    QVBoxLayout,
)

from astrimetrique.core.centroid import CentroidResult, fit_centroid_2d_gaussian
from astrimetrique.core.diff_engine import MovingCandidate
from astrimetrique.core.fits_io import FITSImage
from astrimetrique.core.plate_solver import ReferenceStar
from astrimetrique.core.stretch import StretchMode, apply_stretch, calculate_zscale_limits


class InteractionMode(str, Enum):
    PAN_ZOOM = "Pan & Zoom"
    PICK_STAR = "Pick Reference Star"
    PICK_TARGET = "Pick Target Object"


class ViewMode(str, Enum):
    FRAME_A = "Reference (Epoch 1)"
    FRAME_B = "Target (Epoch 2)"
    DIFF = "Difference Map"
    BLINK = "Blink Mode (A ⟷ B)"


class AstronomicalImageViewer(QWidget):
    """
    Scientific astronomical viewport supporting multi-frame sequence browsing,
    synchronized uniform ZScale dynamic range stretching, and rapid blinking.
    """

    cursorMoved = pyqtSignal(float, float, float)  # (x, y, adu)
    starPicked = pyqtSignal(CentroidResult)
    targetPicked = pyqtSignal(CentroidResult)
    activeFrameChanged = pyqtSignal(str)  # Frame name ("A", "B", "DIFF")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Frame Storage
        self.fits_image_a: Optional[FITSImage] = None  # Reference (Epoch 1)
        self.fits_image_b: Optional[FITSImage] = None  # Target Raw (Epoch 2)
        self.frame_b_aligned: Optional[np.ndarray] = None  # Dedicated Aligned Target array
        self.image_b_warped: Optional[np.ndarray] = None  # Alias for frame_b_aligned
        self.image_diff: Optional[np.ndarray] = None  # Difference map
        
        # View & Interaction State
        self.interaction_mode: InteractionMode = InteractionMode.PICK_STAR
        self.view_mode: ViewMode = ViewMode.FRAME_A
        self.current_blink_frame: str = "A"  # "A" or "B"
        self.stretch_mode: StretchMode = StretchMode.ZSCALE
        self.invert_colors: bool = False
        self.contrast: float = 0.25

        # Locked Uniform ZScale limits across all active sequence frames
        self.z_min: float = 0.0
        self.z_max: float = 65535.0
        self.diff_z_min: float = 0.0
        self.diff_z_max: float = 1000.0

        # Blink Timer
        self.blink_timer = QTimer(self)
        self.blink_interval_ms: int = 500  # Default 0.5s
        self.blink_timer.setInterval(self.blink_interval_ms)
        self.blink_timer.timeout.connect(self._on_blink_tick)

        # UI & Graphics
        self._init_ui()
        self._init_overlays()

        # Marker caches
        self.star_markers: List[pg.GraphicsObject] = []
        self.candidate_markers: List[pg.GraphicsObject] = []
        self.target_marker: Optional[pg.GraphicsObject] = None

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Configure pyqtgraph graphics view
        pg.setConfigOption("background", "#0B0E14")
        pg.setConfigOption("foreground", "#94A3B8")
        pg.setConfigOption("antialias", True)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setAspectLocked(True)
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.invertY(False)  # Astronomical orientation (bottom-left = 0,0)

        # Image Item
        self.img_item = pg.ImageItem()
        self.plot_item.addItem(self.img_item)

        # Crosshair lines
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#00F2FE", width=1, style=Qt.PenStyle.DashLine))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#00F2FE", width=1, style=Qt.PenStyle.DashLine))
        self.plot_item.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_item.addItem(self.crosshair_h, ignoreBounds=True)

        # Centroid fit indicator ring
        self.preview_reticle = pg.QtWidgets.QGraphicsEllipseItem()
        self.preview_reticle.setPen(QPen(QColor("#00F2FE"), 1.5, Qt.PenStyle.SolidLine))
        self.preview_reticle.hide()
        self.plot_item.addItem(self.preview_reticle)

        layout.addWidget(self.plot_widget)

        # Connect mouse events
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.img_item.mouseClickEvent = self._on_image_clicked

    def _init_overlays(self):
        # Frame Badge overlay (shows "FRAME A (EPOCH 1 REF)", "FRAME B (EPOCH 2 ALIGNED)", "DIFF MAP")
        self.badge_group = pg.ItemGroup()
        self.badge_bg = QGraphicsRectItem(12, 12, 190, 26)
        self.badge_bg.setBrush(QBrush(QColor(14, 18, 26, 210)))
        self.badge_bg.setPen(QPen(QColor("#00F2FE"), 1.0))
        self.badge_group.addItem(self.badge_bg)

        self.badge_text = QGraphicsSimpleTextItem("FRAME A (REF)")
        self.badge_text.setPos(20, 16)
        self.badge_text.setBrush(QBrush(QColor("#00F2FE")))
        self.badge_text.setFont(QFont("SF Mono", 10, QFont.Weight.Bold))
        self.badge_group.addItem(self.badge_text)

        self.plot_item.addItem(self.badge_group, ignoreBounds=True)

    @property
    def fits_image(self) -> Optional[FITSImage]:
        """Active reference or displayed FITS image."""
        return self.fits_image_a

    def load_reference_image(self, fits_img: FITSImage):
        """Load Reference Image A (Epoch 1)."""
        self.fits_image_a = fits_img
        self._recompute_uniform_zscale()
        self.set_view_mode(ViewMode.FRAME_A)
        self.plot_item.autoRange()

    def load_target_image(self, fits_img: FITSImage):
        """Load Target Image B (Epoch 2)."""
        self.fits_image_b = fits_img
        self.frame_b_aligned = None  # Reset aligned buffer until alignment runs
        self.image_b_warped = None
        self._recompute_uniform_zscale()
        self.set_view_mode(ViewMode.FRAME_B)

    def set_warped_target_image(self, warped_array: np.ndarray):
        """Set aligned/warped Target Image B array."""
        self.frame_b_aligned = warped_array.astype(np.float32)
        self.image_b_warped = self.frame_b_aligned
        self._recompute_uniform_zscale()
        self.update_display()

    def set_difference_image(self, diff_array: np.ndarray):
        """Set computed difference map."""
        self.image_diff = diff_array.astype(np.float32)
        # Compute stretch limits for diff map
        valid_diff = diff_array[np.isfinite(diff_array)]
        if valid_diff.size > 0:
            self.diff_z_min = float(np.percentile(valid_diff, 1.0))
            self.diff_z_max = float(np.percentile(valid_diff, 99.0))
            if self.diff_z_max <= self.diff_z_min:
                self.diff_z_max = self.diff_z_min + 1.0
        self.set_view_mode(ViewMode.DIFF)

    def _recompute_uniform_zscale(self):
        """
        Compute unified ZScale limits locked across both Reference A and Target B (Aligned).
        This guarantees zero brightness flicker during high-speed blinking.
        """
        arrays_to_pool = []
        if self.fits_image_a is not None:
            arrays_to_pool.append(self.fits_image_a.data.ravel())
        if self.frame_b_aligned is not None:
            arrays_to_pool.append(self.frame_b_aligned.ravel())
        elif self.fits_image_b is not None:
            arrays_to_pool.append(self.fits_image_b.data.ravel())

        if not arrays_to_pool:
            return

        combined_sample = np.concatenate([arr[::max(1, arr.size // 5000)] for arr in arrays_to_pool])
        self.z_min, self.z_max = calculate_zscale_limits(combined_sample, contrast=self.contrast)

    def set_view_mode(self, mode: ViewMode):
        """Switch viewport display mode."""
        self.view_mode = mode

        if mode == ViewMode.BLINK:
            self.blink_timer.start(self.blink_interval_ms)
            self.current_blink_frame = "A"
        else:
            self.blink_timer.stop()

        self.update_display()

    def set_blink_speed(self, interval_ms: int):
        """Update blinking interval in milliseconds (100ms - 2000ms)."""
        self.blink_interval_ms = max(50, min(interval_ms, 3000))
        if self.blink_timer.isActive():
            self.blink_timer.setInterval(self.blink_interval_ms)

    def _on_blink_tick(self):
        """Toggle frame between A and B on timer interval."""
        if self.view_mode != ViewMode.BLINK:
            return
        
        self.current_blink_frame = "B" if self.current_blink_frame == "A" else "A"
        self.update_display()

    def set_stretch_mode(self, mode: StretchMode):
        """Change dynamic stretch algorithm."""
        self.stretch_mode = mode
        self.update_display()

    def set_invert_colors(self, invert: bool):
        """Toggle negative / inverted color display."""
        self.invert_colors = invert
        self.update_display()

    def set_interaction_mode(self, mode: InteractionMode):
        """Set viewer interaction mode."""
        self.interaction_mode = mode
        if mode == InteractionMode.PAN_ZOOM:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def update_display(self):
        """Render active frame to viewport with locked dynamic range."""
        current_data = None
        badge_label = ""
        badge_color = "#00F2FE"

        if self.view_mode == ViewMode.DIFF and self.image_diff is not None:
            current_data = self.image_diff
            badge_label = "DIFFERENCE MAP"
            badge_color = "#FF4B4B"
            vmin, vmax = self.diff_z_min, self.diff_z_max
            self.activeFrameChanged.emit("DIFF")

        elif self.view_mode == ViewMode.FRAME_B or (self.view_mode == ViewMode.BLINK and self.current_blink_frame == "B"):
            if self.frame_b_aligned is not None:
                current_data = self.frame_b_aligned
                badge_label = "FRAME B (EPOCH 2 ALIGNED)"
                badge_color = "#00E676"  # Emerald green for verified alignment
            elif self.fits_image_b is not None:
                current_data = self.fits_image_b.data
                badge_label = "FRAME B (EPOCH 2 UNALIGNED)"
                badge_color = "#FFB703"
            vmin, vmax = self.z_min, self.z_max
            self.activeFrameChanged.emit("B")

        else:  # FRAME_A or BLINK frame A
            if self.fits_image_a is not None:
                current_data = self.fits_image_a.data
                badge_label = "FRAME A (EPOCH 1 REF)"
            badge_color = "#00F2FE"
            vmin, vmax = self.z_min, self.z_max
            self.activeFrameChanged.emit("A")

        if current_data is None:
            return

        # Update Badge Overlay
        self.badge_text.setText(badge_label)
        self.badge_text.setBrush(QBrush(QColor(badge_color)))
        self.badge_bg.setPen(QPen(QColor(badge_color), 1.2))

        # Apply locked uniform stretch
        display_data = apply_stretch(
            current_data,
            mode=self.stretch_mode,
            vmin=vmin if self.stretch_mode == StretchMode.ZSCALE else None,
            vmax=vmax if self.stretch_mode == StretchMode.ZSCALE else None,
            contrast=self.contrast,
            invert=self.invert_colors,
        )

        # Transpose for column-major display in pyqtgraph
        self.img_item.setImage(display_data.T, autoLevels=False)

    def _on_mouse_moved(self, pos):
        """Update crosshair position and emit cursor moved signal."""
        active_img = self.fits_image_a if self.fits_image_a else self.fits_image_b
        if active_img is None:
            return

        mouse_point = self.plot_item.vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()

        self.crosshair_v.setPos(x)
        self.crosshair_h.setPos(y)

        # Read intensity from currently rendered frame
        adu = 0.0
        if self.view_mode == ViewMode.DIFF and self.image_diff is not None:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < self.image_diff.shape[0] and 0 <= ix < self.image_diff.shape[1]:
                adu = float(self.image_diff[iy, ix])
        elif (self.view_mode == ViewMode.FRAME_B or (self.view_mode == ViewMode.BLINK and self.current_blink_frame == "B")) and self.image_b_warped is not None:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < self.image_b_warped.shape[0] and 0 <= ix < self.image_b_warped.shape[1]:
                adu = float(self.image_b_warped[iy, ix])
        else:
            adu = active_img.get_pixel_value(x, y)

        self.cursorMoved.emit(x, y, adu)

    def _on_image_clicked(self, event):
        """Handle mouse click for star picking or target marking."""
        active_img = self.fits_image_a if self.fits_image_a else self.fits_image_b
        if active_img is None:
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.pos()
        click_x = float(pos.x())
        click_y = float(pos.y())

        if not (0 <= click_x < active_img.width and 0 <= click_y < active_img.height):
            return

        if self.interaction_mode in [InteractionMode.PICK_STAR, InteractionMode.PICK_TARGET]:
            # Run sub-pixel centroiding on active frame
            active_array = self.image_b_warped if (self.view_mode == ViewMode.FRAME_B and self.image_b_warped is not None) else active_img.data
            centroid = fit_centroid_2d_gaussian(active_array, click_x, click_y, box_size=17)

            r = max(5.0, centroid.fwhm * 1.5)
            self.preview_reticle.setRect(centroid.x - r, centroid.y - r, 2 * r, 2 * r)
            self.preview_reticle.show()

            if self.interaction_mode == InteractionMode.PICK_STAR:
                self.starPicked.emit(centroid)
            elif self.interaction_mode == InteractionMode.PICK_TARGET:
                self.targetPicked.emit(centroid)

        event.accept()

    def update_star_markers(self, stars: List[ReferenceStar]):
        """Render reference star reticles."""
        for item in self.star_markers:
            self.plot_item.removeItem(item)
        self.star_markers.clear()

        font = QFont("SF Mono", 9, QFont.Weight.Bold)

        for star in stars:
            r = 7.0
            color = QColor("#00E676") if star.enabled else QColor("#FF4B4B")
            pen = QPen(color, 1.5, Qt.PenStyle.SolidLine if star.enabled else Qt.PenStyle.DashLine)

            ellipse = QGraphicsEllipseItem(star.x - r, star.y - r, 2 * r, 2 * r)
            ellipse.setPen(pen)
            self.plot_item.addItem(ellipse)
            self.star_markers.append(ellipse)

            label_text = star.star_id
            if star.solved_ra_deg is not None and star.enabled:
                label_text += f" ({star.res_total_arcsec:.2f}\")"

            text_item = QGraphicsSimpleTextItem(label_text)
            text_item.setPos(star.x + r + 3, star.y - r)
            text_item.setBrush(QBrush(color))
            text_item.setFont(font)
            self.plot_item.addItem(text_item)
            self.star_markers.append(text_item)

    def update_candidate_markers(self, candidates: List[MovingCandidate]):
        """Render transient/moving candidate highlight reticles."""
        for item in self.candidate_markers:
            self.plot_item.removeItem(item)
        self.candidate_markers.clear()

        font = QFont("SF Mono", 9, QFont.Weight.Bold)

        for cand in candidates:
            r = 10.0
            color = QColor("#FF007F") if "Epoch 1" in cand.polarity else QColor("#00F2FE")
            pen = QPen(color, 2.0, Qt.PenStyle.DotLine)

            ellipse = QGraphicsEllipseItem(cand.x - r, cand.y - r, 2 * r, 2 * r)
            ellipse.setPen(pen)
            self.plot_item.addItem(ellipse)
            self.candidate_markers.append(ellipse)

            text_item = QGraphicsSimpleTextItem(f"★ {cand.candidate_id} ({cand.snr:.1f}σ)")
            text_item.setPos(cand.x + r + 4, cand.y - r)
            text_item.setBrush(QBrush(color))
            text_item.setFont(font)
            self.plot_item.addItem(text_item)
            self.candidate_markers.append(text_item)

    def set_target_marker(self, x: float, y: float, label: str = "TARGET"):
        """Render target object reticle."""
        if self.target_marker is not None:
            self.plot_item.removeItem(self.target_marker)
            self.target_marker = None

        group = pg.ItemGroup()
        color = QColor("#FFB703")
        pen = QPen(color, 1.8, Qt.PenStyle.SolidLine)

        r1, r2 = 9.0, 5.0
        c1 = QGraphicsEllipseItem(x - r1, y - r1, 2 * r1, 2 * r1)
        c1.setPen(pen)
        c2 = QGraphicsEllipseItem(x - r2, y - r2, 2 * r2, 2 * r2)
        c2.setPen(pen)
        group.addItem(c1)
        group.addItem(c2)

        text = QGraphicsSimpleTextItem(f"🎯 {label}")
        text.setPos(x + r1 + 4, y - r1)
        text.setBrush(QBrush(color))
        text.setFont(QFont("SF Mono", 10, QFont.Weight.Bold))
        group.addItem(text)

        self.plot_item.addItem(group)
        self.target_marker = group
