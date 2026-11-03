#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qt-based dam resistivity visualization application using VTK Qt interface"""

import sys
import os
import html
import time
import logging
from collections import deque
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
os.environ.setdefault("QT_API", "pyside6")

from qtpy.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QSlider, QLabel, QPushButton, 
                            QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
                            QFileDialog, QMessageBox, QCheckBox, QProgressBar,
                            QSplitter, QFrame, QRadioButton, QTabWidget, QStyle,
                            QToolButton, QPlainTextEdit)

from qtpy.QtCore import Qt, QTimer, Signal, QEventLoop, QPointF, QRectF, QObject, QEvent, QSize, QByteArray
from qtpy.QtGui import (QFont, QPalette, QKeySequence, QShortcut, QPainter, QPen, QBrush,
                        QColor, QLinearGradient, QPolygonF, QFontDatabase, QAction, QIcon, QPixmap)
from qtpy.QtSvg import QSvgRenderer

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import pyvista as pv
import pyqtgraph as pg

import damvis_utils as dvu

log = logging.getLogger("damviz")
log.setLevel(logging.INFO)

# Number of prepared frames kept in memory (roughly 10-20 MB each)
FRAME_CACHE_SIZE = 12

# Number of frames after the displayed one that are prepared in the background
PREFETCH_FRAMES = 3

# Probing: samples along the view ray, and the opacity at which the volume counts as hit
PROBE_RAY_SAMPLES = 512
PROBE_OPACITY_THRESHOLD = 0.1

# Camera directions (from focal point to camera) and view-up vectors of the view presets
CAMERA_PRESETS = {
    'Top': ((0, 0, 1), (0, 1, 0)),
    'Front': ((0, -1, 0), (0, 0, 1)),
    'Side': ((1, 0, 0), (0, 0, 1)),
    'Iso': ((1, -1, 1), (0, 0, 1)),
}

# Delays that coalesce rapid control changes (e.g. dragging a slider)
DATA_DEBOUNCE_MS = 150
ISOSURFACE_DEBOUNCE_MS = 200
SLICE_DEBOUNCE_MS = 60

# Default opacity curve as (relative position, opacity): emphasises the middle of the range
DEFAULT_OPACITY_POINTS = [(0.0, 0.0), (0.24, 0.2), (0.47, 1.0), (0.82, 0.1), (0.94, 0.0), (1.0, 0.0)]

# Opacity presets as (relative position, opacity)
OPACITY_PRESETS = {
    'Full': [(0.0, 1.0), (1.0, 1.0)],
    'Linear Up': [(0.0, 0.0), (1.0, 1.0)],
    'Linear Down': [(0.0, 1.0), (1.0, 0.0)],
    'Max Middle': [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
    'Max Sides': [(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)],
    'Hide Low': [(0.0, 0.0), (0.4, 0.0), (0.6, 0.8), (1.0, 0.8)],
    'Hide High': [(0.0, 0.8), (0.4, 0.8), (0.6, 0.0), (1.0, 0.0)],
}

# Colormap control points as (relative position, r, g, b)
COLORMAPS = {
    'RdYlBu_r': [(0.0, 0.0, 0.0, 1.0), (0.3, 0.0, 1.0, 1.0), (0.5, 1.0, 1.0, 0.0),
                 (0.7, 1.0, 0.5, 0.0), (1.0, 1.0, 0.0, 0.0)],
    'viridis': [(0.0, 0.267, 0.004, 0.329), (0.25, 0.229, 0.322, 0.545), (0.5, 0.127, 0.566, 0.550),
                (0.75, 0.369, 0.788, 0.382), (1.0, 0.993, 0.906, 0.144)],
    'plasma': [(0.0, 0.050, 0.030, 0.529), (0.25, 0.494, 0.016, 0.655), (0.5, 0.808, 0.067, 0.472),
               (0.75, 0.965, 0.451, 0.176), (1.0, 0.984, 0.906, 0.145)],
    'inferno': [(0.0, 0.000, 0.000, 0.014), (0.25, 0.341, 0.062, 0.429), (0.5, 0.733, 0.216, 0.329),
                (0.75, 0.976, 0.576, 0.176), (1.0, 0.988, 0.998, 0.645)],
    'jet': [(0.0, 0.0, 0.0, 0.5), (0.2, 0.0, 0.0, 1.0), (0.4, 0.0, 1.0, 1.0),
            (0.6, 1.0, 1.0, 0.0), (0.8, 1.0, 0.0, 0.0), (1.0, 0.5, 0.0, 0.0)],
    'rainbow': [(0.0, 0.5, 0.0, 1.0), (0.17, 0.0, 0.0, 1.0), (0.33, 0.0, 1.0, 1.0), (0.5, 0.0, 1.0, 0.0),
                (0.67, 1.0, 1.0, 0.0), (0.83, 1.0, 0.5, 0.0), (1.0, 1.0, 0.0, 0.0)],
}


def clip_segment_to_bounds(start, end, bounds):
    """Part of the segment start-end inside an axis-aligned box, as (start, end), or None"""
    direction = end - start
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        low, high = bounds[2 * axis], bounds[2 * axis + 1]
        if abs(direction[axis]) < 1e-12:
            if not low <= start[axis] <= high:
                return None
            continue
        t0 = (low - start[axis]) / direction[axis]
        t1 = (high - start[axis]) / direction[axis]
        t_min = max(t_min, min(t0, t1))
        t_max = min(t_max, max(t0, t1))
    if t_min > t_max:
        return None
    return start + t_min * direction, start + t_max * direction


# Toolbar icons: Fluent UI System Icons (MIT license, see icons/LICENSE),
# https://github.com/microsoft/fluentui-system-icons
ICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
ICON_SIZE = 20


def load_icon(name, color, checked_color):
    """Load a monochrome SVG icon from ICONS_DIR.

    It is drawn in color, or in checked_color when its (checkable) button is
    on, so it stays readable on the highlighted button background.
    """
    with open(os.path.join(ICONS_DIR, f"{name}.svg"), encoding="utf-8") as f:
        svg = f.read()

    icon = QIcon()
    for state, state_color in ((QIcon.State.Off, color), (QIcon.State.On, checked_color)):
        # The Fluent icons have no fill color set (black); use the palette color instead
        colored = svg.replace("<svg ", f'<svg fill="{state_color.name()}" ', 1)
        renderer = QSvgRenderer(QByteArray(colored.encode("utf-8")))
        for scale in (1, 2):  # Sharp on high-DPI screens
            pixmap = QPixmap(ICON_SIZE * scale, ICON_SIZE * scale)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            pixmap.setDevicePixelRatio(scale)
            icon.addPixmap(pixmap, QIcon.Mode.Normal, state)
    return icon


def build_color_function(color_func, colormap, data_min, data_max):
    """Fill a vtkColorTransferFunction with a colormap spread over the data range"""
    color_func.RemoveAllPoints()
    for position, r, g, b in COLORMAPS.get(colormap, COLORMAPS['RdYlBu_r']):
        color_func.AddRGBPoint(data_min + position * (data_max - data_min), r, g, b)


class VTKVisualizationWidget(QWidget):
    """Widget containing VTK render window and interactor"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        
        self.setLayout(layout)
        
        # Initialize VTK components
        self.renderer = vtk.vtkRenderer()
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Trackball interaction: the camera only moves while dragging, in
        # proportion to the mouse movement (the default joystick style keeps
        # rotating while a button is held)
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor_style.SetMouseWheelMotionFactor(0.5)  # Finer zoom steps
        # Disable VTK's single-key commands (w = wireframe, e/q = exit, 3 = stereo, ...);
        # the application provides its own shortcuts. An observer replaces the default handler.
        self.interactor_style.AddObserver("CharEvent", lambda obj, event: None)
        self.interactor.SetInteractorStyle(self.interactor_style)

        # Set up renderer with enhanced lighting
        self.renderer.SetBackground(0.1, 0.2, 0.3)  # Standard VTK background color
        
        # Add enhanced lighting setup (default to Enhanced)
        self.setup_enhanced_lighting('Enhanced')

        # Set up camera for better initial view
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(10, 10, 30)
        camera.SetFocalPoint(10, 12, 24.5)  # Center of typical bounds
        camera.SetViewUp(0, 0, 1)
        
        # Color bar actor, created together with the scene
        self.scalar_bar_actor = None
        
        # Initialize the interactor
        self.interactor.Initialize()
        self.interactor.Start()
    
    def setup_enhanced_lighting(self, quality='Enhanced'):
        """Set up lighting based on quality level"""
        # Remove existing lights
        self.renderer.RemoveAllLights()
        
        if quality == 'Enhanced':
            # Multi-light setup for best visual quality
            self.renderer.SetAutomaticLightCreation(False)
            
            # Key light (main directional light)
            key_light = vtk.vtkLight()
            key_light.SetPosition(10, 10, 10)
            key_light.SetFocalPoint(0, 0, 0)
            key_light.SetColor(1.0, 1.0, 0.95)  # Slightly warm white
            key_light.SetIntensity(0.8)
            key_light.SetConeAngle(60)
            self.renderer.AddLight(key_light)
            
            # Fill light (softer, opposite side)
            fill_light = vtk.vtkLight()
            fill_light.SetPosition(-5, 5, 8)
            fill_light.SetFocalPoint(0, 0, 0)
            fill_light.SetColor(0.8, 0.9, 1.0)  # Cool blue tint
            fill_light.SetIntensity(0.4)
            fill_light.SetConeAngle(80)
            self.renderer.AddLight(fill_light)
            
            # Back light (rim lighting)
            back_light = vtk.vtkLight()
            back_light.SetPosition(-8, -8, 5)
            back_light.SetFocalPoint(0, 0, 0)
            back_light.SetColor(1.0, 0.9, 0.8)  # Warm rim light
            back_light.SetIntensity(0.3)
            back_light.SetConeAngle(120)
            self.renderer.AddLight(back_light)
            
            # Ambient light for overall illumination
            ambient_light = vtk.vtkLight()
            ambient_light.SetLightTypeToSceneLight()
            ambient_light.SetColor(0.4, 0.4, 0.5)  # Subtle blue ambient
            ambient_light.SetIntensity(0.1)
            self.renderer.AddLight(ambient_light)
            
        elif quality == 'Standard':
            # Two-light setup - key and fill
            self.renderer.SetAutomaticLightCreation(False)
            
            # Key light
            key_light = vtk.vtkLight()
            key_light.SetPosition(10, 10, 10)
            key_light.SetFocalPoint(0, 0, 0)
            key_light.SetColor(1.0, 1.0, 1.0)
            key_light.SetIntensity(0.7)
            self.renderer.AddLight(key_light)
            
            # Fill light
            fill_light = vtk.vtkLight()
            fill_light.SetPosition(-5, 5, 8)
            fill_light.SetFocalPoint(0, 0, 0)
            fill_light.SetColor(0.9, 0.9, 1.0)
            fill_light.SetIntensity(0.3)
            self.renderer.AddLight(fill_light)
            
        else:  # 'Minimal'
            # Use VTK's automatic lighting (single light)
            self.renderer.SetAutomaticLightCreation(True)
    
    def create_scalar_bar(self, color_function):
        """Create the scalar bar (color scale) for the given color function.
        
        The bar follows later changes of the color function automatically.
        """
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(color_function)
        
        # Position and size (normalized viewport coordinates)
        scalar_bar.SetPosition(0.9, 0.25)
        scalar_bar.SetWidth(0.08)
        scalar_bar.SetHeight(0.5)
        scalar_bar.SetBarRatio(0.3)         # Bar takes this fraction of the width, labels the rest

        # Use the font sizes below instead of scaling the text to fill the bar
        scalar_bar.UnconstrainedFontSizeOn()

        title_prop = scalar_bar.GetTitleTextProperty()
        label_prop = scalar_bar.GetLabelTextProperty()

        # Title styling
        title_prop.SetColor(1, 1, 1)        # White text
        title_prop.SetFontSize(13)
        title_prop.SetFontFamilyToArial()
        title_prop.BoldOn()
        title_prop.ItalicOff()
        title_prop.ShadowOn()               # Add shadow for better contrast

        # Label styling
        label_prop.SetColor(1, 1, 1)        # White text
        label_prop.SetFontSize(12)
        label_prop.SetFontFamilyToArial()
        label_prop.BoldOff()
        label_prop.ItalicOff()
        label_prop.ShadowOn()               # Add shadow for better contrast
        
        # Additional scalar bar formatting
        scalar_bar.SetNumberOfLabels(6)     # More labels for better precision
        scalar_bar.SetLabelFormat("%.2f")   # Two decimal places
        
        self.scalar_bar_actor = scalar_bar
        self.renderer.AddActor2D(self.scalar_bar_actor)
        return scalar_bar
    
    def reset_camera(self):
        """Reset camera to fit all objects"""
        self.renderer.ResetCamera()
        self.render_window.Render()
    
    def render(self):
        """Render the scene"""
        self.render_window.Render()

    def capture_screenshot(self, filename):
        """Capture screenshot of the current render window"""
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()


class OpacityEditor(QWidget):
    """Editor for the opacity transfer function.

    Shows a histogram of the rendered values with the opacity curve on top and
    the colormap below. Drag points to change the curve, click to add a point
    and right-click (or press Delete) to remove one. Points are stored as
    (position, opacity) pairs, where position runs from 0 (data min) to 1
    (data max).
    """

    changed = Signal()
    hover_text_changed = Signal(str)

    MARGIN = 8
    COLORBAR_HEIGHT = 12
    POINT_RADIUS = 5
    HIT_RADIUS = 9
    MIN_GAP = 0.002  # Smallest distance between neighbouring points
    HISTOGRAM_BINS = 100

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points = list(DEFAULT_OPACITY_POINTS)
        self._colormap = 'RdYlBu_r'
        self._data_range = (0.0, 1.0)
        self._histogram = None       # Bar heights scaled to [0, 1]
        self._histogram_key = None
        self._drag_index = None
        self._selected_index = None

        self.setMinimumHeight(180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    # --- Data -------------------------------------------------------------

    def points(self):
        return list(self._points)

    def set_points(self, points):
        """Replace the curve; the first and last point are pinned to the range ends"""
        points = sorted((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)) for x, y in points)
        points[0] = (0.0, points[0][1])
        points[-1] = (1.0, points[-1][1])
        self._points = points
        self._selected_index = None
        self.update()
        self.changed.emit()

    def set_colormap(self, colormap):
        if colormap != self._colormap:
            self._colormap = colormap
            self.update()

    def set_histogram(self, values, data_min, data_max):
        """Show the distribution of values over [data_min, data_max]"""
        self._data_range = (data_min, data_max)
        key = (id(values), data_min, data_max)
        if key == self._histogram_key:
            return
        self._histogram_key = key

        if values is None or len(values) == 0 or data_max <= data_min:
            self._histogram = None
        else:
            counts, _ = np.histogram(values, bins=self.HISTOGRAM_BINS, range=(data_min, data_max))
            # Square-root heights keep small bins visible without flattening the shape
            heights = np.sqrt(counts)
            self._histogram = heights / heights.max() if heights.max() > 0 else None
        self.update()

    def opacity_at(self, x):
        xs, ys = zip(*self._points)
        return float(np.interp(x, xs, ys))

    # --- Geometry ---------------------------------------------------------

    def plot_rect(self):
        m = self.MARGIN
        return QRectF(m, m, self.width() - 2 * m, self.height() - 2 * m - self.COLORBAR_HEIGHT - 4)

    def colorbar_rect(self):
        plot = self.plot_rect()
        return QRectF(plot.left(), plot.bottom() + 4, plot.width(), self.COLORBAR_HEIGHT)

    def to_widget(self, x, y):
        rect = self.plot_rect()
        return QPointF(rect.left() + x * rect.width(), rect.bottom() - y * rect.height())

    def from_widget(self, pos):
        rect = self.plot_rect()
        x = (pos.x() - rect.left()) / rect.width()
        y = (rect.bottom() - pos.y()) / rect.height()
        return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)

    def point_at(self, pos):
        """Index of the point under pos, or None"""
        best, best_distance = None, self.HIT_RADIUS
        for i, (x, y) in enumerate(self._points):
            p = self.to_widget(x, y)
            distance = ((p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2) ** 0.5
            if distance <= best_distance:
                best, best_distance = i, distance
        return best

    def colormap_gradient(self, rect, alpha=255):
        gradient = QLinearGradient(rect.left(), 0, rect.right(), 0)
        for position, r, g, b in COLORMAPS.get(self._colormap, COLORMAPS['RdYlBu_r']):
            color = QColor.fromRgbF(r, g, b)
            color.setAlpha(alpha)
            gradient.setColorAt(position, color)
        return gradient

    # --- Painting ---------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        rect = self.plot_rect()

        painter.fillRect(rect, palette.color(QPalette.ColorRole.Base))

        # Opacity grid lines
        grid_pen = QPen(palette.color(QPalette.ColorRole.Mid))
        grid_pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        for level in (0.25, 0.5, 0.75):
            y = rect.bottom() - level * rect.height()
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        # Histogram
        if self._histogram is not None:
            bar_color = palette.color(QPalette.ColorRole.Text)
            bar_color.setAlpha(45)
            bar_width = rect.width() / len(self._histogram)
            for i, height in enumerate(self._histogram):
                if height > 0:
                    bar_height = height * rect.height()
                    painter.fillRect(QRectF(rect.left() + i * bar_width, rect.bottom() - bar_height,
                                            bar_width, bar_height), bar_color)

        # Area under the curve, colored like the rendering
        curve = QPolygonF([self.to_widget(x, y) for x, y in self._points])
        area = QPolygonF(curve)
        area.append(QPointF(rect.right(), rect.bottom()))
        area.append(QPointF(rect.left(), rect.bottom()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.colormap_gradient(rect, alpha=110)))
        painter.drawPolygon(area)

        # Curve and points
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Text), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolyline(curve)
        for i, point in enumerate(curve):
            selected = i == self._selected_index
            painter.setBrush(palette.color(QPalette.ColorRole.Highlight if selected else QPalette.ColorRole.Base))
            painter.drawEllipse(point, self.POINT_RADIUS, self.POINT_RADIUS)

        painter.setPen(palette.color(QPalette.ColorRole.Mid))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        # Colormap strip
        colorbar = self.colorbar_rect()
        painter.fillRect(colorbar, QBrush(self.colormap_gradient(colorbar)))
        painter.drawRect(colorbar)

    # --- Interaction ------------------------------------------------------

    def move_point(self, index, pos):
        x, y = self.from_widget(pos)
        if index == 0:
            x = 0.0
        elif index == len(self._points) - 1:
            x = 1.0
        else:
            x = min(max(x, self._points[index - 1][0] + self.MIN_GAP),
                    self._points[index + 1][0] - self.MIN_GAP)
        self._points[index] = (x, y)
        self.update()
        self.changed.emit()

    def remove_point(self, index):
        """Remove an inner point (the end points are always kept)"""
        if index is not None and 0 < index < len(self._points) - 1:
            del self._points[index]
            self._selected_index = None
            self.update()
            self.changed.emit()

    def mousePressEvent(self, event):
        pos = event.position()
        index = self.point_at(pos)

        if event.button() == Qt.MouseButton.RightButton:
            self.remove_point(index)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if index is None:
            # Add a new point between its neighbours
            x, y = self.from_widget(pos)
            if not self.MIN_GAP < x < 1.0 - self.MIN_GAP:
                return
            index = next(i for i, (px, _) in enumerate(self._points) if px > x)
            self._points.insert(index, (x, y))

        self._drag_index = self._selected_index = index
        self.move_point(index, pos)

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._drag_index is not None:
            self.move_point(self._drag_index, pos)
        else:
            hovering = self.point_at(pos) is not None
            self.setCursor(Qt.CursorShape.PointingHandCursor if hovering else Qt.CursorShape.CrossCursor)

        x, _ = self.from_widget(pos)
        data_min, data_max = self._data_range
        value = data_min + x * (data_max - data_min)
        self.hover_text_changed.emit(f"Value {value:.3f}   opacity {self.opacity_at(x):.2f}")

    def mouseReleaseEvent(self, event):
        self._drag_index = None

    def leaveEvent(self, event):
        self.hover_text_changed.emit("")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.remove_point(self._selected_index)
        else:
            super().keyPressEvent(event)


class ProbePanel(QWidget):
    """Value at a probed point: readout for the current frame and time series over all frames"""

    frame_clicked = Signal(int)
    clear_requested = Signal()

    HINT = "Click on visible data in the 3D view to probe its value and see it over time."

    def __init__(self, parent=None):
        super().__init__(parent)
        self._point = None
        self._scalar_name = ""
        self._values = {}  # frame index -> value (NaN if outside the mesh, missing while computing)
        self._frames = []
        self._current_frame = None

        layout = QVBoxLayout()

        self.info_label = QLabel(self.HINT)
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.info_label)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(self.palette().color(QPalette.ColorRole.Base))
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('bottom', "Frame")
        self.plot.setMinimumHeight(220)
        self.curve = self.plot.plot([], [], pen=pg.mkPen('#1f77b4', width=2), symbol='o', symbolSize=7,
                                    symbolBrush='#1f77b4', connect='finite')
        self.curve.sigPointsClicked.connect(self.on_points_clicked)
        self.current_line = pg.InfiniteLine(angle=90, movable=False,
                                            pen=pg.mkPen('#d03030', width=1, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.current_line)
        self.current_line.setVisible(False)
        layout.addWidget(self.plot)

        buttons = QHBoxLayout()
        self.copy_button = QPushButton("Copy Data")
        self.copy_button.setToolTip("Copy the time series as CSV to the clipboard")
        self.clear_button = QPushButton("Clear Probe")
        buttons.addWidget(self.copy_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        layout.addStretch()
        self.setLayout(layout)

        self.copy_button.clicked.connect(self.copy_data)
        self.clear_button.clicked.connect(self.clear_requested)
        self.set_enabled(False)

    def set_enabled(self, enabled):
        self.copy_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)

    def set_probe(self, point, scalar_name, frames):
        """Start a new probe; values are added with set_value as they become available"""
        self._point = point
        self._scalar_name = scalar_name
        self._frames = list(frames)
        self._values = {}
        self.plot.setLabel('left', scalar_name)
        self.set_enabled(True)
        self.refresh()

    def set_value(self, frame, value):
        self._values[frame] = value
        self.refresh()

    def set_current_frame(self, frame):
        self._current_frame = frame
        self.refresh()

    def clear(self):
        self._point = None
        self._values = {}
        self.curve.setData([], [])
        self.current_line.setVisible(False)
        self.info_label.setText(self.HINT)
        self.set_enabled(False)

    def refresh(self):
        if self._point is None:
            return

        frames = [f for f in self._frames if f in self._values]
        self.curve.setData(np.array(frames, dtype=float), np.array([self._values[f] for f in frames], dtype=float))

        if self._current_frame is not None:
            self.current_line.setPos(self._current_frame)
            self.current_line.setVisible(True)

        x, y, z = self._point
        lines = [f"Point: x = {x:.2f}, y = {y:.2f}, z = {z:.2f}"]
        value = self._values.get(self._current_frame)
        if self._current_frame in self._values:
            text = "outside the mesh" if np.isnan(value) else f"{value:.4g}"
            lines.append(f"{self._scalar_name} at frame {self._current_frame}: {text}")
        if len(self._values) < len(self._frames):
            lines.append(f"Computing time series... {len(self._values)}/{len(self._frames)} frames")
        self.info_label.setText("\n".join(lines))

    def on_points_clicked(self, item, points, *args):
        if len(points):
            self.frame_clicked.emit(int(round(points[0].pos().x())))

    def copy_data(self):
        rows = [f"frame,{self._scalar_name}"]
        rows += [f"{f},{self._values[f]}" for f in self._frames if f in self._values]
        QApplication.clipboard().setText("\n".join(rows))
        log.info("Copied %d probe values to the clipboard", len(rows) - 1)


class ControlPanel(QWidget):
    """Control panel for visualization parameters"""
    
    # Signals, grouped by how expensive the resulting update is
    display_changed = Signal()      # transfer functions, visibility, lighting (instant)
    isosurfaces_changed = Signal()  # isosurface geometry (contouring)
    data_changed = Signal()         # scalar, clipping, resampling (worker process)
    slices_changed = Signal()       # slice planes
    
    def __init__(self, parent=None, main_app=None):
        super().__init__(parent)
        self.main_app = main_app
        
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """Set up the control panel UI"""
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.opacity_tab = self.create_opacity_tab()
        self.render_tab = self.create_render_tab()
        self.slicing_tab = self.create_slicing_tab()
        self.probe_panel = ProbePanel()

        # Add tabs to tab widget
        self.tab_widget.addTab(self.render_tab, "Rendering")
        self.tab_widget.addTab(self.opacity_tab, "Opacity")
        self.tab_widget.addTab(self.slicing_tab, "Slicing")
        self.tab_widget.addTab(self.probe_panel, "Probe")
        
        layout.addWidget(self.tab_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def create_opacity_tab(self):
        """Create the Opacity Transfer Function tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Opacity controls group
        opacity_group = QGroupBox("Opacity Transfer Function")
        opacity_layout = QVBoxLayout()
        
        # Create min/max labels at the top
        minmax_layout = QHBoxLayout()
        self.left_label = QLabel("Min")
        self.left_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.left_label.setStyleSheet("font-weight: bold; color: blue;")
        
        self.right_label = QLabel("Max")
        self.right_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.right_label.setStyleSheet("font-weight: bold; color: red;")
        
        minmax_layout.addWidget(self.left_label)
        minmax_layout.addStretch()
        minmax_layout.addWidget(self.right_label)
        
        opacity_layout.addLayout(minmax_layout)
        
        self.opacity_editor = OpacityEditor()
        opacity_layout.addWidget(self.opacity_editor)

        self.opacity_hint_label = QLabel()
        self.opacity_hint_label.setStyleSheet("color: gray;")
        opacity_layout.addWidget(self.opacity_hint_label)
        self.opacity_editor.hover_text_changed.connect(self.on_opacity_hover)
        self.on_opacity_hover("")

        # Opacity preset controls
        presets_layout = QGridLayout()
        preset_names = list(OPACITY_PRESETS) + ['Reverse']
        for i, name in enumerate(preset_names):
            button = QPushButton(name)
            if name == 'Reverse':
                button.setToolTip("Mirror the curve between low and high values")
                button.clicked.connect(self.reverse_opacity)
            else:
                button.clicked.connect(lambda checked=False, name=name: self.apply_opacity_preset(name))
            presets_layout.addWidget(button, i // 4, i % 4)

        opacity_layout.addLayout(presets_layout)
        opacity_group.setLayout(opacity_layout)
        layout.addWidget(opacity_group)
        
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def create_slicing_tab(self):
        """Create the Slicing tab with slice planes and the clipping bounds"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Slice planes through the (clipped) data, positioned within its extent
        slices_group = QGroupBox("Slice Planes")
        slices_layout = QGridLayout()
        self.slice_checkboxes = []
        self.slice_sliders = []
        self.slice_labels = []
        for i, axis in enumerate("XYZ"):
            checkbox = QCheckBox(f"{axis} slice")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1000)
            slider.setValue(500)
            slider.setEnabled(False)
            label = QLabel("")
            label.setMinimumWidth(70)
            checkbox.toggled.connect(slider.setEnabled)
            slices_layout.addWidget(checkbox, i, 0)
            slices_layout.addWidget(slider, i, 1)
            slices_layout.addWidget(label, i, 2)
            self.slice_checkboxes.append(checkbox)
            self.slice_sliders.append(slider)
            self.slice_labels.append(label)
        slices_layout.setColumnStretch(1, 1)
        slices_group.setLayout(slices_layout)
        layout.addWidget(slices_group)

        # Bounds controls group
        bounds_group = QGroupBox("Clipping Bounds")
        bounds_layout = QGridLayout()
        
        bounds_labels = ['X Min:', 'X Max:', 'Y Min:', 'Y Max:', 'Z Min:', 'Z Max:']
        bounds_defaults = [2, 17, 2, 22, 22, 27]
        
        self.bounds_spinboxes = []
        
        for i, (label_text, default_val) in enumerate(zip(bounds_labels, bounds_defaults)):
            label = QLabel(label_text)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1000, 1000)
            spinbox.setValue(default_val)
            spinbox.setDecimals(1)
            
            row = i // 2
            col = (i % 2) * 2
            
            bounds_layout.addWidget(label, row, col)
            bounds_layout.addWidget(spinbox, row, col + 1)
            
            self.bounds_spinboxes.append(spinbox)
        
        self.edit_box_checkbox = QCheckBox("Edit clipping box in 3D view")
        self.edit_box_checkbox.setToolTip("Drag the faces of the box to change the clipping bounds. "
                                          "The data is updated when you release the mouse.")
        bounds_layout.addWidget(self.edit_box_checkbox, 3, 0, 1, 4)

        bounds_group.setLayout(bounds_layout)
        layout.addWidget(bounds_group)
        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_render_tab(self):
        """Create the Rendering Options tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Rendering controls group
        render_group = QGroupBox("Rendering Options")
        render_layout = QVBoxLayout()
        
        # Colormap selection
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['RdYlBu_r', 'viridis', 'plasma', 'inferno', 'jet', 'rainbow'])
        self.colormap_combo.setCurrentText('RdYlBu_r')
        colormap_layout.addWidget(self.colormap_combo)
        render_layout.addLayout(colormap_layout)
        
        # Active scalars selection
        scalars_layout = QHBoxLayout()
        scalars_layout.addWidget(QLabel("Active Scalars:"))
        self.active_scalars_combo = QComboBox()
        self.active_scalars_combo.addItems(['Resistivity(log10)'])
        self.active_scalars_combo.setCurrentText('Resistivity(log10)')
        scalars_layout.addWidget(self.active_scalars_combo)
        render_layout.addLayout(scalars_layout)
        
        # Data range controls
        range_group_layout = QVBoxLayout()
        
        # Min value control
        min_range_layout = QHBoxLayout()
        min_range_layout.addWidget(QLabel("Data Min:"))
        self.data_min_spinbox = QDoubleSpinBox()
        self.data_min_spinbox.setRange(-10000, 10000)
        self.data_min_spinbox.setValue(0.0)
        self.data_min_spinbox.setDecimals(3)
        self.data_min_spinbox.setSingleStep(0.1)
        min_range_layout.addWidget(self.data_min_spinbox)
        
        # Auto-detect button for min
        self.auto_min_btn = QPushButton("Auto")
        self.auto_min_btn.setFixedSize(50, 25)
        min_range_layout.addWidget(self.auto_min_btn)
        
        range_group_layout.addLayout(min_range_layout)
        
        # Max value control
        max_range_layout = QHBoxLayout()
        max_range_layout.addWidget(QLabel("Data Max:"))
        self.data_max_spinbox = QDoubleSpinBox()
        self.data_max_spinbox.setRange(-10000, 10000)
        self.data_max_spinbox.setValue(1.0)
        self.data_max_spinbox.setDecimals(3)
        self.data_max_spinbox.setSingleStep(0.1)
        max_range_layout.addWidget(self.data_max_spinbox)
        
        # Auto-detect button for max
        self.auto_max_btn = QPushButton("Auto")
        self.auto_max_btn.setFixedSize(50, 25)
        max_range_layout.addWidget(self.auto_max_btn)
        
        range_group_layout.addLayout(max_range_layout)
        
        # Auto-detect both button
        auto_both_layout = QHBoxLayout()
        self.auto_range_btn = QPushButton("Auto-Detect Range")
        self.auto_range_btn.setFixedSize(140, 25)
        auto_both_layout.addWidget(self.auto_range_btn)
        auto_both_layout.addStretch()
        range_group_layout.addLayout(auto_both_layout)
        
        render_layout.addLayout(range_group_layout)
        
        # Resampling controls group
        resample_group_layout = QVBoxLayout()
        
        # Enable resampling checkbox
        self.enable_resampling_checkbox = QCheckBox("Enable Resampling (for unstructured meshes)")
        self.enable_resampling_checkbox.setChecked(True)
        self.enable_resampling_checkbox.setToolTip("Resample unstructured mesh to uniform grid for volume rendering. Disable to use unstructured grid directly.")
        resample_group_layout.addWidget(self.enable_resampling_checkbox)
        
        # Target cells for resampling (only enabled when resampling is on)
        target_cells_layout = QHBoxLayout()
        target_cells_layout.addWidget(QLabel("Target Cells:"))
        self.target_cells_spinbox = QSpinBox()
        self.target_cells_spinbox.setRange(10000, 2000000)
        self.target_cells_spinbox.setValue(500000)
        self.target_cells_spinbox.setSuffix(" cells")
        self.target_cells_spinbox.setSingleStep(50000)
        target_cells_layout.addWidget(self.target_cells_spinbox)
        resample_group_layout.addLayout(target_cells_layout)
        
        render_layout.addLayout(resample_group_layout)
        
        # Lighting quality selection
        lighting_layout = QHBoxLayout()
        lighting_layout.addWidget(QLabel("Lighting:"))
        self.lighting_combo = QComboBox()
        self.lighting_combo.addItems(['Enhanced', 'Standard', 'Minimal'])
        self.lighting_combo.setCurrentText('Enhanced')
        lighting_layout.addWidget(self.lighting_combo)
        render_layout.addLayout(lighting_layout)

        # Volume shading control
        self.shading_checkbox = QCheckBox("Volume Shading")
        self.shading_checkbox.setChecked(True)
        self.shading_checkbox.setToolTip("Light the volume using data gradients. Gives depth cues, "
                                         "but also shows the cells of the inversion mesh as facets.")
        render_layout.addWidget(self.shading_checkbox)

        # Volume rendering control
        self.show_volume_checkbox = QCheckBox("Show Volume Rendering")
        self.show_volume_checkbox.setChecked(True)
        render_layout.addWidget(self.show_volume_checkbox)
        
        # Auto-hide volume when isosurfaces are opaque
        self.auto_hide_volume_checkbox = QCheckBox("Auto-hide volume with opaque surfaces")
        self.auto_hide_volume_checkbox.setChecked(True)
        self.auto_hide_volume_checkbox.setToolTip("Automatically disable volume when isosurface opacity ≥ 0.9")
        render_layout.addWidget(self.auto_hide_volume_checkbox)
        
        # Isosurface controls
        isosurface_group = QGroupBox("Isosurface Controls")
        isosurface_layout = QVBoxLayout()
        
        # Enable isosurfaces checkbox
        self.show_isosurfaces_checkbox = QCheckBox("Show Isosurfaces")
        self.show_isosurfaces_checkbox.setChecked(False)
        isosurface_layout.addWidget(self.show_isosurfaces_checkbox)
        
        # Isosurface mode selection
        iso_mode_layout = QHBoxLayout()
        iso_mode_layout.addWidget(QLabel("Mode:"))
        self.iso_single_radio = QRadioButton("Single Value")
        self.iso_multiple_radio = QRadioButton("Multiple Surfaces")
        self.iso_single_radio.setChecked(True)
        iso_mode_layout.addWidget(self.iso_single_radio)
        iso_mode_layout.addWidget(self.iso_multiple_radio)
        isosurface_layout.addLayout(iso_mode_layout)
        
        # Single isosurface value control
        iso_value_layout = QHBoxLayout()
        iso_value_layout.addWidget(QLabel("Iso Value:"))
        self.iso_value_spinbox = QDoubleSpinBox()
        self.iso_value_spinbox.setRange(-10000, 10000)
        self.iso_value_spinbox.setValue(2.0)
        self.iso_value_spinbox.setDecimals(3)
        self.iso_value_spinbox.setSingleStep(0.1)
        iso_value_layout.addWidget(self.iso_value_spinbox)
        isosurface_layout.addLayout(iso_value_layout)
        
        # Multiple isosurfaces number control
        iso_num_layout = QHBoxLayout()
        iso_num_layout.addWidget(QLabel("Num Surfaces:"))
        self.iso_num_spinbox = QSpinBox()
        self.iso_num_spinbox.setRange(2, 20)
        self.iso_num_spinbox.setValue(5)
        self.iso_num_spinbox.setEnabled(False)
        iso_num_layout.addWidget(self.iso_num_spinbox)
        isosurface_layout.addLayout(iso_num_layout)
        
        # Isosurface opacity control
        iso_opacity_layout = QHBoxLayout()
        iso_opacity_layout.addWidget(QLabel("Iso Opacity:"))
        self.iso_opacity_spinbox = QDoubleSpinBox()
        self.iso_opacity_spinbox.setRange(0.0, 1.0)
        self.iso_opacity_spinbox.setValue(0.8)
        self.iso_opacity_spinbox.setDecimals(2)
        self.iso_opacity_spinbox.setSingleStep(0.1)
        iso_opacity_layout.addWidget(self.iso_opacity_spinbox)
        isosurface_layout.addLayout(iso_opacity_layout)
        
        isosurface_group.setLayout(isosurface_layout)
        render_layout.addWidget(isosurface_group)
        
        # Show bounds checkbox
        self.show_bounds_checkbox = QCheckBox("Show Bounds")
        self.show_bounds_checkbox.setChecked(True)
        render_layout.addWidget(self.show_bounds_checkbox)
        
        # Show color bar checkbox
        self.show_colorbar_checkbox = QCheckBox("Show Color Bar")
        self.show_colorbar_checkbox.setChecked(True)
        render_layout.addWidget(self.show_colorbar_checkbox)
        
        # Show frame label checkbox
        self.show_frame_label_checkbox = QCheckBox("Show Frame Label")
        self.show_frame_label_checkbox.setChecked(True)
        render_layout.addWidget(self.show_frame_label_checkbox)
        
        render_group.setLayout(render_layout)
        layout.addWidget(render_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def connect_signals(self):
        """Connect widget signals"""
        # Instant display updates
        display_signals = [
            self.opacity_editor.changed,
            self.colormap_combo.currentTextChanged,
            self.data_min_spinbox.valueChanged,
            self.data_max_spinbox.valueChanged,
            self.lighting_combo.currentTextChanged,
            self.shading_checkbox.toggled,
            self.show_volume_checkbox.toggled,
            self.auto_hide_volume_checkbox.toggled,
            self.iso_opacity_spinbox.valueChanged,
            self.show_bounds_checkbox.toggled,
            self.show_colorbar_checkbox.toggled,
            self.show_frame_label_checkbox.toggled,
        ]
        for signal in display_signals:
            signal.connect(self.display_changed)
        
        # Isosurface geometry (multiple surfaces are spread over the data range)
        isosurface_signals = [
            self.show_isosurfaces_checkbox.toggled,
            self.iso_single_radio.toggled,
            self.iso_value_spinbox.valueChanged,
            self.iso_num_spinbox.valueChanged,
            self.data_min_spinbox.valueChanged,
            self.data_max_spinbox.valueChanged,
        ]
        for signal in isosurface_signals:
            signal.connect(self.isosurfaces_changed)
        
        # Data reloading
        data_signals = [spinbox.valueChanged for spinbox in self.bounds_spinboxes] + [
            self.active_scalars_combo.currentTextChanged,
            self.enable_resampling_checkbox.toggled,
            self.target_cells_spinbox.valueChanged,
        ]
        for signal in data_signals:
            signal.connect(self.data_changed)
        
        self.data_min_spinbox.valueChanged.connect(self.on_data_range_changed)
        self.data_max_spinbox.valueChanged.connect(self.on_data_range_changed)
        self.enable_resampling_checkbox.toggled.connect(self.on_resampling_toggled)
        self.iso_single_radio.toggled.connect(self.on_iso_mode_changed)
        
        # Slice planes
        for checkbox, slider in zip(self.slice_checkboxes, self.slice_sliders):
            checkbox.toggled.connect(self.slices_changed)
            slider.valueChanged.connect(self.slices_changed)

        # Connect data range auto-detect buttons
        self.auto_min_btn.clicked.connect(self.auto_detect_min)
        self.auto_max_btn.clicked.connect(self.auto_detect_max)
        self.auto_range_btn.clicked.connect(self.auto_detect_range)
    
    def on_data_range_changed(self):
        """Handle data range spinbox changes - update min/max labels immediately"""
        # Get current values from spinboxes
        current_min = self.data_min_spinbox.value()
        current_max = self.data_max_spinbox.value()
        
        # Update the min/max labels above the sliders
        self.update_minmax_labels(current_min, current_max)
    
    def on_resampling_toggled(self):
        """Handle resampling checkbox toggle - enable/disable target cells control"""
        is_enabled = self.enable_resampling_checkbox.isChecked()
        self.target_cells_spinbox.setEnabled(is_enabled)
    
    def on_iso_mode_changed(self):
        """Handle isosurface mode change - enable/disable appropriate controls"""
        is_single_mode = self.iso_single_radio.isChecked()
        
        # Enable/disable controls based on mode
        self.iso_value_spinbox.setEnabled(is_single_mode)
        self.iso_num_spinbox.setEnabled(not is_single_mode)
    
    def on_opacity_hover(self, text):
        """Show the value under the cursor, or usage hints"""
        self.opacity_hint_label.setText(
            text or "Drag points to edit · click to add · right-click or Delete to remove")

    def apply_opacity_preset(self, name):
        self.opacity_editor.set_points(OPACITY_PRESETS[name])

    def reverse_opacity(self):
        """Swap the opacity of low and high values"""
        self.opacity_editor.set_points([(1.0 - x, y) for x, y in self.opacity_editor.points()])

    def set_opacity_points(self, points):
        """Set the opacity curve as (relative position, opacity) points"""
        self.opacity_editor.set_points(points)

    def set_bounds_values(self, bounds_list):
        """Set bounds spinbox values"""
        for i, value in enumerate(bounds_list):
            if i < len(self.bounds_spinboxes):
                self.bounds_spinboxes[i].setValue(value)
    
    def show_progress(self, show=True):
        """Show/hide progress bar"""
        self.progress_bar.setVisible(show)
    
    def show_busy(self, busy=True):
        """Show/hide an indeterminate progress bar"""
        self.progress_bar.setRange(0, 0 if busy else 100)
        self.progress_bar.setVisible(busy)
    
    def set_progress(self, value):
        """Set progress bar value"""
        self.progress_bar.setValue(value)
    
    def get_opacity_points(self):
        """Get the opacity curve as (relative position, opacity) points"""
        return self.opacity_editor.points()

    def get_slices(self):
        """Enabled slice planes as (axis index, relative position) pairs"""
        return [(axis, slider.value() / slider.maximum())
                for axis, (checkbox, slider) in enumerate(zip(self.slice_checkboxes, self.slice_sliders))
                if checkbox.isChecked()]

    def get_bounds_values(self):
        """Get current bounds values from spinboxes"""
        return [spinbox.value() for spinbox in self.bounds_spinboxes]
    
    def get_colormap(self):
        """Get current colormap selection"""
        return self.colormap_combo.currentText()
    
    def get_target_cells(self):
        """Get current target cells value"""
        return self.target_cells_spinbox.value()
    
    def get_lighting_quality(self):
        """Get current lighting quality selection"""
        return self.lighting_combo.currentText()
    
    def get_active_scalars(self):
        """Get current active scalars selection"""
        return self.active_scalars_combo.currentText()
    
    def get_data_min(self):
        """Get current data minimum value"""
        return self.data_min_spinbox.value()
    
    def get_data_max(self):
        """Get current data maximum value"""
        return self.data_max_spinbox.value()
    
    def is_resampling_enabled(self):
        """Get resampling checkbox state"""
        return self.enable_resampling_checkbox.isChecked()
    
    def is_show_bounds_enabled(self):
        """Get show bounds checkbox state"""
        return self.show_bounds_checkbox.isChecked()
    
    def is_show_colorbar_enabled(self):
        """Get show color bar checkbox state"""
        return self.show_colorbar_checkbox.isChecked()
    
    def is_shading_enabled(self):
        """Get volume shading checkbox state"""
        return self.shading_checkbox.isChecked()

    def is_show_frame_label_enabled(self):
        """Get show frame label checkbox state"""
        return self.show_frame_label_checkbox.isChecked()
    
    def is_show_volume_enabled(self):
        """Get show volume rendering checkbox state"""
        return self.show_volume_checkbox.isChecked()
    
    def is_auto_hide_volume_enabled(self):
        """Get auto-hide volume checkbox state"""
        return self.auto_hide_volume_checkbox.isChecked()
    
    def is_show_isosurfaces_enabled(self):
        """Get show isosurfaces checkbox state"""
        return self.show_isosurfaces_checkbox.isChecked()
    
    def is_iso_single_mode(self):
        """Get isosurface mode - True for single value, False for multiple"""
        return self.iso_single_radio.isChecked()
    
    def get_iso_value(self):
        """Get current isosurface value"""
        return self.iso_value_spinbox.value()
    
    def get_iso_num_surfaces(self):
        """Get number of isosurfaces for multiple mode"""
        return self.iso_num_spinbox.value()
    
    def get_iso_opacity(self):
        """Get current isosurface opacity"""
        return self.iso_opacity_spinbox.value()
    
    def update_minmax_labels(self, global_min, global_max):
        """Update the min/max labels with actual data range values"""
        self.left_label.setText(f"{global_min:.3f}")
        self.right_label.setText(f"{global_max:.3f}")
    
    def update_available_scalars(self, scalar_names, default_scalar='Resistivity(log10)'):
        """Update the available scalars in the combo box"""
        current_selection = self.active_scalars_combo.currentText()
        
        # Clear and repopulate the combo box
        self.active_scalars_combo.clear()
        self.active_scalars_combo.addItems(scalar_names)
        
        # Try to restore previous selection, otherwise use default
        # (cell arrays are listed with a " (cell)" suffix, so accept either form)
        default_cell_scalar = f"{default_scalar} (cell)"
        if current_selection in scalar_names:
            self.active_scalars_combo.setCurrentText(current_selection)
        elif default_scalar in scalar_names:
            self.active_scalars_combo.setCurrentText(default_scalar)
        elif default_cell_scalar in scalar_names:
            self.active_scalars_combo.setCurrentText(default_cell_scalar)
        else:
            # If neither available, select the first item
            if scalar_names:
                self.active_scalars_combo.setCurrentIndex(0)
    
    def set_data_range(self, data_min, data_max):
        """Set the data range spinbox values without emitting change signals"""
        for spinbox, value in ((self.data_min_spinbox, data_min), (self.data_max_spinbox, data_max)):
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)
        self.update_minmax_labels(data_min, data_max)
    
    def auto_detect_min(self):
        """Auto-detect minimum value from current data"""
        if self.main_app:
            self.main_app.auto_detect_scalar_min()
    
    def auto_detect_max(self):
        """Auto-detect maximum value from current data"""
        if self.main_app:
            self.main_app.auto_detect_scalar_max()
    
    def auto_detect_range(self):
        """Auto-detect both min and max values from current data"""
        if self.main_app:
            self.main_app.auto_detect_scalar_range()


class _LogEmitter(QObject):
    record_emitted = Signal(int, str)


class QtLogHandler(logging.Handler):
    """Logging handler that forwards records to the GUI thread via a Qt signal"""

    def __init__(self):
        super().__init__()
        self.emitter = _LogEmitter()
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record):
        # PyVista forwards VTK errors to the root logger as warnings; show them as errors
        if record.name == "root" and isinstance(record.msg, str) and record.msg.startswith("ERROR:"):
            record = logging.makeLogRecord(dict(record.__dict__, name="vtk", msg=record.msg[len("ERROR:"):].strip(),
                                                levelno=logging.ERROR, levelname="ERROR"))
        try:
            self.emitter.record_emitted.emit(record.levelno, self.format(record))
        except RuntimeError:
            pass  # Panel already destroyed


class LogPanel(QWidget):
    """Log of application messages and VTK warnings/errors"""

    LEVELS = {'Info': logging.INFO, 'Warning': logging.WARNING, 'Error': logging.ERROR}
    COLORS = {logging.WARNING: '#d08000', logging.ERROR: '#d03030'}
    MAX_RECORDS = 5000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._records = deque(maxlen=self.MAX_RECORDS)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.addWidget(QLabel("Log"))
        header.addStretch()
        header.addWidget(QLabel("Show:"))
        self.level_combo = QComboBox()
        self.level_combo.addItems(list(self.LEVELS))
        header.addWidget(self.level_combo)
        self.clear_button = QPushButton("Clear")
        header.addWidget(self.clear_button)
        layout.addLayout(header)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(self.MAX_RECORDS)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        layout.addWidget(self.text)
        self.setLayout(layout)

        self.handler = QtLogHandler()
        self.handler.emitter.record_emitted.connect(self.add_record)
        self.level_combo.currentTextChanged.connect(self.refresh)
        self.clear_button.clicked.connect(self.clear)

    def min_level(self):
        return self.LEVELS[self.level_combo.currentText()]

    def add_record(self, levelno, text):
        self._records.append((levelno, text))
        if levelno >= self.min_level():
            self._append(levelno, text)

    def _append(self, levelno, text):
        scrollbar = self.text.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 2

        escaped = html.escape(text).replace("\n", "<br>")
        if levelno >= logging.ERROR:
            color = self.COLORS[logging.ERROR]
        elif levelno >= logging.WARNING:
            color = self.COLORS[logging.WARNING]
        else:
            color = None
        if color:
            escaped = f'<span style="color:{color}">{escaped}</span>'
        self.text.appendHtml(f'<div style="white-space:pre">{escaped}</div>')

        # Follow new messages unless the user scrolled up to read older ones
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def refresh(self):
        """Redisplay the stored records with the selected level filter"""
        self.text.clear()
        for levelno, text in self._records:
            if levelno >= self.min_level():
                self._append(levelno, text)

    def clear(self):
        self._records.clear()
        self.text.clear()


class PlaybackBar(QWidget):
    """Timeline below the 3D view: frame slider, step and playback controls.
    
    The slider runs over positions in the list of frames, so gaps in the
    frame numbering are skipped.
    """
    
    frame_changed = Signal(int)  # Frame index (as in the file names)
    play_toggled = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames = []
        
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 2, 4, 2)
        
        style = self.style()
        def tool_button(icon, tooltip):
            button = QToolButton()
            button.setIcon(style.standardIcon(icon))
            button.setToolTip(tooltip)
            button.setAutoRaise(True)
            layout.addWidget(button)
            return button
        
        self.first_button = tool_button(QStyle.StandardPixmap.SP_MediaSkipBackward, "First frame (Home)")
        self.prev_button = tool_button(QStyle.StandardPixmap.SP_MediaSeekBackward, "Previous frame (Left)")
        self.play_button = tool_button(QStyle.StandardPixmap.SP_MediaPlay, "Play/Pause (Space)")
        self.next_button = tool_button(QStyle.StandardPixmap.SP_MediaSeekForward, "Next frame (Right)")
        self.last_button = tool_button(QStyle.StandardPixmap.SP_MediaSkipForward, "Last frame (End)")
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setPageStep(1)
        layout.addWidget(self.slider, stretch=1)
        
        self.label = QLabel("No data")
        self.label.setMinimumWidth(140)
        layout.addWidget(self.label)
        
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 30)
        self.fps_spinbox.setValue(5)
        self.fps_spinbox.setSuffix(" fps")
        self.fps_spinbox.setToolTip("Playback speed (also used for video export). "
                                    "Playback waits for frames that are still loading.")
        layout.addWidget(self.fps_spinbox)
        
        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setChecked(True)
        layout.addWidget(self.loop_checkbox)
        
        self.setLayout(layout)
        
        self.first_button.clicked.connect(lambda: self.go_to_position(0))
        self.prev_button.clicked.connect(lambda: self.step(-1, wrap=False))
        self.next_button.clicked.connect(lambda: self.step(1, wrap=False))
        self.last_button.clicked.connect(lambda: self.go_to_position(len(self._frames) - 1))
        self.play_button.clicked.connect(self.play_toggled)
        self.slider.valueChanged.connect(self.on_slider_changed)
    
    def set_frames(self, frames):
        """Set the available frame indices and select the first one (without signalling)"""
        self._frames = list(frames)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self._frames) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.update_label()
    
    def current_frame(self):
        """Frame index currently selected, or None without data"""
        if not self._frames:
            return None
        return self._frames[self.slider.value()]
    
    def frames_after_current(self, count):
        """Up to count frame indices following the current one (wrapping when looping)"""
        if not self._frames:
            return []
        position = self.slider.value()
        frames = []
        for offset in range(1, min(count, len(self._frames) - 1) + 1):
            next_position = position + offset
            if next_position >= len(self._frames):
                if not self.loop_checkbox.isChecked():
                    break
                next_position %= len(self._frames)
            frames.append(self._frames[next_position])
        return frames
    
    def go_to_position(self, position):
        if self._frames:
            self.slider.setValue(max(0, min(position, len(self._frames) - 1)))
    
    def step(self, delta, wrap=None):
        """Move delta frames. Returns False at the end when not wrapping.
        
        wrap defaults to the Loop checkbox.
        """
        if not self._frames:
            return False
        if wrap is None:
            wrap = self.loop_checkbox.isChecked()
        position = self.slider.value() + delta
        if not 0 <= position < len(self._frames):
            if not wrap:
                return False
            position %= len(self._frames)
        self.slider.setValue(position)
        return True
    
    def set_playing(self, playing):
        icon = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.play_button.setIcon(self.style().standardIcon(icon))
    
    def update_label(self):
        frame = self.current_frame()
        if frame is None:
            self.label.setText("No data")
        else:
            self.label.setText(f"Frame {frame}  ({self.slider.value() + 1}/{len(self._frames)})")
    
    def on_slider_changed(self):
        self.update_label()
        self.frame_changed.emit(self.current_frame())


class DamVisualizationApp(QMainWindow):
    """Main application window"""

    # Emitted (possibly from a worker thread) when a frame has been prepared
    frame_loaded = Signal(object, object)
    # Emitted (from a worker thread) when a probe value of a frame is available
    probe_value_loaded = Signal(int, int, object)

    def __init__(self):
        super().__init__()

        # Initialize data
        self.data_location = None
        self.vtk_files = {}

        # Visualization parameters
        self.opacity_points = list(DEFAULT_OPACITY_POINTS)
        self.bounds = [2, 17, 2, 22, 22, 27]
        self.colormap = 'RdYlBu_r'
        self.active_scalars = 'Resistivity(log10)'
        self.target_cells = 500000
        self.enable_resampling = True
        self.show_bounds = True
        self.show_colorbar = True
        self.show_volume = True
        self.auto_hide_volume = True
        self.show_isosurfaces = False
        self.iso_single_mode = True
        self.iso_value = 2.0
        self.iso_num_surfaces = 5
        self.iso_opacity = 0.8
        self.lighting_quality = 'Enhanced'
        self.global_min = 0.0  # Will be auto-detected from actual data
        self.global_max = 1.0  # Will be auto-detected from actual data

        # Frame loading state
        self._frame_cache = dvu.FrameCache(FRAME_CACHE_SIZE)
        self._frame_data = None        # FrameData currently shown
        self._displayed_key = None     # Cache key of the frame currently shown
        self._pending = {}             # Cache key -> future of frames being prepared by the worker
        self._auto_range_pending = False
        self._reset_camera_pending = False
        self._exporting = False

        # Frames are prepared in a separate process since VTK keeps the GIL
        # while filtering, which would freeze the UI if a thread was used.
        self._executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        self._executor.submit(dvu.warm_up)

        # Probe state; time series values are computed in their own worker process
        # so they don't delay frame loading
        self._probe_point = None
        self._probe_scalar = None
        self._probe_id = 0
        self._probe_futures = []
        self._probe_queue = []  # (frame index, FrameData) of loaded frames still to sample
        self._probe_executor = None
        self._press_position = None
        self._saved_camera = None

        self.setup_ui()
        self.setup_scene()
        self.setup_timers()
        self.connect_signals()

        # Try to load default data location
        default_path = "/home/bmjl/lu2023-17-17/Inversion_RealData/Results"
        if os.path.exists(default_path):
            log.info("Loading default data location %s", default_path)
            self.load_data_location(default_path)
        else:
            log.info("Use Open (Ctrl+O) to load a folder with VTK files")

    def setup_ui(self):
        """Set up the main UI"""
        self.setWindowTitle("Dam Resistivity Visualization")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()

        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # VTK visualization widget with the timeline below it
        view_container = QWidget()
        view_layout = QVBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)
        self.vtk_widget = VTKVisualizationWidget()
        self.playback = PlaybackBar()
        view_layout.addWidget(self.vtk_widget, stretch=1)
        view_layout.addWidget(self.playback)
        view_container.setLayout(view_layout)
        splitter.addWidget(view_container)

        # Control panel
        self.control_panel = ControlPanel(main_app=self)
        self.control_panel.setMaximumWidth(600)
        self.control_panel.setMinimumWidth(400)
        # Initialize min/max labels with global data range
        self.control_panel.update_minmax_labels(self.global_min, self.global_max)
        splitter.addWidget(self.control_panel)

        # Set splitter proportions
        splitter.setSizes([1000, 400])

        # Log across the full window width at the bottom, resizable
        self.log_panel = LogPanel()
        logging.getLogger().addHandler(self.log_panel.handler)
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(splitter)
        main_splitter.addWidget(self.log_panel)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setSizes([750, 150])

        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)

        # Menu bar
        self.create_menu_bar()

        # Status bar
        self.statusBar().showMessage("Ready")

    def setup_scene(self):
        """Create the persistent VTK pipeline.

        Actors and transfer functions are created once and updated in place,
        so changing a display setting only needs a re-render.
        """
        renderer = self.vtk_widget.renderer

        self.color_func = vtk.vtkColorTransferFunction()
        self.opacity_func = vtk.vtkPiecewiseFunction()

        # Volume property with enhanced lighting
        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.SetColor(self.color_func)
        self.volume_property.SetScalarOpacity(self.opacity_func)
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.SetAmbient(0.2)      # Ambient lighting (base illumination)
        self.volume_property.SetDiffuse(0.7)      # Diffuse lighting (directional light scattering)
        self.volume_property.SetSpecular(0.3)     # Specular lighting (shiny highlights)
        self.volume_property.SetSpecularPower(20) # Specular power (shininess concentration)
        self.volume_property.SetScalarOpacityUnitDistance(0.5)  # Controls opacity density

        # GPU mapper for resampled image data. Voxels outside the mesh are hidden
        # with a binary mask, and jittered ray start positions break up the
        # regular sampling pattern that otherwise shows up as moire stripes.
        self.image_mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.image_mapper.SetBlendModeToComposite()
        self.image_mapper.SetMaskTypeToBinary()
        self.image_mapper.SetUseJittering(True)

        # Mapper for the tetrahedral mesh when resampling is disabled
        self.mesh_mapper = vtk.vtkProjectedTetrahedraMapper()

        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetProperty(self.volume_property)
        self.volume_actor.SetVisibility(False)
        self._volume_added = False  # Added to the renderer once it has input data

        self.iso_actor = None

        self.bounds_actor = self.create_bounds_actor()
        self.bounds_actor.SetVisibility(False)
        renderer.AddActor(self.bounds_actor)

        self.scalar_bar_actor = self.vtk_widget.create_scalar_bar(self.color_func)
        self.scalar_bar_actor.SetVisibility(False)

        # Frame label in the upper left corner (also visible in exported videos)
        self.frame_text_actor = vtk.vtkTextActor()
        text_prop = self.frame_text_actor.GetTextProperty()
        text_prop.SetFontSize(14)
        text_prop.SetColor(1, 1, 1)
        text_prop.SetFontFamilyToArial()
        text_prop.ShadowOn()
        text_prop.SetVerticalJustificationToTop()
        self.frame_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.frame_text_actor.SetPosition(0.02, 0.97)
        self.frame_text_actor.SetVisibility(False)
        renderer.AddActor2D(self.frame_text_actor)

        # Slice planes (recreated on change) and probe marker
        self.slice_actors = []
        self.probe_source = vtk.vtkSphereSource()
        self.probe_source.SetThetaResolution(16)
        self.probe_source.SetPhiResolution(16)
        probe_mapper = vtk.vtkPolyDataMapper()
        probe_mapper.SetInputConnection(self.probe_source.GetOutputPort())
        self.probe_actor = vtk.vtkActor()
        self.probe_actor.SetMapper(probe_mapper)
        self.probe_actor.GetProperty().SetColor(1.0, 0.0, 1.0)
        self.probe_actor.SetVisibility(False)
        self.probe_actor.PickableOff()
        renderer.AddActor(self.probe_actor)

        # Box widget for editing the clipping bounds in the 3D view
        self.box_representation = vtk.vtkBoxRepresentation()
        self.box_representation.SetPlaceFactor(1.0)
        self.box_representation.GetOutlineProperty().SetColor(1.0, 1.0, 1.0)
        self.box_widget = vtk.vtkBoxWidget2()
        self.box_widget.SetInteractor(self.vtk_widget.interactor)
        self.box_widget.SetRepresentation(self.box_representation)
        self.box_widget.RotationEnabledOff()
        self.box_widget.AddObserver("EndInteractionEvent", self.on_box_edited)
        self._box_bounds = None

        # Plain clicks (without dragging) in the 3D view probe the data. A Qt event
        # filter is used since the VTK interactor style grabs the mouse events.
        self.vtk_widget.vtk_widget.installEventFilter(self)

    def setup_timers(self):
        """Create timers that coalesce rapid control changes into one update"""
        self._display_timer = QTimer(self)
        self._display_timer.setSingleShot(True)
        self._display_timer.setInterval(0)
        self._display_timer.timeout.connect(self.update_display)

        self._isosurface_timer = QTimer(self)
        self._isosurface_timer.setSingleShot(True)
        self._isosurface_timer.setInterval(ISOSURFACE_DEBOUNCE_MS)
        self._isosurface_timer.timeout.connect(self.update_isosurfaces)

        self._data_timer = QTimer(self)
        self._data_timer.setSingleShot(True)
        self._data_timer.setInterval(DATA_DEBOUNCE_MS)
        self._data_timer.timeout.connect(self.request_frame)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self.on_play_tick)

        self._slice_timer = QTimer(self)
        self._slice_timer.setSingleShot(True)
        self._slice_timer.setInterval(SLICE_DEBOUNCE_MS)
        self._slice_timer.timeout.connect(self.update_slices)

    def create_action(self, text, handler=None, shortcut=None, tooltip=None, checkable=False, checked=False,
                      icon=None):
        """Create an action shared by the menus and the toolbar"""
        action = QAction(text, self)
        if icon:
            # Checked buttons get a light accent background (see style_toolbar), so use the accent color
            palette = self.palette()
            action.setIcon(load_icon(icon, palette.color(QPalette.ColorRole.ButtonText),
                                     palette.color(QPalette.ColorRole.Highlight)))
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if tooltip:
            action.setToolTip(f"{tooltip} ({shortcut})" if shortcut else tooltip)
        if checkable:
            action.setCheckable(True)
            action.setChecked(checked)
            if handler:
                action.toggled.connect(handler)
        elif handler:
            action.triggered.connect(handler)
        return action

    def style_toolbar(self):
        """Windows 11 look for the menu bar and toolbar.

        Roomy buttons with rounded hover and pressed backgrounds and a light
        accent tint for checked buttons. Menu bar items and toolbar buttons
        use the same outer margin so they line up. Colors are derived from
        the palette so light and dark themes both work.
        """
        palette = self.palette()
        text = palette.color(QPalette.ColorRole.ButtonText)
        accent = palette.color(QPalette.ColorRole.Highlight)

        def rgba(color, alpha):
            return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha})"

        self.menuBar().setStyleSheet(f"""
            QMenuBar {{ padding: 2px 6px; }}
            QMenuBar::item {{ padding: 5px 10px; border-radius: 4px; background: transparent; }}
            QMenuBar::item:selected {{ background: {rgba(text, 0.07)}; }}
            QMenuBar::item:pressed {{ background: {rgba(text, 0.11)}; }}
        """)
        self.toolbar.setStyleSheet(f"""
            QToolBar {{ padding: 2px 6px 6px 6px; spacing: 2px; border: none; }}
            QToolBar::separator {{ width: 1px; margin: 8px 6px; background: {rgba(text, 0.15)}; }}
            QToolButton {{ padding: 7px; border: none; border-radius: 4px; background: transparent; }}
            QToolButton:hover {{ background: {rgba(text, 0.07)}; }}
            QToolButton:pressed {{ background: {rgba(text, 0.11)}; }}
            QToolButton:checked {{ background: {rgba(accent, 0.15)}; }}
            QToolButton:checked:hover {{ background: {rgba(accent, 0.22)}; }}
        """)

    def create_checkbox_action(self, text, checkbox, tooltip, icon=None):
        """Checkable action kept in sync with a control panel checkbox"""
        action = self.create_action(text, checkbox.setChecked, tooltip=tooltip,
                                    checkable=True, checked=checkbox.isChecked(), icon=icon)
        checkbox.toggled.connect(action.setChecked)
        return action

    def create_menu_bar(self):
        """Create the menu bar and toolbar"""
        cp = self.control_panel

        # Actions
        open_action = self.create_action('Open...', self.open_data_location, "Ctrl+O",
                                         "Open a folder with VTK files", icon="folder_open")
        screenshot_action = self.create_action('Screenshot...', self.save_screenshot, "Ctrl+P",
                                               "Save the 3D view as a PNG image", icon="camera")
        export_video_action = self.create_action('Export Video...', self.on_create_video,
                                                 tooltip="Render all frames to an MP4 video", icon="video_clip")
        exit_action = self.create_action('Exit', self.close)

        reset_camera_action = self.create_action('Reset Camera', self.vtk_widget.reset_camera,
                                                 tooltip="Fit the data in the view (R)", icon="zoom_fit")
        preset_actions = [self.create_action(name, lambda checked=False, name=name: self.set_camera_preset(name),
                                             f"Ctrl+{i}", f"{name} view of the data")
                          for i, name in enumerate(CAMERA_PRESETS, start=1)]
        save_view_action = self.create_action('Save View', self.save_camera, "Ctrl+Shift+S")
        restore_view_action = self.create_action('Restore View', self.restore_camera, "Ctrl+Shift+R")

        volume_action = self.create_checkbox_action('Volume', cp.show_volume_checkbox,
                                                    "Show the volume rendering", icon="cube")
        iso_action = self.create_checkbox_action('Isosurfaces', cp.show_isosurfaces_checkbox,
                                                 "Show isosurfaces", icon="layer")
        bounds_action = self.create_checkbox_action('Bounds', cp.show_bounds_checkbox,
                                                    "Show the axes grid", icon="border_all")
        colorbar_action = self.create_checkbox_action('Color Bar', cp.show_colorbar_checkbox,
                                                      "Show the color bar", icon="color")
        clip_box_action = self.create_checkbox_action('Clip Box', cp.edit_box_checkbox,
                                                      "Edit the clipping box in the 3D view", icon="crop")
        clear_probe_action = self.create_action('Clear Probe', self.clear_probe,
                                                tooltip="Remove the probe marker and time series",
                                                icon="location_dismiss")
        show_log_action = self.create_action('Log', self.log_panel.setVisible, "Ctrl+L",
                                             "Show the log", checkable=True, checked=True, icon="window_console")

        # File menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        file_menu.addAction(screenshot_action)
        file_menu.addAction(export_video_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('View')
        view_menu.addAction(reset_camera_action)
        for action in preset_actions:
            view_menu.addAction(action)
        view_menu.addAction(save_view_action)
        view_menu.addAction(restore_view_action)
        view_menu.addSeparator()
        for action in (volume_action, iso_action, bounds_action, colorbar_action, clip_box_action):
            view_menu.addAction(action)
        view_menu.addSeparator()
        view_menu.addAction(show_log_action)

        # Toolbar with the most used actions
        self.toolbar = self.addToolBar("Toolbar")
        self.toolbar.setObjectName("main_toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toolbar.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.style_toolbar()
        groups = [
            [open_action, screenshot_action, export_video_action],
            [reset_camera_action] + preset_actions,
            [volume_action, iso_action, bounds_action, colorbar_action],
            [clip_box_action, clear_probe_action, show_log_action],
        ]
        for i, group in enumerate(groups):
            if i:
                self.toolbar.addSeparator()
            for action in group:
                self.toolbar.addAction(action)
        # There are no clear icons for the view directions; show their names
        for action in preset_actions:
            self.toolbar.widgetForAction(action).setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        view_menu.addSeparator()
        view_menu.addAction(self.toolbar.toggleViewAction())

        # Keyboard shortcuts, active anywhere in the window (text fields keep their own keys)
        shortcuts = [
            ("Space", self.toggle_playback),
            ("Left", lambda: self.playback.step(-1, wrap=False)),
            ("Right", lambda: self.playback.step(1, wrap=False)),
            ("Home", self.playback.first_button.click),
            ("End", self.playback.last_button.click),
            ("R", self.vtk_widget.reset_camera),
        ]
        for key, handler in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(handler)

    def connect_signals(self):
        """Connect signals"""
        self.control_panel.display_changed.connect(self._display_timer.start)
        self.control_panel.isosurfaces_changed.connect(self._isosurface_timer.start)
        self.control_panel.data_changed.connect(self._data_timer.start)
        self.control_panel.active_scalars_combo.currentTextChanged.connect(self.on_active_scalars_changed)

        self.playback.frame_changed.connect(self.request_frame)
        self.playback.play_toggled.connect(self.toggle_playback)
        self.playback.fps_spinbox.valueChanged.connect(self.update_play_interval)

        self.frame_loaded.connect(self.on_frame_loaded)

        self.control_panel.slices_changed.connect(self._slice_timer.start)
        self.control_panel.edit_box_checkbox.toggled.connect(self._display_timer.start)
        self.control_panel.probe_panel.frame_clicked.connect(self.go_to_frame)
        self.control_panel.probe_panel.clear_requested.connect(self.clear_probe)
        self.probe_value_loaded.connect(self.on_probe_value_loaded)

    def closeEvent(self, event):
        """Stop the worker process when the window closes"""
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._probe_executor is not None:
            self._probe_executor.shutdown(wait=False, cancel_futures=True)
        logging.getLogger().removeHandler(self.log_panel.handler)
        super().closeEvent(event)

    def open_data_location(self):
        """Open dialog to select data location"""
        folder = QFileDialog.getExistingDirectory(self, "Select Data Location")
        if folder:
            self.load_data_location(folder)

    def load_data_location(self, folder_path):
        """Load VTK files from data location"""
        try:
            # Find VTK files
            vtk_files = {}
            for file in os.listdir(folder_path):
                if file.startswith("dcinv") and file.endswith(".vtk"):
                    number = int(file.split('_')[-1].split('.')[0])
                    vtk_files[number] = file

            if not vtk_files:
                log.warning("No VTK files found in %s", folder_path)
                QMessageBox.warning(self, "Warning", "No VTK files found in selected directory")
                return

            self.data_location = folder_path
            self.vtk_files = {key: vtk_files[key] for key in sorted(vtk_files)}
            log.info("Found %d VTK files in %s", len(self.vtk_files), folder_path)
            self._frame_cache.clear()
            self._displayed_key = None
            self.clear_probe()

            # Update controls
            self.set_playing(False)
            self.playback.set_frames(self.vtk_files.keys())
            self.control_panel.set_opacity_points(self.opacity_points)
            self.control_panel.set_bounds_values(self.bounds)

            # Detect available scalars from the first VTK file
            self.detect_available_scalars()

            # Show the first frame, fitting the camera and data range to it
            self._auto_range_pending = True
            self._reset_camera_pending = True
            self.request_frame()

            self.statusBar().showMessage(f"Found {len(self.vtk_files)} VTK files in {folder_path}, loading first frame...")

        except Exception as e:
            log.exception("Failed to load data from %s", folder_path)
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def detect_available_scalars(self):
        """Detect available scalar fields from the first VTK file"""
        try:
            if not self.vtk_files:
                return

            # Load the first file to inspect available scalars
            first_file_idx = min(self.vtk_files.keys())
            file_path = os.path.join(self.data_location, self.vtk_files[first_file_idx])
            mesh = pv.read(file_path)

            # Get all available scalar arrays (both point and cell data)
            available_scalars = []

            # Point data arrays
            for array_name in mesh.point_data.keys():
                available_scalars.append(array_name)

            # Cell data arrays
            for array_name in mesh.cell_data.keys():
                available_scalars.append(f"{array_name} (cell)")

            # Remove duplicates and sort
            available_scalars = sorted(list(set(available_scalars)))

            # Update the control panel combo box
            if available_scalars:
                self.control_panel.update_available_scalars(available_scalars, self.active_scalars)
                # Update the app's active_scalars to match what the combo box actually selected
                self.active_scalars = self.control_panel.get_active_scalars()
                log.info("Available scalars: %s; using %s", ", ".join(available_scalars), self.active_scalars)
            else:
                log.warning("No scalar arrays found in %s", file_path)

        except Exception as e:
            log.error("Error detecting scalars: %s", e)
            # Use default if detection fails
            self.control_panel.update_available_scalars(['Resistivity(log10)'], self.active_scalars)
            # Update the app's active_scalars to match what the combo box actually selected
            self.active_scalars = self.control_panel.get_active_scalars()

    def on_active_scalars_changed(self):
        """Fit the data range to a newly selected scalar once it is loaded"""
        self._auto_range_pending = True

    def read_parameters(self):
        """Copy the current settings from the control panel"""
        self.opacity_points = self.control_panel.get_opacity_points()
        self.bounds = self.control_panel.get_bounds_values()
        self.colormap = self.control_panel.get_colormap()
        self.active_scalars = self.control_panel.get_active_scalars()
        self.target_cells = self.control_panel.get_target_cells()
        self.enable_resampling = self.control_panel.is_resampling_enabled()
        self.global_min = self.control_panel.get_data_min()
        self.global_max = self.control_panel.get_data_max()
        self.show_bounds = self.control_panel.is_show_bounds_enabled()
        self.show_colorbar = self.control_panel.is_show_colorbar_enabled()
        self.show_volume = self.control_panel.is_show_volume_enabled()
        self.auto_hide_volume = self.control_panel.is_auto_hide_volume_enabled()
        self.show_isosurfaces = self.control_panel.is_show_isosurfaces_enabled()
        self.iso_single_mode = self.control_panel.is_iso_single_mode()
        self.iso_value = self.control_panel.get_iso_value()
        self.iso_num_surfaces = self.control_panel.get_iso_num_surfaces()
        self.iso_opacity = self.control_panel.get_iso_opacity()
        self.lighting_quality = self.control_panel.get_lighting_quality()

    # ------------------------------------------------------------------
    # Frame loading (expensive, runs in the worker process)
    # ------------------------------------------------------------------

    def frame_key(self, frame_index):
        """Cache key describing everything the prepared data of a frame depends on"""
        cp = self.control_panel
        return (self.data_location, frame_index, cp.get_active_scalars(),
                tuple(cp.get_bounds_values()), cp.is_resampling_enabled(), cp.get_target_cells())

    def submit_frame(self, key):
        """Start preparing a frame in the worker process and return the future"""
        data_location, frame_index, scalar, bounds, resample, target_cells = key
        file_path = os.path.join(data_location, self.vtk_files[frame_index])
        return self._executor.submit(dvu.prepare_frame, file_path, scalar.replace(" (cell)", ""),
                                     bounds, resample, target_cells)

    def request_frame(self):
        """Show the frame selected in the timeline, loading it if needed"""
        if self._exporting:
            return

        frame_index = self.playback.current_frame()
        if frame_index not in self.vtk_files:
            return

        key = self.frame_key(frame_index)
        if key == self._displayed_key:
            return

        frame = self._frame_cache.get(key)
        if frame is not None:
            self.show_frame(key, frame)
            return

        self.control_panel.show_busy(True)
        self.statusBar().showMessage(f"Loading frame {frame_index}...")

        if key in self._pending:
            return  # Shown by on_frame_loaded when ready

        # Drop queued prefetches so the selected frame is prepared next
        for pending_key, future in list(self._pending.items()):
            # A successful cancel runs on_frame_loaded right away, which may already remove the key
            if future.cancel():
                self._pending.pop(pending_key, None)

        # Only one frame is prepared at a time. Intermediate requests (e.g. while
        # dragging the frame slider) are skipped; on_frame_loaded requests the
        # latest frame again when the running one is done.
        if self._pending:
            return

        self.submit_pending(key)

    def submit_pending(self, key):
        """Prepare a frame in the worker process and handle it in on_frame_loaded"""
        future = self.submit_frame(key)
        future.submitted_at = time.perf_counter()
        self._pending[key] = future
        future.add_done_callback(lambda f, key=key: self._emit_frame_loaded(key, f))

    def prefetch_frames(self):
        """Prepare the frames following the displayed one in the background"""
        if self._exporting:
            return
        for frame_index in self.playback.frames_after_current(PREFETCH_FRAMES):
            key = self.frame_key(frame_index)
            if key not in self._pending and self._frame_cache.get(key) is None:
                self.submit_pending(key)

    def _emit_frame_loaded(self, key, future):
        """Forward a finished future to the GUI thread"""
        try:
            self.frame_loaded.emit(key, future)
        except RuntimeError:
            pass  # Window already destroyed

    def on_frame_loaded(self, key, future):
        """Handle a frame prepared by the worker process"""
        if self._pending.get(key) is future:
            del self._pending[key]

        if future.cancelled():
            return

        try:
            frame = dvu.FrameData(future.result())
        except Exception as e:
            log.error("Failed to load frame %s: %s", key[1], e)
            if self.frame_key(self.playback.current_frame()) == key:
                # Don't retry the selected frame and don't wait for it while playing
                self.control_panel.show_busy(False)
                self.set_playing(False)
                self.statusBar().showMessage(f"Failed to load frame {key[1]}: {e}")
            else:
                self.request_frame()
            return

        self._frame_cache.put(key, frame)
        log.info("Prepared frame %s (%s) in %.1f s", key[1], frame.scalar_name,
                 time.perf_counter() - future.submitted_at)

        # Show it if it is the selected frame, otherwise continue with the selected one
        self.request_frame()

    def show_frame(self, key, frame):
        """Display prepared frame data"""
        self._frame_data = frame
        self._displayed_key = key
        self.control_panel.show_busy(False)

        if self._auto_range_pending:
            self._auto_range_pending = False
            self.apply_auto_range(initial_iso_value=True)

        # Pick the mapper matching the data type
        if isinstance(frame.volume, pv.ImageData):
            mapper = self.image_mapper
            mapper.SetMaskInput(frame.mask)
            # The default sample distance (1.0) skips several voxels per step,
            # which causes banding; sample twice per voxel instead.
            mapper.SetSampleDistance(min(frame.volume.spacing) / 2)
        else:
            mapper = self.mesh_mapper
        mapper.SetInputData(frame.volume)
        self.volume_actor.SetMapper(mapper)
        if not self._volume_added:
            self.vtk_widget.renderer.AddVolume(self.volume_actor)
            self._volume_added = True

        frame_index = key[1]
        frame_number = list(self.vtk_files).index(frame_index) + 1
        self.frame_text_actor.SetInput(f"Frame {frame_index}  ({frame_number}/{len(self.vtk_files)})")

        self.update_slices()
        self.update_isosurfaces()

        if self._reset_camera_pending:
            self._reset_camera_pending = False
            self.vtk_widget.reset_camera()

        # Keep the probe in sync with the frame and scalar
        self.control_panel.probe_panel.set_current_frame(frame_index)
        if self._probe_point is not None and self._probe_scalar != frame.scalar_name:
            self.start_probe(self._probe_point)

        self.statusBar().showMessage(f"Displaying frame {frame_index}: {self.vtk_files[frame_index]}")
        self.prefetch_frames()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def toggle_playback(self):
        self.set_playing(not self._play_timer.isActive())

    def set_playing(self, playing):
        if playing and self.vtk_files:
            self.update_play_interval()
            self._play_timer.start()
        else:
            self._play_timer.stop()
        self.playback.set_playing(self._play_timer.isActive())

    def update_play_interval(self):
        self._play_timer.setInterval(int(1000 / self.playback.fps_spinbox.value()))

    def on_play_tick(self):
        """Advance to the next frame once the current one is displayed"""
        if self._displayed_key != self.frame_key(self.playback.current_frame()):
            return  # Still loading, playback waits for it
        if not self.playback.step(1):
            self.set_playing(False)  # Reached the end without looping

    # ------------------------------------------------------------------
    # Display updates (cheap, run directly in the GUI thread)
    # ------------------------------------------------------------------

    def update_isosurfaces(self):
        """Recreate the isosurface geometry for the current frame"""
        self.read_parameters()

        if self.iso_actor is not None:
            self.vtk_widget.renderer.RemoveActor(self.iso_actor)
            self.iso_actor = None

        if self.show_isosurfaces and self._frame_data is not None:
            self.iso_actor = self.create_isosurface_actor(self._frame_data)
            if self.iso_actor is not None:
                self.vtk_widget.renderer.AddActor(self.iso_actor)

        self.update_display()

    def update_display(self):
        """Apply display settings to the existing actors and re-render"""
        self.read_parameters()
        has_data = self._frame_data is not None

        self.vtk_widget.setup_enhanced_lighting(self.lighting_quality)
        self.volume_property.SetShade(self.control_panel.is_shading_enabled())
        build_color_function(self.color_func, self.colormap, self.global_min, self.global_max)

        editor = self.control_panel.opacity_editor
        editor.set_colormap(self.colormap)
        editor.set_histogram(self._frame_data.histogram_values() if has_data else None,
                             self.global_min, self.global_max)

        # Hide the volume when opaque isosurfaces would cover it anyway
        volume_visible = self.show_volume and not (
            self.auto_hide_volume and self.iso_actor is not None and self.iso_opacity >= 0.9)
        self.volume_actor.SetVisibility(has_data and volume_visible)

        # Reduce volume opacity when isosurfaces are nearly opaque to reduce bleeding
        opacity_scale = 0.7 if self.iso_actor is not None and self.iso_opacity >= 0.8 else 1.0
        self.update_opacity_function(opacity_scale)

        if self.iso_actor is not None:
            iso_opacity = self.iso_opacity
            # For multiple surfaces, make them slightly more transparent to avoid visual clutter
            if not self.iso_single_mode:
                iso_opacity *= 0.7
            # If nearly opaque, make it fully opaque to properly occlude volume rendering
            if iso_opacity >= 0.9:
                iso_opacity = 1.0
            self.iso_actor.GetProperty().SetOpacity(iso_opacity)

        self.bounds_actor.SetBounds(self.bounds)
        self.bounds_actor.SetVisibility(has_data and self.show_bounds)

        self.scalar_bar_actor.SetTitle(self.active_scalars.replace(" (cell)", ""))
        self.scalar_bar_actor.SetVisibility(has_data and self.show_colorbar)
        self.frame_text_actor.SetVisibility(has_data and self.control_panel.is_show_frame_label_enabled())
        self.update_box_widget()

        self.vtk_widget.render()

    def update_opacity_function(self, scale=1.0):
        """Map the opacity slider values onto the data range"""
        self.opacity_func.RemoveAllPoints()

        data_range = self.global_max - self.global_min
        if data_range <= 0:
            return

        for position, opacity in self.opacity_points:
            self.opacity_func.AddPoint(self.global_min + position * data_range, opacity * scale)

    def isosurface_values(self):
        """Values at which isosurfaces are extracted"""
        if not self.iso_single_mode and self.global_max > self.global_min and self.iso_num_surfaces > 1:
            # Spread multiple isosurfaces evenly between min and max
            step = (self.global_max - self.global_min) / (self.iso_num_surfaces + 1)
            return [self.global_min + i * step for i in range(1, self.iso_num_surfaces + 1)]
        return [self.iso_value]

    def create_isosurface_actor(self, frame):
        """Create an actor with the isosurface(s) of the frame, or None if empty"""
        try:
            iso_values = self.isosurface_values()
            surface = frame.mesh.contour(isosurfaces=iso_values, scalars=frame.scalar_name)

            if surface.n_points == 0:
                message = f"No isosurface found at {', '.join(f'{v:.3f}' for v in iso_values)}"
                log.info(message)
                self.statusBar().showMessage(message)
                return None

            # Colored with the same (live) color function as the volume
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(surface)
            mapper.SetLookupTable(self.color_func)
            mapper.UseLookupTableScalarRangeOn()
            mapper.ScalarVisibilityOn()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetInterpolationToGouraud()  # Smooth shading
            actor.GetProperty().SetSpecular(0.6)  # Add some shininess
            actor.GetProperty().SetSpecularPower(30)

            # Ensure surfaces are visible from both sides
            actor.GetProperty().SetBackfaceCulling(False)
            actor.GetProperty().SetFrontfaceCulling(False)

            return actor

        except Exception as e:
            log.error("Error creating isosurface: %s", e)
            return None

    def create_bounds_actor(self):
        """Create ParaView-style axes grid for bounding box"""
        if not self.show_bounds:
            return None
        
        # Extract bounds values
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        
        # Create cube axes actor (ParaView-style grid)
        cube_axes = vtk.vtkCubeAxesActor()
        cube_axes.SetBounds(self.bounds)
        cube_axes.SetCamera(self.vtk_widget.renderer.GetActiveCamera())
        
        # Set axes properties
        cube_axes.GetTitleTextProperty(0).SetColor(1.0, 1.0, 1.0)  # X axis title
        cube_axes.GetTitleTextProperty(1).SetColor(1.0, 1.0, 1.0)  # Y axis title  
        cube_axes.GetTitleTextProperty(2).SetColor(1.0, 1.0, 1.0)  # Z axis title
        
        cube_axes.GetLabelTextProperty(0).SetColor(0.8, 0.8, 0.8)  # X axis labels
        cube_axes.GetLabelTextProperty(1).SetColor(0.8, 0.8, 0.8)  # Y axis labels
        cube_axes.GetLabelTextProperty(2).SetColor(0.8, 0.8, 0.8)  # Z axis labels
        
        # Set font sizes (smaller)
        for i in range(3):
            cube_axes.GetTitleTextProperty(i).SetFontSize(10)
            cube_axes.GetLabelTextProperty(i).SetFontSize(8)
            cube_axes.GetTitleTextProperty(i).SetFontFamilyToArial()
            cube_axes.GetLabelTextProperty(i).SetFontFamilyToArial()
        
        # Set axis titles
        cube_axes.SetXTitle("X")
        cube_axes.SetYTitle("Y") 
        cube_axes.SetZTitle("Z")
        
        # Configure tick marks and grid
        cube_axes.SetTickLocationToBoth()  # Ticks on both sides
        cube_axes.SetFlyModeToOuterEdges()  # Draw on outer edges
        
        # Set number of ticks/labels for each axis (fewer for cleaner look)
        cube_axes.SetXAxisTickVisibility(True)
        cube_axes.SetYAxisTickVisibility(True)
        cube_axes.SetZAxisTickVisibility(True)
        cube_axes.SetXAxisLabelVisibility(True)
        cube_axes.SetYAxisLabelVisibility(True)
        cube_axes.SetZAxisLabelVisibility(True)
        
        # Grid lines properties
        cube_axes.SetGridLineLocation(vtk.vtkCubeAxesActor.VTK_GRID_LINES_ALL)
        cube_axes.GetXAxesGridlinesProperty().SetColor(0.3, 0.3, 0.3)  # Dark gray
        cube_axes.GetYAxesGridlinesProperty().SetColor(0.3, 0.3, 0.3)
        cube_axes.GetZAxesGridlinesProperty().SetColor(0.3, 0.3, 0.3)
        
        # Main axes lines properties  
        cube_axes.GetXAxesLinesProperty().SetColor(0.8, 0.8, 0.8)  # Light gray
        cube_axes.GetYAxesLinesProperty().SetColor(0.8, 0.8, 0.8)
        cube_axes.GetZAxesLinesProperty().SetColor(0.8, 0.8, 0.8)
        
        # Enable/disable specific features
        cube_axes.SetDrawXGridlines(True)
        cube_axes.SetDrawYGridlines(True) 
        cube_axes.SetDrawZGridlines(True)
        
        return cube_axes
    
    # ------------------------------------------------------------------
    # Slice planes
    # ------------------------------------------------------------------

    def update_slices(self):
        """Recreate the slice planes for the current frame and slider positions"""
        renderer = self.vtk_widget.renderer
        for actor in self.slice_actors:
            renderer.RemoveActor(actor)
        self.slice_actors = []

        frame = self._frame_data
        slices = self.control_panel.get_slices()
        for axis, label in enumerate(self.control_panel.slice_labels):
            label.setText("")

        if frame is not None and frame.mesh.n_cells > 0:
            bounds = frame.mesh.bounds
            center = list(frame.mesh.center)
            for axis, position in slices:
                low, high = bounds[2 * axis], bounds[2 * axis + 1]
                origin = list(center)
                origin[axis] = low + position * (high - low)
                self.control_panel.slice_labels[axis].setText(f"{'xyz'[axis]} = {origin[axis]:.2f}")

                normal = [0, 0, 0]
                normal[axis] = 1
                plane = frame.mesh.slice(normal=normal, origin=origin)
                if plane.n_points == 0:
                    continue

                # Colored with the shared color function, unlit so colors match the color bar
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(plane)
                mapper.SetLookupTable(self.color_func)
                mapper.UseLookupTableScalarRangeOn()
                mapper.ScalarVisibilityOn()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().LightingOff()
                renderer.AddActor(actor)
                self.slice_actors.append(actor)

        self.vtk_widget.render()

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        """Detect plain left clicks (press and release without dragging) in the 3D view"""
        if obj is self.vtk_widget.vtk_widget and event.type() in (QEvent.Type.MouseButtonPress,
                                                                  QEvent.Type.MouseButtonRelease):
            if event.button() == Qt.MouseButton.LeftButton:
                position = event.position()
                if event.type() == QEvent.Type.MouseButtonPress:
                    # Modifier clicks are camera gestures (pan, roll)
                    plain = event.modifiers() == Qt.KeyboardModifier.NoModifier
                    self._press_position = position if plain else None
                elif self._press_position is not None:
                    press, self._press_position = self._press_position, None
                    if (position - press).manhattanLength() <= 4:  # Not dragged
                        self.on_view_clicked(obj, position)
        return super().eventFilter(obj, event)

    def on_view_clicked(self, view, position):
        # Convert to VTK display coordinates (device pixels, origin at the bottom left)
        ratio = view.devicePixelRatioF()
        x = int(position.x() * ratio)
        y = int((view.height() - position.y()) * ratio) - 1
        if self.box_widget.GetEnabled() and \
                self.box_representation.ComputeInteractionState(x, y, 0) != vtk.vtkBoxRepresentation.Outside:
            return  # Click on the clipping box
        self.probe_at(x, y)

    def view_ray(self, x, y):
        """World points where the view ray through display point (x, y) enters and leaves the view"""
        renderer = self.vtk_widget.renderer
        points = []
        for depth in (0.0, 1.0):
            renderer.SetDisplayPoint(x, y, depth)
            renderer.DisplayToWorld()
            world = renderer.GetWorldPoint()
            points.append(np.array(world[:3]) / world[3])
        return points

    def pick_volume(self, x, y):
        """First point along the view ray where the rendered volume is visibly opaque, or None"""
        frame = self._frame_data
        if frame is None or not self.volume_actor.GetVisibility() or self.global_max <= self.global_min:
            return None

        start, end = self.view_ray(x, y)
        segment = clip_segment_to_bounds(start, end, frame.mesh.bounds)
        if segment is None:
            return None

        points = np.linspace(segment[0], segment[1], PROBE_RAY_SAMPLES)
        values, valid = dvu.sample_points(frame.mesh, points, frame.scalar_name)

        data_range = self.global_max - self.global_min
        positions, opacities = zip(*self.opacity_points)
        opacity = np.interp(values, self.global_min + np.array(positions) * data_range, opacities)
        hits = np.flatnonzero(valid & (opacity >= PROBE_OPACITY_THRESHOLD))
        return points[hits[0]] if len(hits) else None

    def pick_surface(self, x, y):
        """Point on a slice plane or isosurface under the display point, or None"""
        actors = self.slice_actors + ([self.iso_actor] if self.iso_actor is not None else [])
        if not actors:
            return None
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.PickFromListOn()
        for actor in actors:
            picker.AddPickList(actor)
        if picker.Pick(x, y, 0, self.vtk_widget.renderer):
            return np.array(picker.GetPickPosition())
        return None

    def probe_at(self, x, y):
        """Probe the visible data under display point (x, y)"""
        if self._frame_data is None:
            return

        hits = [p for p in (self.pick_volume(x, y), self.pick_surface(x, y)) if p is not None]
        if not hits:
            self.statusBar().showMessage("No visible data under the cursor")
            return

        # Use the hit nearest to the camera
        camera = np.array(self.vtk_widget.renderer.GetActiveCamera().GetPosition())
        point = min(hits, key=lambda p: np.linalg.norm(p - camera))
        self.start_probe(point)

    def start_probe(self, point):
        """Show the probe at a point and compute its time series"""
        frame = self._frame_data
        self._probe_point = point
        self._probe_scalar = frame.scalar_name
        self._probe_id += 1

        # Marker size relative to the data
        self.probe_source.SetCenter(*point)
        self.probe_source.SetRadius(0.008 * frame.mesh.length)
        self.probe_actor.SetVisibility(True)
        self.vtk_widget.render()

        panel = self.control_panel.probe_panel
        panel.set_probe(point, frame.scalar_name, self.vtk_files)
        panel.set_current_frame(self._displayed_key[1])
        self.control_panel.tab_widget.setCurrentWidget(panel)

        # Value of the displayed frame right away
        values, valid = dvu.sample_points(frame.mesh, [point], frame.scalar_name)
        value = float(values[0]) if valid[0] else float('nan')
        panel.set_value(self._displayed_key[1], value)
        log.info("Probe at (%.2f, %.2f, %.2f): %s = %.4g", *point, frame.scalar_name, value)

        # Other frames: sample frames that are already loaded in the GUI (one per
        # event loop pass to stay responsive), read the others in a worker process
        for future in self._probe_futures:
            future.cancel()
        self._probe_futures = []
        cached = {key[1]: data for key, data in self._frame_cache.items()
                  if key[0] == self.data_location and data.scalar_name == frame.scalar_name}
        self._probe_queue = [(frame_index, cached[frame_index]) for frame_index in self.vtk_files
                             if frame_index in cached and frame_index != self._displayed_key[1]]
        QTimer.singleShot(0, lambda probe_id=self._probe_id: self.probe_next_cached(probe_id))

        for frame_index in self.vtk_files:
            if frame_index != self._displayed_key[1] and frame_index not in cached:
                self.submit_probe(frame_index)

    def probe_next_cached(self, probe_id):
        """Sample the probe point in the next loaded frame"""
        if probe_id != self._probe_id or not self._probe_queue:
            return
        frame_index, frame = self._probe_queue.pop(0)
        values, valid = dvu.sample_points(frame.mesh, [self._probe_point], frame.scalar_name)
        if valid[0]:
            self.control_panel.probe_panel.set_value(frame_index, float(values[0]))
        else:
            self.submit_probe(frame_index)  # Outside the clipped mesh of the cached frame
        QTimer.singleShot(0, lambda: self.probe_next_cached(probe_id))

    def submit_probe(self, frame_index):
        """Compute the probe value of a frame in the probe worker process"""
        if self._probe_executor is None:
            self._probe_executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        future = self._probe_executor.submit(dvu.probe_frame,
                                             os.path.join(self.data_location, self.vtk_files[frame_index]),
                                             self._probe_scalar, tuple(self._probe_point))
        future.add_done_callback(lambda f, probe_id=self._probe_id, frame_index=frame_index:
                                 self._emit_probe_value(probe_id, frame_index, f))
        self._probe_futures.append(future)

    def _emit_probe_value(self, probe_id, frame_index, future):
        """Forward a finished probe future to the GUI thread"""
        try:
            self.probe_value_loaded.emit(probe_id, frame_index, future)
        except RuntimeError:
            pass  # Window already destroyed

    def on_probe_value_loaded(self, probe_id, frame_index, future):
        if probe_id != self._probe_id or future.cancelled():
            return  # Result of an earlier probe
        try:
            value = future.result()
        except Exception as e:
            log.error("Failed to probe frame %s: %s", frame_index, e)
            value = float('nan')
        self.control_panel.probe_panel.set_value(frame_index, value)

    def clear_probe(self):
        for future in self._probe_futures:
            future.cancel()
        self._probe_futures = []
        self._probe_point = None
        self._probe_id += 1
        self.probe_actor.SetVisibility(False)
        self.control_panel.probe_panel.clear()
        self.vtk_widget.render()

    def go_to_frame(self, frame_index):
        if frame_index in self.vtk_files:
            self.set_playing(False)
            self.playback.go_to_position(list(self.vtk_files).index(frame_index))

    # ------------------------------------------------------------------
    # Clipping box widget
    # ------------------------------------------------------------------

    def update_box_widget(self):
        """Show the clipping box widget at the current bounds when enabled (rendered by the caller)"""
        enabled = self.control_panel.edit_box_checkbox.isChecked() and self._frame_data is not None
        if enabled:
            bounds = tuple(self.control_panel.get_bounds_values())
            if bounds != self._box_bounds or not self.box_widget.GetEnabled():
                self.box_representation.PlaceWidget(bounds)
                self._box_bounds = bounds
            self.box_widget.On()
        else:
            self.box_widget.Off()

    def on_box_edited(self, widget, event):
        """Apply the box to the clipping bounds when the user releases it"""
        bounds = [round(v, 1) for v in self.box_representation.GetBounds()]
        self._box_bounds = tuple(bounds)
        self.control_panel.set_bounds_values(bounds)
        log.info("Clipping bounds set to %s", ", ".join(f"{v:.1f}" for v in bounds))

    # ------------------------------------------------------------------
    # Camera presets
    # ------------------------------------------------------------------

    def set_camera_preset(self, name):
        """Look at the data from a fixed direction"""
        direction, view_up = CAMERA_PRESETS[name]
        renderer = self.vtk_widget.renderer
        camera = renderer.GetActiveCamera()
        bounds = renderer.ComputeVisiblePropBounds()
        center = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + np.array(direction, dtype=float)))
        camera.SetViewUp(*view_up)
        renderer.ResetCamera()  # Keeps the direction, fits the distance
        self.vtk_widget.render()

    def save_screenshot(self):
        """Save the 3D view as a PNG image"""
        path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "", "PNG Images (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.vtk_widget.capture_screenshot(path)
        log.info("Screenshot saved to %s", path)

    def save_camera(self):
        camera = self.vtk_widget.renderer.GetActiveCamera()
        self._saved_camera = (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp(),
                              camera.GetViewAngle(), camera.GetParallelScale())
        self.statusBar().showMessage("View saved")

    def restore_camera(self):
        if self._saved_camera is None:
            self.statusBar().showMessage("No saved view")
            return
        position, focal_point, view_up, view_angle, parallel_scale = self._saved_camera
        camera = self.vtk_widget.renderer.GetActiveCamera()
        camera.SetPosition(position)
        camera.SetFocalPoint(focal_point)
        camera.SetViewUp(view_up)
        camera.SetViewAngle(view_angle)
        camera.SetParallelScale(parallel_scale)
        self.vtk_widget.renderer.ResetCameraClippingRange()
        self.vtk_widget.render()

    # ------------------------------------------------------------------
    # Data range
    # ------------------------------------------------------------------

    def apply_auto_range(self, update_min=True, update_max=True, initial_iso_value=False):
        """Fit the data range to the displayed frame"""
        if self._frame_data is None:
            log.warning("No data loaded")
            return

        data_range = self._frame_data.robust_range()
        if data_range is None:
            log.warning("No data for '%s' inside the clipping bounds", self.active_scalars)
            return

        auto_min, auto_max = data_range
        new_min = auto_min if update_min else self.control_panel.get_data_min()
        new_max = auto_max if update_max else self.control_panel.get_data_max()
        self.control_panel.set_data_range(new_min, new_max)
        log.info("Data range for %s set to [%.3f, %.3f]", self._frame_data.scalar_name, auto_min, auto_max)

        if initial_iso_value:
            # Set a reasonable default isosurface value (midpoint of data range)
            iso_spinbox = self.control_panel.iso_value_spinbox
            iso_spinbox.blockSignals(True)
            iso_spinbox.setValue(auto_min + 0.5 * (auto_max - auto_min))
            iso_spinbox.blockSignals(False)

        self.read_parameters()

    def auto_detect_scalar_min(self):
        """Auto-detect minimum value for current scalar and update the UI"""
        self.apply_auto_range(update_max=False)
        self.update_isosurfaces()

    def auto_detect_scalar_max(self):
        """Auto-detect maximum value for current scalar and update the UI"""
        self.apply_auto_range(update_min=False)
        self.update_isosurfaces()

    def auto_detect_scalar_range(self):
        """Auto-detect both min and max values for current scalar and update the UI"""
        self.apply_auto_range()
        self.update_isosurfaces()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def on_create_video(self):
        """Create video from frames"""
        if not self.vtk_files:
            QMessageBox.warning(self, "Warning", "No data loaded to create video")
            return

        # Ask for output file
        options = QFileDialog.Option.DontUseNativeDialog
        output_file, _ = QFileDialog.getSaveFileName(self, "Save Video As", "", "MP4 Files (*.mp4);;All Files (*)", options=options)
        if not output_file:
            return

        self.set_playing(False)
        self._exporting = True
        frame_indices = sorted(self.vtk_files.keys())
        progress = self.control_panel.progress_bar
        progress.setRange(0, len(frame_indices))
        self.control_panel.show_progress(True)

        try:
            # Create a temporary directory to store images
            temp_dir = "temp_video_frames"
            os.makedirs(temp_dir, exist_ok=True)

            # Queue all frames up front so the worker prepares the next frames
            # while the current one is rendered
            keys = [self.frame_key(frame_index) for frame_index in frame_indices]
            futures = {key: self.submit_frame(key) for key in keys if self._frame_cache.get(key) is None}

            # Render each frame with the current camera and save as image
            for image_number, key in enumerate(keys):
                frame = self._frame_cache.get(key)
                if frame is None:
                    # Cached frames may have been evicted in the meantime
                    future = futures.get(key) or self.submit_frame(key)
                    frame = dvu.FrameData(future.result())
                    self._frame_cache.put(key, frame)
                self.show_frame(key, frame)
                image_path = os.path.join(temp_dir, f"frame_{image_number:04d}.png")
                self.vtk_widget.capture_screenshot(image_path)

                progress.setValue(image_number + 1)
                QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

            # Use ffmpeg to create video from images (libx264 needs even frame sizes)
            fps = self.playback.fps_spinbox.value()
            ffmpeg_command = (f'ffmpeg -y -framerate {fps} -i "{temp_dir}/frame_%04d.png" '
                              f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "{output_file}"')
            if os.system(ffmpeg_command) != 0:
                raise RuntimeError("ffmpeg failed, see console output for details")

            # Clean up temporary images
            #for file in os.listdir(temp_dir):
            #    os.remove(os.path.join(temp_dir, file))
            #os.rmdir(temp_dir)

            log.info("Video saved to %s", output_file)
            QMessageBox.information(self, "Success", f"Video saved to {output_file}")

        except Exception as e:
            log.error("Failed to create video: %s", e)
            QMessageBox.critical(self, "Error", f"Failed to create video: {str(e)}")

        finally:
            self._exporting = False
            self.control_panel.show_progress(False)
            # Return to the frame selected in the timeline
            self.request_frame()


def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Dam Resistivity Visualizer")
    app.setApplicationVersion("1.0")
       
    # Create and show main window
    window = DamVisualizationApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()