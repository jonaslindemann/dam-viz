#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qt-based dam resistivity visualization application using VTK Qt interface"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QSlider, QLabel, QPushButton, 
                            QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
                            QFileDialog, QMessageBox, QCheckBox, QProgressBar,
                            QSplitter, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPalette

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import pyvista as pv

import damvis_utils as dvu


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
        
        # Set up renderer

        self.renderer.SetBackground(0.2, 0.3, 0.4)  # Standard VTK background color

        # Set up camera for better initial view
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(10, 10, 30)
        camera.SetFocalPoint(10, 12, 24.5)  # Center of typical bounds
        camera.SetViewUp(0, 0, 1)
        
        # Initialize color bar actor only
        self.scalar_bar_actor = None
        self.cube_axes_actor = None
        
        # Initialize the interactor
        self.interactor.Initialize()
        self.interactor.Start()
    
    def add_scalar_bar(self, color_function=None, data_range=None, title="Resistivity (log10)", show_bar=True):
        """Add/remove scalar bar (color scale)"""
        if self.scalar_bar_actor:
            self.renderer.RemoveActor(self.scalar_bar_actor)
            self.scalar_bar_actor = None
        
        if show_bar and color_function and data_range:
            # Create scalar bar
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(color_function)
            scalar_bar.SetTitle(title)
            scalar_bar.SetNumberOfLabels(4)
            
            # Position and size - improved readability
            scalar_bar.SetPosition(0.88, 0.2)   # Slightly more inward for better visibility
            scalar_bar.SetWidth(0.1)            # Wider for better readability
            scalar_bar.SetHeight(0.6)           # Taller for better proportion
            
            # Enhanced font styling for better readability
            title_prop = scalar_bar.GetTitleTextProperty()
            label_prop = scalar_bar.GetLabelTextProperty()
            
            # Title styling
            title_prop.SetColor(1, 1, 1)        # White text
            title_prop.SetFontSize(16)          # Much larger and readable
            title_prop.SetFontFamilyToArial()   # Clean, modern font
            title_prop.BoldOn()                 # Bold for emphasis
            title_prop.ShadowOn()               # Add shadow for better contrast
            
            # Label styling  
            label_prop.SetColor(1, 1, 1)        # White text
            label_prop.SetFontSize(14)          # Larger and readable
            label_prop.SetFontFamilyToArial()   # Clean, modern font
            label_prop.ShadowOn()               # Add shadow for better contrast
            
            # Additional scalar bar formatting
            scalar_bar.SetNumberOfLabels(6)     # More labels for better precision
            scalar_bar.SetLabelFormat("%.2f")   # Two decimal places
            
            self.scalar_bar_actor = scalar_bar
            self.renderer.AddActor2D(self.scalar_bar_actor)
    
    def add_volume_actor(self, volume_actor):
        """Add volume actor to renderer"""
        if volume_actor:
            # Check if it's a volume or regular actor
            if hasattr(volume_actor, 'GetMapper') and isinstance(volume_actor.GetMapper(), vtk.vtkSmartVolumeMapper):
                self.renderer.AddVolume(volume_actor)  # Use AddVolume for volume actors
            else:
                self.renderer.AddActor(volume_actor)   # Use AddActor for regular actors
        self.render_window.Render()
    
    def add_cube_axes_actor(self, cube_axes_actor):
        """Add cube axes actor to renderer"""
        if cube_axes_actor:
            if self.cube_axes_actor:
                self.renderer.RemoveActor(self.cube_axes_actor)
            
            self.cube_axes_actor = cube_axes_actor
            self.renderer.AddActor(cube_axes_actor)
        self.render_window.Render()
    
    def remove_all_actors(self):
        """Remove all actors from renderer"""
        self.renderer.RemoveAllViewProps()
        # Reset actor references
        self.scalar_bar_actor = None
        self.cube_axes_actor = None
        self.render_window.Render()
    
    def reset_camera(self):
        """Reset camera to fit all objects"""
        self.renderer.ResetCamera()
        self.render_window.Render()
    
    def render(self):
        """Render the scene"""
        self.render_window.Render()


class ControlPanel(QWidget):
    """Control panel for visualization parameters"""
    
    # Signals
    apply_changes = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """Set up the control panel UI"""
        layout = QVBoxLayout()
        
        # Frame selection group
        frame_group = QGroupBox("Frame Selection")
        frame_layout = QVBoxLayout()
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.setValue(0)
        
        self.frame_label = QLabel("Frame: 0")
        
        frame_layout.addWidget(self.frame_label)
        frame_layout.addWidget(self.frame_slider)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)
        
        # Opacity controls group
        opacity_group = QGroupBox("Opacity Transfer Function")
        opacity_layout = QVBoxLayout()
        
        # Create min/max labels at the top
        minmax_layout = QHBoxLayout()
        self.left_label = QLabel("Min")
        self.left_label.setAlignment(Qt.AlignLeft)
        self.left_label.setStyleSheet("font-weight: bold; color: blue;")

        self.right_label = QLabel("Max")
        self.right_label.setAlignment(Qt.AlignRight)
        self.right_label.setStyleSheet("font-weight: bold; color: red;")
        
        minmax_layout.addWidget(self.left_label)
        minmax_layout.addStretch()  # Push labels to opposite sides
        minmax_layout.addWidget(self.right_label)
        
        opacity_layout.addLayout(minmax_layout)
        
        # Create horizontal layout for sliders
        sliders_layout = QHBoxLayout()
        
        self.opacity_sliders = []
        self.opacity_labels = []
        
        for i in range(18):
            # Create vertical layout for each slider
            slider_column = QVBoxLayout()
            
            # Value label at top            
            # Vertical slider
            slider = QSlider(Qt.Vertical)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(10)  # Default value
            slider.setFixedHeight(150)  # Set consistent height
            slider.setMinimumWidth(30)  # Set consistent width
            slider_column.addWidget(slider)
            
            # Level label at bottom
                        
            # Add to horizontal layout
            sliders_layout.addLayout(slider_column)
            
            self.opacity_sliders.append(slider)
        
        opacity_layout.addLayout(sliders_layout)
        opacity_group.setLayout(opacity_layout)
        layout.addWidget(opacity_group)
        
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
        
        bounds_group.setLayout(bounds_layout)
        layout.addWidget(bounds_group)
        
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
        
        # Target cells for resampling
        target_cells_layout = QHBoxLayout()
        target_cells_layout.addWidget(QLabel("Target Cells:"))
        self.target_cells_spinbox = QSpinBox()
        self.target_cells_spinbox.setRange(10000, 2000000)
        self.target_cells_spinbox.setValue(500000)
        self.target_cells_spinbox.setSuffix(" cells")
        self.target_cells_spinbox.setSingleStep(50000)
        target_cells_layout.addWidget(self.target_cells_spinbox)
        render_layout.addLayout(target_cells_layout)
        
        # Show bounds checkbox
        self.show_bounds_checkbox = QCheckBox("Show Bounds")
        self.show_bounds_checkbox.setChecked(True)
        render_layout.addWidget(self.show_bounds_checkbox)
        
        # Show color bar checkbox
        self.show_colorbar_checkbox = QCheckBox("Show Color Bar")
        self.show_colorbar_checkbox.setChecked(True)
        render_layout.addWidget(self.show_colorbar_checkbox)
        
        render_group.setLayout(render_layout)
        layout.addWidget(render_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply Changes")
        self.reset_button = QPushButton("Reset Camera")
        
        # Set fixed size for all buttons to make them uniform
        button_size = (120, 35)
        self.apply_button.setFixedSize(*button_size)
        self.reset_button.setFixedSize(*button_size)
        
        # Initialize dirty flag and button styles
        self._is_dirty = False
        self._setup_button_styles()
        
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reset_button)
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _setup_button_styles(self):
        """Set up button styles for clean and dirty states"""
        self._clean_style = """
            QPushButton { 
                background-color: #2196F3; 
                color: white;
                font-weight: bold; 
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { 
                background-color: #1976D2; 
            }
            QPushButton:pressed { 
                background-color: #1565C0; 
            }
        """
        
        self._dirty_style = """
            QPushButton { 
                background-color: #FF9800; 
                color: white;
                font-weight: bold; 
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { 
                background-color: #F57C00; 
            }
            QPushButton:pressed { 
                background-color: #E65100; 
            }
        """
        
        # Set initial clean state
        self.apply_button.setStyleSheet(self._clean_style)
        self.apply_button.setText("Apply Changes")
    
    def set_dirty(self, dirty=True):
        """Set dirty flag and update button appearance"""
        self._is_dirty = dirty
        if dirty:
            self.apply_button.setStyleSheet(self._dirty_style)
            self.apply_button.setText("Apply Changes")
        else:
            self.apply_button.setStyleSheet(self._clean_style)
            self.apply_button.setText("Apply Changes")
    
    def is_dirty(self):
        """Check if parameters have been modified"""
        return self._is_dirty
    
    def connect_signals(self):
        """Connect widget signals"""
        # Connect parameter change handlers to set dirty flag
        self.frame_slider.valueChanged.connect(self.on_parameter_changed)
        
        for i, slider in enumerate(self.opacity_sliders):
            slider.valueChanged.connect(self.on_opacity_changed)
            slider.valueChanged.connect(self.on_parameter_changed)
        
        for spinbox in self.bounds_spinboxes:
            spinbox.valueChanged.connect(self.on_bounds_changed)
            spinbox.valueChanged.connect(self.on_parameter_changed)
        
        self.colormap_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.target_cells_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.show_bounds_checkbox.toggled.connect(self.on_parameter_changed)
        self.show_colorbar_checkbox.toggled.connect(self.on_parameter_changed)
        
        # Connect apply button
        self.apply_button.clicked.connect(self.on_apply_clicked)
    
    def on_parameter_changed(self):
        """Handle any parameter change - set dirty flag"""
        # Update frame label if it was the frame slider
        sender = self.sender()
        if sender == self.frame_slider:
            self.frame_label.setText(f"Frame: {sender.value()}")
        
        # Set dirty flag
        self.set_dirty(True)
    
    def on_opacity_changed(self):
        """Handle opacity slider changes - update labels only"""
        pass
    
    def on_bounds_changed(self):
        """Handle bounds spinbox changes - no immediate action"""
        pass  # Just update the UI, changes applied when Apply is clicked
    
    def on_apply_clicked(self):
        """Handle apply button click"""
        self.apply_changes.emit()
        self.set_dirty(False)  # Clear dirty flag after applying
    
    def set_frame_range(self, min_frame, max_frame):
        """Set the range for frame slider"""
        self.frame_slider.setMinimum(min_frame)
        self.frame_slider.setMaximum(max_frame)
    
    def set_opacity_values(self, opacity_list):
        """Set opacity slider values"""
        for i, value in enumerate(opacity_list):
            if i < len(self.opacity_sliders):
                self.opacity_sliders[i].setValue(int(value * 100))
    
    def set_bounds_values(self, bounds_list):
        """Set bounds spinbox values"""
        for i, value in enumerate(bounds_list):
            if i < len(self.bounds_spinboxes):
                self.bounds_spinboxes[i].setValue(value)
    
    def show_progress(self, show=True):
        """Show/hide progress bar"""
        self.progress_bar.setVisible(show)
    
    def set_progress(self, value):
        """Set progress bar value"""
        self.progress_bar.setValue(value)
    
    def get_opacity_values(self):
        """Get current opacity values from sliders"""
        opacity_values = []
        for slider in self.opacity_sliders:
            value = slider.value() / 100.0
            opacity_values.append(value)
        return opacity_values
    
    def get_bounds_values(self):
        """Get current bounds values from spinboxes"""
        return [spinbox.value() for spinbox in self.bounds_spinboxes]
    
    def get_colormap(self):
        """Get current colormap selection"""
        return self.colormap_combo.currentText()
    
    def get_target_cells(self):
        """Get current target cells value"""
        return self.target_cells_spinbox.value()
    
    def is_show_bounds_enabled(self):
        """Get show bounds checkbox state"""
        return self.show_bounds_checkbox.isChecked()
    
    def is_show_colorbar_enabled(self):
        """Get show color bar checkbox state"""
        return self.show_colorbar_checkbox.isChecked()
    
    def get_current_frame(self):
        """Get current frame value from slider"""
        return self.frame_slider.value()
    
    def update_minmax_labels(self, global_min, global_max):
        """Update the min/max labels with actual data range values"""
        self.left_label.setText(f"{global_min:.3f}")
        self.right_label.setText(f"{global_max:.3f}")


class DamVisualizationApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize data
        self.data_location = None
        self.vtk_files = {}
        self.current_volume_actor = None
        self.bounds_actor = None
        self.current_color_function = None
        
        # Visualization parameters
        self.opacity = [0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0]
        self.bounds = [2, 17, 2, 22, 22, 27]
        self.colormap = 'RdYlBu_r'
        self.target_cells = 500000
        self.show_bounds = True
        self.show_colorbar = True
        self.global_min = -0.189
        self.global_max = 4.970
        
        self.setup_ui()
        self.connect_signals()
        
        # Try to load default data location
        default_path = "/home/bmjl/lu2023-17-17/Inversion_RealData/Results"
        if os.path.exists(default_path):
            self.load_data_location(default_path)
    
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
        splitter = QSplitter(Qt.Horizontal)
        
        # VTK visualization widget
        self.vtk_widget = VTKVisualizationWidget()
        splitter.addWidget(self.vtk_widget)
        
        # Control panel
        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(600)
        self.control_panel.setMinimumWidth(400)
        # Initialize min/max labels with global data range
        self.control_panel.update_minmax_labels(self.global_min, self.global_max)
        splitter.addWidget(self.control_panel)
        
        # Set splitter proportions
        splitter.setSizes([1000, 400])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # Menu bar
        self.create_menu_bar()
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = file_menu.addAction('Open Data Location...')
        open_action.triggered.connect(self.open_data_location)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        reset_camera_action = view_menu.addAction('Reset Camera')
        reset_camera_action.triggered.connect(self.vtk_widget.reset_camera)
    
    def connect_signals(self):
        """Connect signals"""
        self.control_panel.apply_changes.connect(self.apply_parameter_changes)
        self.control_panel.reset_button.clicked.connect(self.vtk_widget.reset_camera)
    
    def open_data_location(self):
        """Open dialog to select data location"""
        folder = QFileDialog.getExistingDirectory(self, "Select Data Location")
        if folder:
            self.load_data_location(folder)
    
    def load_data_location(self, folder_path):
        """Load VTK files from data location"""
        self.data_location = folder_path
        self.vtk_files = {}
        
        try:
            # Find VTK files
            for file in os.listdir(folder_path):
                if file.startswith("dcinv") and file.endswith(".vtk"):
                    number = int(file.split('_')[-1].split('.')[0])
                    self.vtk_files[number] = file
            
            if not self.vtk_files:
                QMessageBox.warning(self, "Warning", "No VTK files found in selected directory")
                return
            
            # Sort files
            sorted_keys = sorted(self.vtk_files.keys())
            self.vtk_files = {key: self.vtk_files[key] for key in sorted_keys}
            
            # Update control panel
            min_frame = min(self.vtk_files.keys())
            max_frame = max(self.vtk_files.keys())
            self.control_panel.set_frame_range(min_frame, max_frame)
            self.control_panel.set_opacity_values(self.opacity)
            self.control_panel.set_bounds_values(self.bounds)
            self.control_panel.update_minmax_labels(self.global_min, self.global_max)
            
            # Set initial frame but don't load until Apply is clicked
            self.control_panel.frame_slider.setValue(min_frame)
            
            # Load first frame initially
            self.update_visualization(min_frame)
            
            # Ensure camera is properly positioned for the initial view
            self.vtk_widget.reset_camera()
            
            # Clear dirty flag after initial load
            self.control_panel.set_dirty(False)
            
            self.statusBar().showMessage(f"Loaded {len(self.vtk_files)} VTK files from {folder_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
    
    def create_volume_actor(self, mesh):
        """Create VTK volume actor from mesh"""
        try:
            # Ensure mesh has point data
            if mesh.active_scalars_name in mesh.cell_data:
                mesh = mesh.cell_data_to_point_data()
            
            # Clip mesh
            clipped = mesh.clip_box(bounds=self.bounds, invert=False)
            
            # Resample to uniform grid
            resampled = dvu.resample_to_uniform_grid(clipped, target_cells=self.target_cells)
            resampled.set_active_scalars('Resistivity(log10)')
            
            # Convert PyVista mesh to VTK ImageData for volume rendering
            if hasattr(resampled, 'cast_to_image_data'):
                vtk_data = resampled.cast_to_image_data()
            else:
                # Fallback: resampled should be ImageData already from resample_to_uniform_grid
                vtk_data = resampled
            
            print(f"VTK data type: {type(vtk_data)}")
            print(f"VTK data bounds: {vtk_data.GetBounds()}")
            print(f"VTK data dimensions: {vtk_data.GetDimensions()}")
            
            # Create volume mapper
            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputData(vtk_data)
            mapper.SetRequestedRenderModeToGPU()  # Use GPU rendering if available
            
            # Create volume property
            volume_property = vtk.vtkVolumeProperty()
            volume_property.ShadeOn()
            volume_property.SetInterpolationTypeToLinear()
            
            # Create color transfer function
            color_func = vtk.vtkColorTransferFunction()
            
            # Set up colormap based on selected colormap
            if self.colormap == 'RdYlBu_r':
                # Red-Yellow-Blue reversed
                color_func.AddRGBPoint(self.global_min, 0.0, 0.0, 1.0)  # Blue
                color_func.AddRGBPoint(self.global_min + 0.3 * (self.global_max - self.global_min), 0.0, 1.0, 1.0)  # Cyan
                color_func.AddRGBPoint(self.global_min + 0.5 * (self.global_max - self.global_min), 1.0, 1.0, 0.0)  # Yellow
                color_func.AddRGBPoint(self.global_min + 0.7 * (self.global_max - self.global_min), 1.0, 0.5, 0.0)  # Orange
                color_func.AddRGBPoint(self.global_max, 1.0, 0.0, 0.0)  # Red
            elif self.colormap == 'viridis':
                # Viridis colormap
                color_func.AddRGBPoint(self.global_min, 0.267, 0.004, 0.329)  # Dark purple
                color_func.AddRGBPoint(self.global_min + 0.25 * (self.global_max - self.global_min), 0.229, 0.322, 0.545)  # Purple-blue
                color_func.AddRGBPoint(self.global_min + 0.5 * (self.global_max - self.global_min), 0.127, 0.566, 0.550)  # Teal
                color_func.AddRGBPoint(self.global_min + 0.75 * (self.global_max - self.global_min), 0.369, 0.788, 0.382)  # Green
                color_func.AddRGBPoint(self.global_max, 0.993, 0.906, 0.144)  # Yellow
            elif self.colormap == 'plasma':
                # Plasma colormap
                color_func.AddRGBPoint(self.global_min, 0.050, 0.030, 0.529)  # Dark blue
                color_func.AddRGBPoint(self.global_min + 0.25 * (self.global_max - self.global_min), 0.494, 0.016, 0.655)  # Purple
                color_func.AddRGBPoint(self.global_min + 0.5 * (self.global_max - self.global_min), 0.808, 0.067, 0.472)  # Magenta
                color_func.AddRGBPoint(self.global_min + 0.75 * (self.global_max - self.global_min), 0.965, 0.451, 0.176)  # Orange
                color_func.AddRGBPoint(self.global_max, 0.984, 0.906, 0.145)  # Yellow
            elif self.colormap == 'inferno':
                # Inferno colormap
                color_func.AddRGBPoint(self.global_min, 0.000, 0.000, 0.014)  # Almost black
                color_func.AddRGBPoint(self.global_min + 0.25 * (self.global_max - self.global_min), 0.341, 0.062, 0.429)  # Dark purple
                color_func.AddRGBPoint(self.global_min + 0.5 * (self.global_max - self.global_min), 0.733, 0.216, 0.329)  # Red
                color_func.AddRGBPoint(self.global_min + 0.75 * (self.global_max - self.global_min), 0.976, 0.576, 0.176)  # Orange
                color_func.AddRGBPoint(self.global_max, 0.988, 0.998, 0.645)  # Light yellow
            elif self.colormap == 'jet':
                # Jet colormap (traditional blue-cyan-yellow-red)
                color_func.AddRGBPoint(self.global_min, 0.0, 0.0, 0.5)  # Dark blue
                color_func.AddRGBPoint(self.global_min + 0.2 * (self.global_max - self.global_min), 0.0, 0.0, 1.0)  # Blue
                color_func.AddRGBPoint(self.global_min + 0.4 * (self.global_max - self.global_min), 0.0, 1.0, 1.0)  # Cyan
                color_func.AddRGBPoint(self.global_min + 0.6 * (self.global_max - self.global_min), 1.0, 1.0, 0.0)  # Yellow
                color_func.AddRGBPoint(self.global_min + 0.8 * (self.global_max - self.global_min), 1.0, 0.0, 0.0)  # Red
                color_func.AddRGBPoint(self.global_max, 0.5, 0.0, 0.0)  # Dark red
            elif self.colormap == 'rainbow':
                # Rainbow colormap (spectral colors)
                color_func.AddRGBPoint(self.global_min, 0.5, 0.0, 1.0)  # Purple
                color_func.AddRGBPoint(self.global_min + 0.17 * (self.global_max - self.global_min), 0.0, 0.0, 1.0)  # Blue
                color_func.AddRGBPoint(self.global_min + 0.33 * (self.global_max - self.global_min), 0.0, 1.0, 1.0)  # Cyan
                color_func.AddRGBPoint(self.global_min + 0.5 * (self.global_max - self.global_min), 0.0, 1.0, 0.0)  # Green
                color_func.AddRGBPoint(self.global_min + 0.67 * (self.global_max - self.global_min), 1.0, 1.0, 0.0)  # Yellow
                color_func.AddRGBPoint(self.global_min + 0.83 * (self.global_max - self.global_min), 1.0, 0.5, 0.0)  # Orange
                color_func.AddRGBPoint(self.global_max, 1.0, 0.0, 0.0)  # Red
            else:
                # Default fallback to RdYlBu_r if unknown colormap
                color_func.AddRGBPoint(self.global_min, 0.0, 0.0, 1.0)  # Blue
                color_func.AddRGBPoint(self.global_min + 0.3 * (self.global_max - self.global_min), 0.0, 1.0, 1.0)  # Cyan
                color_func.AddRGBPoint(self.global_min + 0.5 * (self.global_max - self.global_min), 1.0, 1.0, 0.0)  # Yellow
                color_func.AddRGBPoint(self.global_min + 0.7 * (self.global_max - self.global_min), 1.0, 0.5, 0.0)  # Orange
                color_func.AddRGBPoint(self.global_max, 1.0, 0.0, 0.0)  # Red
            
            volume_property.SetColor(color_func)
            
            # Create opacity transfer function
            opacity_func = vtk.vtkPiecewiseFunction()
            
            # Map opacity values to data range
            data_range = self.global_max - self.global_min
            for i, opacity_val in enumerate(self.opacity):
                if data_range > 0:
                    value = self.global_min + (i / max(1, len(self.opacity) - 1)) * data_range
                    opacity_func.AddPoint(value, opacity_val)
            
            volume_property.SetScalarOpacity(opacity_func)
            
            # Create volume actor
            volume_actor = vtk.vtkVolume()
            volume_actor.SetMapper(mapper)
            volume_actor.SetProperty(volume_property)
            
            # Store color function for scalar bar
            self.current_color_function = color_func
            
            # Print volume bounds for debugging
            bounds = volume_actor.GetBounds()
            print(f"Volume actor bounds: {bounds}")
            
            return volume_actor
            
        except Exception as e:
            print(f"Error creating volume actor: {e}")
            # Return a simple wireframe as fallback
            return self.create_fallback_actor(mesh)
    
    def create_fallback_actor(self, mesh):
        """Create a fallback wireframe actor if volume rendering fails"""
        try:
            # Clip mesh
            clipped = mesh.clip_box(bounds=self.bounds, invert=False)
            
            # Convert to VTK PolyData
            surface = clipped.extract_surface()
            vtk_polydata = surface
            
            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(vtk_polydata)
            mapper.SetScalarRange(self.global_min, self.global_max)
            
            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetColor(1.0, 0.5, 0.0)  # Orange
            
            return actor
            
        except Exception as e:
            print(f"Error creating fallback actor: {e}")
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
    
    def update_visualization(self, frame_index):
        """Update visualization for given frame"""
        if not self.vtk_files or frame_index not in self.vtk_files:
            return
        
        try:
            # Remove existing actors
            self.vtk_widget.remove_all_actors()
            
            # Load mesh
            file_path = os.path.join(self.data_location, self.vtk_files[frame_index])
            mesh = pv.read(file_path)
            mesh.set_active_scalars("Resistivity(log10)")
            
            # Create volume actor
            self.current_volume_actor = self.create_volume_actor(mesh)
            if self.current_volume_actor:
                self.vtk_widget.add_volume_actor(self.current_volume_actor)
                print(f"Volume actor created and added for frame {frame_index}")
            else:
                print(f"Failed to create volume actor for frame {frame_index}")
            
            # Add bounds with ParaView-style axes grid if enabled
            if self.show_bounds:
                self.bounds_actor = self.create_bounds_actor()
                if self.bounds_actor:
                    self.vtk_widget.add_cube_axes_actor(self.bounds_actor)
            
            # Add color bar if enabled and we have a volume actor with color function
            if self.show_colorbar and self.current_volume_actor and self.current_color_function:
                self.vtk_widget.add_scalar_bar(
                    color_function=self.current_color_function,
                    data_range=[self.global_min, self.global_max],
                    title="Resistivity (log10)",
                    show_bar=True
                )
            
            # Reset camera to fit the new content and render
            self.vtk_widget.reset_camera()
            self.vtk_widget.render()
            
            self.statusBar().showMessage(f"Displaying frame {frame_index}: {self.vtk_files[frame_index]}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update visualization: {str(e)}")
    
    def apply_parameter_changes(self):
        """Apply all parameter changes from the control panel"""
        # Get current values from control panel
        self.opacity = self.control_panel.get_opacity_values()
        self.bounds = self.control_panel.get_bounds_values()
        self.colormap = self.control_panel.get_colormap()
        self.target_cells = self.control_panel.get_target_cells()
        self.show_bounds = self.control_panel.is_show_bounds_enabled()
        self.show_colorbar = self.control_panel.is_show_colorbar_enabled()
        current_frame = self.control_panel.get_current_frame()
        
        # Update visualization with current frame
        self.update_visualization(current_frame)
        
        self.statusBar().showMessage("Parameters applied successfully")
    



def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Dam Resistivity Visualizer")
    app.setApplicationVersion("1.0")
       
    # Create and show main window
    window = DamVisualizationApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()