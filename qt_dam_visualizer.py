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
                            QSplitter, QFrame, QRadioButton)

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
        
        # Set up renderer with enhanced lighting
        self.renderer.SetBackground(0.1, 0.2, 0.3)  # Standard VTK background color
        
        # Add enhanced lighting setup (default to Enhanced)
        self.setup_enhanced_lighting('Enhanced')

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

    def capture_screenshot(self, filename):
        """Capture screenshot of the current render window"""
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()


class ControlPanel(QWidget):
    """Control panel for visualization parameters"""
    
    # Signals
    apply_changes = pyqtSignal()
    
    def __init__(self, parent=None, main_app=None):
        super().__init__(parent)
        self.main_app = main_app
        
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

        self.frame_video_button = QPushButton("Create Video")
        self.frame_video_button.setEnabled(True)  # Initially enabled

        frame_layout.addWidget(self.frame_label)
        frame_layout.addWidget(self.frame_slider)
        frame_layout.addWidget(self.frame_video_button)

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
        
        # Opacity preset controls
        presets_layout = QHBoxLayout()
        
        # Create preset buttons
        self.preset_full_btn = QPushButton("Full")
        self.preset_linear_up_btn = QPushButton("Linear Up")
        self.preset_linear_down_btn = QPushButton("Linear Down")
        self.preset_max_middle_btn = QPushButton("Max Middle")
        self.preset_max_sides_btn = QPushButton("Max Sides")
        
        # Set fixed size for all preset buttons
        preset_button_size = (70, 25)
        self.preset_full_btn.setFixedSize(*preset_button_size)
        self.preset_linear_up_btn.setFixedSize(*preset_button_size)
        self.preset_linear_down_btn.setFixedSize(*preset_button_size)
        self.preset_max_middle_btn.setFixedSize(*preset_button_size)
        self.preset_max_sides_btn.setFixedSize(*preset_button_size)
        
        # Add buttons to layout
        presets_layout.addWidget(self.preset_full_btn)
        presets_layout.addWidget(self.preset_linear_up_btn)
        presets_layout.addWidget(self.preset_linear_down_btn)
        presets_layout.addWidget(self.preset_max_middle_btn)
        presets_layout.addWidget(self.preset_max_sides_btn)
        presets_layout.addStretch()  # Push buttons to the left
        
        opacity_layout.addLayout(presets_layout)
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
        
        # Active scalars selection
        scalars_layout = QHBoxLayout()
        scalars_layout.addWidget(QLabel("Active Scalars:"))
        self.active_scalars_combo = QComboBox()
        self.active_scalars_combo.addItems(['Resistivity(log10)'])  # Default item, will be updated when data loads
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
        self.data_min_spinbox.setValue(0.0)  # Will be set to actual data min when data loads
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
        self.data_max_spinbox.setValue(1.0)  # Will be set to actual data max when data loads
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
        
        # Lighting quality selection
        lighting_layout = QHBoxLayout()
        lighting_layout.addWidget(QLabel("Lighting:"))
        self.lighting_combo = QComboBox()
        self.lighting_combo.addItems(['Enhanced', 'Standard', 'Minimal'])
        self.lighting_combo.setCurrentText('Enhanced')
        lighting_layout.addWidget(self.lighting_combo)
        render_layout.addLayout(lighting_layout)
        
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
        isosurface_group_layout = QVBoxLayout()
        
        # Enable isosurfaces checkbox
        self.show_isosurfaces_checkbox = QCheckBox("Show Isosurfaces")
        self.show_isosurfaces_checkbox.setChecked(False)
        isosurface_group_layout.addWidget(self.show_isosurfaces_checkbox)
        
        # Isosurface mode selection
        iso_mode_layout = QHBoxLayout()
        iso_mode_layout.addWidget(QLabel("Mode:"))
        self.iso_single_radio = QRadioButton("Single Value")
        self.iso_multiple_radio = QRadioButton("Multiple Surfaces")
        self.iso_single_radio.setChecked(True)  # Default to single value
        iso_mode_layout.addWidget(self.iso_single_radio)
        iso_mode_layout.addWidget(self.iso_multiple_radio)
        isosurface_group_layout.addLayout(iso_mode_layout)
        
        # Single isosurface value control
        iso_value_layout = QHBoxLayout()
        iso_value_layout.addWidget(QLabel("Iso Value:"))
        self.iso_value_spinbox = QDoubleSpinBox()
        self.iso_value_spinbox.setRange(-10000, 10000)
        self.iso_value_spinbox.setValue(2.0)  # Default isosurface value
        self.iso_value_spinbox.setDecimals(3)
        self.iso_value_spinbox.setSingleStep(0.1)
        iso_value_layout.addWidget(self.iso_value_spinbox)
        isosurface_group_layout.addLayout(iso_value_layout)
        
        # Multiple isosurfaces number control
        iso_num_layout = QHBoxLayout()
        iso_num_layout.addWidget(QLabel("Num Surfaces:"))
        self.iso_num_spinbox = QSpinBox()
        self.iso_num_spinbox.setRange(2, 20)
        self.iso_num_spinbox.setValue(5)  # Default number of surfaces
        self.iso_num_spinbox.setEnabled(False)  # Disabled by default (single mode)
        iso_num_layout.addWidget(self.iso_num_spinbox)
        isosurface_group_layout.addLayout(iso_num_layout)
        
        # Isosurface opacity control
        iso_opacity_layout = QHBoxLayout()
        iso_opacity_layout.addWidget(QLabel("Iso Opacity:"))
        self.iso_opacity_spinbox = QDoubleSpinBox()
        self.iso_opacity_spinbox.setRange(0.0, 1.0)
        self.iso_opacity_spinbox.setValue(0.8)  # Default opacity
        self.iso_opacity_spinbox.setDecimals(2)
        self.iso_opacity_spinbox.setSingleStep(0.1)
        iso_opacity_layout.addWidget(self.iso_opacity_spinbox)
        isosurface_group_layout.addLayout(iso_opacity_layout)
        
        render_layout.addLayout(isosurface_group_layout)
        
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
        self.active_scalars_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.data_min_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.data_max_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.data_min_spinbox.valueChanged.connect(self.on_data_range_changed)
        self.data_max_spinbox.valueChanged.connect(self.on_data_range_changed)
        self.target_cells_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.lighting_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.show_volume_checkbox.toggled.connect(self.on_parameter_changed)
        self.auto_hide_volume_checkbox.toggled.connect(self.on_parameter_changed)
        self.show_isosurfaces_checkbox.toggled.connect(self.on_parameter_changed)
        self.iso_single_radio.toggled.connect(self.on_iso_mode_changed)
        self.iso_multiple_radio.toggled.connect(self.on_iso_mode_changed)
        self.iso_value_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.iso_num_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.iso_opacity_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.show_bounds_checkbox.toggled.connect(self.on_parameter_changed)
        self.show_colorbar_checkbox.toggled.connect(self.on_parameter_changed)
        
        # Connect opacity preset buttons
        self.preset_full_btn.clicked.connect(self.apply_opacity_preset_full)
        self.preset_linear_up_btn.clicked.connect(self.apply_opacity_preset_linear_up)
        self.preset_linear_down_btn.clicked.connect(self.apply_opacity_preset_linear_down)
        self.preset_max_middle_btn.clicked.connect(self.apply_opacity_preset_max_middle)
        self.preset_max_sides_btn.clicked.connect(self.apply_opacity_preset_max_sides)
        
        # Connect data range auto-detect buttons
        self.auto_min_btn.clicked.connect(self.auto_detect_min)
        self.auto_max_btn.clicked.connect(self.auto_detect_max)
        self.auto_range_btn.clicked.connect(self.auto_detect_range)
        
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
    
    def on_data_range_changed(self):
        """Handle data range spinbox changes - update min/max labels immediately"""
        # Get current values from spinboxes
        current_min = self.data_min_spinbox.value()
        current_max = self.data_max_spinbox.value()
        
        # Update the min/max labels above the sliders
        self.update_minmax_labels(current_min, current_max)
    
    def on_iso_mode_changed(self):
        """Handle isosurface mode change - enable/disable appropriate controls"""
        is_single_mode = self.iso_single_radio.isChecked()
        
        # Enable/disable controls based on mode
        self.iso_value_spinbox.setEnabled(is_single_mode)
        self.iso_num_spinbox.setEnabled(not is_single_mode)
        
        # Set dirty flag
        self.set_dirty(True)
    
    def on_apply_clicked(self):
        """Handle apply button click"""
        self.apply_changes.emit()
        self.set_dirty(False)  # Clear dirty flag after applying
    
    def apply_opacity_preset_full(self):
        """Set all opacity sliders to maximum (100%)"""
        for slider in self.opacity_sliders:
            slider.setValue(100)
        self.on_opacity_changed()  # Update labels
        self.set_dirty(True)
    
    def apply_opacity_preset_linear_up(self):
        """Set opacity sliders in linear increasing pattern"""
        num_sliders = len(self.opacity_sliders)
        for i, slider in enumerate(self.opacity_sliders):
            # Linear increase from 0% to 100%
            value = int((i / (num_sliders - 1)) * 100)
            slider.setValue(value)
        self.on_opacity_changed()  # Update labels
        self.set_dirty(True)
    
    def apply_opacity_preset_linear_down(self):
        """Set opacity sliders in linear decreasing pattern"""
        num_sliders = len(self.opacity_sliders)
        for i, slider in enumerate(self.opacity_sliders):
            # Linear decrease from 100% to 0%
            value = int(((num_sliders - 1 - i) / (num_sliders - 1)) * 100)
            slider.setValue(value)
        self.on_opacity_changed()  # Update labels
        self.set_dirty(True)
    
    def apply_opacity_preset_max_middle(self):
        """Set opacity sliders with maximum in the middle"""
        num_sliders = len(self.opacity_sliders)
        middle = (num_sliders - 1) / 2.0
        for i, slider in enumerate(self.opacity_sliders):
            # Gaussian-like curve centered at middle
            distance = abs(i - middle) / middle
            value = int((1.0 - distance) * 100)
            slider.setValue(max(0, value))
        self.on_opacity_changed()  # Update labels
        self.set_dirty(True)
    
    def apply_opacity_preset_max_sides(self):
        """Set opacity sliders with maximum at left and right sides"""
        num_sliders = len(self.opacity_sliders)
        middle = (num_sliders - 1) / 2.0
        for i, slider in enumerate(self.opacity_sliders):
            # Inverted gaussian - high at sides, low in middle
            distance = abs(i - middle) / middle
            value = int(distance * 100)
            slider.setValue(min(100, value))
        self.on_opacity_changed()  # Update labels
        self.set_dirty(True)
    
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
    
    def is_show_bounds_enabled(self):
        """Get show bounds checkbox state"""
        return self.show_bounds_checkbox.isChecked()
    
    def is_show_colorbar_enabled(self):
        """Get show color bar checkbox state"""
        return self.show_colorbar_checkbox.isChecked()
    
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
    
    def get_current_frame(self):
        """Get current frame value from slider"""
        return self.frame_slider.value()
    
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
        if current_selection in scalar_names:
            self.active_scalars_combo.setCurrentText(current_selection)
        elif default_scalar in scalar_names:
            self.active_scalars_combo.setCurrentText(default_scalar)
        else:
            # If neither available, select the first item
            if scalar_names:
                self.active_scalars_combo.setCurrentIndex(0)
    
    def set_data_range(self, data_min, data_max):
        """Set the data range spinbox values"""
        self.data_min_spinbox.setValue(data_min)
        self.data_max_spinbox.setValue(data_max)
    
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
        self.active_scalars = 'Resistivity(log10)'
        self.target_cells = 500000
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
        self.current_iso_actors = []  # Changed to list for multiple isosurfaces
        self.global_min = 0.0  # Will be auto-detected from actual data
        self.global_max = 1.0  # Will be auto-detected from actual data
        
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
        self.control_panel = ControlPanel(main_app=self)
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

        self.control_panel.frame_video_button.clicked.connect(self.on_create_video)
    
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
            
            # Detect available scalars from the first VTK file
            self.detect_available_scalars()
            
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
                print(f"Available scalars: {available_scalars}")
                print(f"Selected active scalar: {self.active_scalars}")
                
                # Auto-detect data range for the selected scalar
                self.auto_detect_initial_data_range(mesh)
            else:
                print("No scalar arrays found in VTK file")
                
        except Exception as e:
            print(f"Error detecting scalars: {e}")
            # Use default if detection fails
            self.control_panel.update_available_scalars(['Resistivity(log10)'], self.active_scalars)
            # Update the app's active_scalars to match what the combo box actually selected
            self.active_scalars = self.control_panel.get_active_scalars()
    
    def auto_detect_initial_data_range(self, mesh):
        """Auto-detect and set initial data range when data is first loaded"""
        try:
            # Handle cell data vs point data for selected scalars
            scalar_name = self.active_scalars
            if "(cell)" in scalar_name:
                scalar_name = scalar_name.replace(" (cell)", "")
                if scalar_name in mesh.cell_data:
                    mesh = mesh.cell_data_to_point_data()
            
            # Get the scalar data
            if scalar_name in mesh.point_data:
                data = mesh.point_data[scalar_name]
                auto_min = float(np.nanmin(data))
                auto_max = float(np.nanmax(data))
                
                # Update global values
                self.global_min = auto_min
                self.global_max = auto_max
                
                # Update control panel spinboxes
                self.control_panel.data_min_spinbox.setValue(auto_min)
                self.control_panel.data_max_spinbox.setValue(auto_max)
                
                # Update the min/max labels
                self.control_panel.update_minmax_labels(auto_min, auto_max)
                
                # Set a reasonable default isosurface value (midpoint of data range)
                default_iso_value = auto_min + 0.5 * (auto_max - auto_min)
                self.control_panel.iso_value_spinbox.setValue(default_iso_value)
                self.iso_value = default_iso_value
                
                print(f"Auto-detected initial data range for {self.active_scalars}: [{auto_min:.3f}, {auto_max:.3f}]")
                print(f"Set default isosurface value: {default_iso_value:.3f}")
            else:
                print(f"Warning: Could not auto-detect range for scalar '{scalar_name}'")
                
        except Exception as e:
            print(f"Error auto-detecting initial data range: {e}")
    
    def create_gradient_opacity_function(self):
        """Create gradient opacity function for better depth perception"""
        gradient_opacity = vtk.vtkPiecewiseFunction()
        
        # Define gradient opacity - higher gradients (edges) are more opaque
        gradient_opacity.AddPoint(0.0, 0.0)    # No gradient = transparent
        gradient_opacity.AddPoint(50.0, 0.2)   # Low gradient = slightly opaque
        gradient_opacity.AddPoint(100.0, 0.5)  # Medium gradient = more opaque
        gradient_opacity.AddPoint(200.0, 0.8)  # High gradient = very opaque (edges)
        gradient_opacity.AddPoint(500.0, 1.0)  # Very high gradient = fully opaque
        
        return gradient_opacity
    
    def create_isosurface_actors(self, mesh):
        """Create VTK isosurface actors from mesh (single or multiple surfaces)"""
        try:
            # Handle cell data vs point data for selected scalars
            scalar_name = self.active_scalars
            if "(cell)" in scalar_name:
                # Remove the "(cell)" suffix and convert cell data to point data
                scalar_name = scalar_name.replace(" (cell)", "")
                if scalar_name in mesh.cell_data:
                    mesh = mesh.cell_data_to_point_data()
            else:
                # For point data, ensure it exists
                if scalar_name not in mesh.point_data and mesh.active_scalars_name in mesh.cell_data:
                    mesh = mesh.cell_data_to_point_data()
            
            # Set the active scalars
            try:
                mesh.set_active_scalars(scalar_name)
                print(f"Creating isosurface(s) with active scalars: {scalar_name}")
            except:
                # Fallback to default if the selected scalar doesn't exist
                print(f"Warning: Scalar '{scalar_name}' not found for isosurface, using default")
                if 'Resistivity(log10)' in mesh.point_data:
                    mesh.set_active_scalars('Resistivity(log10)')
                else:
                    # Use the first available scalar
                    available_scalars = list(mesh.point_data.keys())
                    if available_scalars:
                        mesh.set_active_scalars(available_scalars[0])
                        print(f"Using fallback scalar for isosurface: {available_scalars[0]}")
            
            # Clip mesh
            clipped = mesh.clip_box(bounds=self.bounds, invert=False)
            
            # Determine isosurface values
            if self.iso_single_mode:
                iso_values = [self.iso_value]
            else:
                # Create multiple isosurfaces between min and max
                if self.global_max > self.global_min and self.iso_num_surfaces > 1:
                    iso_values = []
                    step = (self.global_max - self.global_min) / (self.iso_num_surfaces + 1)
                    for i in range(1, self.iso_num_surfaces + 1):
                        iso_values.append(self.global_min + i * step)
                else:
                    # Fallback to single value if range is invalid
                    iso_values = [self.iso_value]
            
            print(f"Creating isosurfaces at values: {iso_values}")
            
            actors = []
            for i, iso_val in enumerate(iso_values):
                # Create isosurface using contour filter
                iso_surface = clipped.contour(isosurfaces=[iso_val])
                
                if iso_surface.n_points == 0:
                    print(f"Warning: No isosurface generated for value {iso_val}")
                    continue
                
                # Create mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(iso_surface)
                mapper.SetScalarRange(self.global_min, self.global_max)
                
                # Create color transfer function for isosurface
                color_func = vtk.vtkColorTransferFunction()
                
                # Use same colormap as volume but for isosurface
                if self.colormap == 'RdYlBu_r':
                    color_func.AddRGBPoint(self.global_min, 0.0, 0.0, 1.0)  # Blue
                    color_func.AddRGBPoint(self.global_min + 0.3 * (self.global_max - self.global_min), 0.0, 1.0, 1.0)  # Cyan
                    color_func.AddRGBPoint(self.global_min + 0.5 * (self.global_max - self.global_min), 1.0, 1.0, 0.0)  # Yellow
                    color_func.AddRGBPoint(self.global_min + 0.7 * (self.global_max - self.global_min), 1.0, 0.5, 0.0)  # Orange
                    color_func.AddRGBPoint(self.global_max, 1.0, 0.0, 0.0)  # Red
                elif self.colormap == 'viridis':
                    color_func.AddRGBPoint(self.global_min, 0.267, 0.004, 0.329)  # Dark purple
                    color_func.AddRGBPoint(self.global_min + 0.25 * (self.global_max - self.global_min), 0.229, 0.322, 0.545)  # Purple-blue
                    color_func.AddRGBPoint(self.global_min + 0.5 * (self.global_max - self.global_min), 0.127, 0.566, 0.550)  # Teal
                    color_func.AddRGBPoint(self.global_min + 0.75 * (self.global_max - self.global_min), 0.369, 0.788, 0.382)  # Green
                    color_func.AddRGBPoint(self.global_max, 0.993, 0.906, 0.144)  # Yellow
                else:
                    # Default fallback - use a single color based on iso value position in range
                    if self.global_max > self.global_min:
                        normalized_value = (iso_val - self.global_min) / (self.global_max - self.global_min)
                        # Color based on position: blue (low) -> green (mid) -> red (high)
                        if normalized_value < 0.5:
                            r = 0.0
                            g = normalized_value * 2.0
                            b = 1.0 - normalized_value * 2.0
                        else:
                            r = (normalized_value - 0.5) * 2.0
                            g = 1.0 - (normalized_value - 0.5) * 2.0
                            b = 0.0
                        color_func.AddRGBPoint(self.global_min, r, g, b)
                        color_func.AddRGBPoint(self.global_max, r, g, b)
                    else:
                        color_func.AddRGBPoint(self.global_min, 0.0, 0.8, 1.0)  # Default cyan
                        color_func.AddRGBPoint(self.global_max, 0.0, 0.8, 1.0)
                
                mapper.SetLookupTable(color_func)
                
                # Create actor
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Set actor properties
                base_opacity = self.iso_opacity
                # For multiple surfaces, make them slightly more transparent to avoid visual clutter
                if not self.iso_single_mode and len(iso_values) > 1:
                    base_opacity *= 0.7  # Reduce opacity for multiple surfaces
                
                actor.GetProperty().SetOpacity(base_opacity)
                actor.GetProperty().SetInterpolationToGouraud()  # Smooth shading
                actor.GetProperty().SetSpecular(0.6)  # Add some shininess
                actor.GetProperty().SetSpecularPower(30)
                
                # Ensure surfaces are visible from both sides
                actor.GetProperty().SetBackfaceCulling(False)  # Render back faces
                actor.GetProperty().SetFrontfaceCulling(False)  # Render front faces
                
                # Make the surface more opaque to properly occlude volume rendering
                if base_opacity >= 0.9:  # If nearly opaque, make it fully opaque
                    actor.GetProperty().SetOpacity(1.0)
                
                actors.append(actor)
                print(f"Isosurface actor created for value {iso_val} with {iso_surface.n_points} points")
            
            return actors
            
        except Exception as e:
            print(f"Error creating isosurface actors: {e}")
            return []
    
    def create_volume_actor(self, mesh):
        """Create VTK volume actor from mesh"""
        try:
            # Handle cell data vs point data for selected scalars
            scalar_name = self.active_scalars
            if "(cell)" in scalar_name:
                # Remove the "(cell)" suffix and convert cell data to point data
                scalar_name = scalar_name.replace(" (cell)", "")
                if scalar_name in mesh.cell_data:
                    mesh = mesh.cell_data_to_point_data()
            else:
                # For point data, ensure it exists
                if scalar_name not in mesh.point_data and mesh.active_scalars_name in mesh.cell_data:
                    mesh = mesh.cell_data_to_point_data()
            
            # Set the active scalars
            try:
                mesh.set_active_scalars(scalar_name)
                print(f"Using active scalars: {scalar_name}")
            except:
                # Fallback to default if the selected scalar doesn't exist
                print(f"Warning: Scalar '{scalar_name}' not found, using default")
                if 'Resistivity(log10)' in mesh.point_data:
                    mesh.set_active_scalars('Resistivity(log10)')
                else:
                    # Use the first available scalar
                    available_scalars = list(mesh.point_data.keys())
                    if available_scalars:
                        mesh.set_active_scalars(available_scalars[0])
                        print(f"Using fallback scalar: {available_scalars[0]}")
            
            # Clip mesh
            clipped = mesh.clip_box(bounds=self.bounds, invert=False)
            
            # Resample to uniform grid
            resampled = dvu.resample_to_uniform_grid(clipped, target_cells=self.target_cells)
            
            # Calculate data range for the active scalars
            self.update_data_range(resampled)
            
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
            
            # Improve depth testing for volume rendering
            mapper.SetBlendModeToComposite()  # Use composite blending for better depth handling
            
            # Create volume property with enhanced lighting
            volume_property = vtk.vtkVolumeProperty()
            
            # Enable shading for realistic lighting
            volume_property.ShadeOn()
            volume_property.SetInterpolationTypeToLinear()
            
            # Enhanced lighting properties
            volume_property.SetAmbient(0.2)      # Ambient lighting (base illumination)
            volume_property.SetDiffuse(0.7)      # Diffuse lighting (directional light scattering)
            volume_property.SetSpecular(0.3)     # Specular lighting (shiny highlights)
            volume_property.SetSpecularPower(20) # Specular power (shininess concentration)
            
            # Enable gradient opacity for better depth perception
            volume_property.SetGradientOpacity(0, self.create_gradient_opacity_function())
            
            # Set scattering properties for more realistic volume rendering
            volume_property.SetScalarOpacityUnitDistance(0.5)  # Controls opacity density
            
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
    
    def update_data_range(self, mesh):
        """Update global min/max values based on current active scalars"""
        try:
            # Get the active scalar array
            if mesh.active_scalars is not None:
                # Get auto-detected range
                auto_min, auto_max = mesh.active_scalars.min(), mesh.active_scalars.max()
                
                # Get manual data range from control panel
                manual_min = self.control_panel.get_data_min()
                manual_max = self.control_panel.get_data_max()
                
                old_min, old_max = self.global_min, self.global_max
                
                # Check if user has manually modified the values from the auto-detected ones
                # Only use manual values if they've been deliberately changed from auto values
                tolerance = 0.001
                if (abs(manual_min - auto_min) > tolerance or abs(manual_max - auto_max) > tolerance):
                    # User has manually changed values, use them
                    self.global_min = manual_min
                    self.global_max = manual_max
                    print(f"Using manual data range for '{mesh.active_scalars_name}': [{self.global_min:.3f}, {self.global_max:.3f}]")
                else:
                    # Use auto-detected values and update spinboxes to match
                    self.global_min, self.global_max = auto_min, auto_max
                    # Update spinboxes to reflect the auto-detected values
                    self.control_panel.data_min_spinbox.setValue(auto_min)
                    self.control_panel.data_max_spinbox.setValue(auto_max)
                    print(f"Using auto-detected data range for '{mesh.active_scalars_name}': [{self.global_min:.3f}, {self.global_max:.3f}]")
                
                # Update the control panel min/max labels if the range changed significantly
                if abs(old_min - self.global_min) > 0.001 or abs(old_max - self.global_max) > 0.001:
                    self.control_panel.update_minmax_labels(self.global_min, self.global_max)
                    
            else:
                print("Warning: No active scalars found, keeping existing data range")
                
        except Exception as e:
            print(f"Error updating data range: {e}")
    
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
            
            # Update lighting setup
            self.vtk_widget.setup_enhanced_lighting(self.lighting_quality)
            
            # Load mesh
            file_path = os.path.join(self.data_location, self.vtk_files[frame_index])
            mesh = pv.read(file_path)
            # Active scalars will be set in create_volume_actor based on user selection
            
            # Determine if volume should be shown (considering auto-hide feature)
            should_show_volume = self.show_volume
            if self.auto_hide_volume and self.show_isosurfaces and self.iso_opacity >= 0.9:
                should_show_volume = False
                print(f"Auto-hiding volume due to opaque isosurfaces (opacity: {self.iso_opacity})")
            
            # Create volume actor if enabled
            if should_show_volume:
                self.current_volume_actor = self.create_volume_actor(mesh)
                if self.current_volume_actor:
                    self.vtk_widget.add_volume_actor(self.current_volume_actor)
                    print(f"Volume actor created and added for frame {frame_index}")
                else:
                    print(f"Failed to create volume actor for frame {frame_index}")
            else:
                self.current_volume_actor = None
                if not self.show_volume:
                    print(f"Volume rendering disabled for frame {frame_index}")
                else:
                    print(f"Volume rendering auto-hidden for frame {frame_index}")
            
            # Create isosurface actors if enabled
            if self.show_isosurfaces:
                self.current_iso_actors = self.create_isosurface_actors(mesh)
                if self.current_iso_actors:
                    for i, iso_actor in enumerate(self.current_iso_actors):
                        # Set render order to ensure isosurfaces render after volume
                        iso_actor.GetProperty().SetRenderLinesAsTubes(False)
                        iso_actor.GetProperty().SetRenderPointsAsSpheres(False)
                        
                        self.vtk_widget.add_volume_actor(iso_actor)  # Use add_volume_actor for regular actors too
                    print(f"{len(self.current_iso_actors)} isosurface actor(s) created and added for frame {frame_index}")
                    
                    # If we have both volume and isosurfaces, adjust volume opacity when isosurfaces are opaque
                    if self.show_volume and self.current_volume_actor and self.iso_opacity >= 0.8:
                        # Reduce volume opacity when isosurfaces are nearly opaque to reduce bleeding
                        volume_property = self.current_volume_actor.GetProperty()
                        current_opacity_func = volume_property.GetScalarOpacity()
                        
                        # Scale down the volume opacity slightly
                        scaled_opacity_func = vtk.vtkPiecewiseFunction()
                        for i, opacity_val in enumerate(self.opacity):
                            if self.global_max > self.global_min:
                                value = self.global_min + (i / max(1, len(self.opacity) - 1)) * (self.global_max - self.global_min)
                                # Reduce volume opacity by 30% when isosurfaces are present and opaque
                                scaled_opacity_func.AddPoint(value, opacity_val * 0.7)
                        
                        volume_property.SetScalarOpacity(scaled_opacity_func)
                        print("Reduced volume opacity to prevent bleeding through opaque isosurfaces")
                else:
                    print(f"Failed to create isosurface actors for frame {frame_index}")
            else:
                self.current_iso_actors = []
            
            # Add bounds with ParaView-style axes grid if enabled
            if self.show_bounds:
                self.bounds_actor = self.create_bounds_actor()
                if self.bounds_actor:
                    self.vtk_widget.add_cube_axes_actor(self.bounds_actor)
            
            # Add color bar if enabled and we have a volume actor with color function
            if self.show_colorbar and self.current_volume_actor and self.current_color_function:
                # Clean up the scalar name for display (remove "(cell)" suffix if present)
                display_name = self.active_scalars.replace(" (cell)", "")
                self.vtk_widget.add_scalar_bar(
                    color_function=self.current_color_function,
                    data_range=[self.global_min, self.global_max],
                    title=display_name,
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
        self.active_scalars = self.control_panel.get_active_scalars()
        self.target_cells = self.control_panel.get_target_cells()
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
        current_frame = self.control_panel.get_current_frame()
        
        # Update visualization with current frame
        self.update_visualization(current_frame)
        
        self.statusBar().showMessage("Parameters applied successfully")

    def on_create_video(self):

        """Create video from frames"""
        if not self.vtk_files:
            QMessageBox.warning(self, "Warning", "No data loaded to create video")
            return
        
        # Ask for output file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        output_file, _ = QFileDialog.getSaveFileName(self, "Save Video As", "", "MP4 Files (*.mp4);;All Files (*)", options=options)
        if not output_file:
            return
        
        try:
            # Create a temporary directory to store images
            temp_dir = "temp_video_frames"
            os.makedirs(temp_dir, exist_ok=True)
            
            image_number = 0

            # Render each frame and save as image
            for frame_index in sorted(self.vtk_files.keys()):
                self.update_visualization(frame_index)
                image_path = os.path.join(temp_dir, f"frame_{image_number:04d}.png")
                self.vtk_widget.capture_screenshot(image_path)
                print(f"Saved frame {frame_index} to {image_path}")
                image_number += 1

            # Use ffmpeg to create video from images
            ffmpeg_command = f"ffmpeg -y -framerate 10 -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_file}"
            os.system(ffmpeg_command)
            
            # Clean up temporary images
            #for file in os.listdir(temp_dir):
            #    os.remove(os.path.join(temp_dir, file))
            #os.rmdir(temp_dir)
            
            QMessageBox.information(self, "Success", f"Video saved to {output_file}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create video: {str(e)}")

    def auto_detect_scalar_min(self):
        """Auto-detect minimum value for current scalar and update the UI"""
        if not self.vtk_files or not self.active_scalars:
            print("No data loaded or no active scalars selected")
            return
            
        try:
            # Get current frame
            current_frame = self.control_panel.get_current_frame()
            if current_frame not in self.vtk_files:
                print(f"Frame {current_frame} not found")
                return
            
            # Load the current mesh
            file_path = os.path.join(self.data_location, self.vtk_files[current_frame])
            mesh = pv.read(file_path)
            
            # Handle cell data vs point data for selected scalars
            scalar_name = self.active_scalars
            if "(cell)" in scalar_name:
                scalar_name = scalar_name.replace(" (cell)", "")
                if scalar_name in mesh.cell_data:
                    mesh = mesh.cell_data_to_point_data()
            
            # Get the scalar data
            if scalar_name in mesh.point_data:
                data = mesh.point_data[scalar_name]
                auto_min = float(np.nanmin(data))
                
                # Update the control panel spinbox
                self.control_panel.data_min_spinbox.setValue(auto_min)
                print(f"Auto-detected minimum for {self.active_scalars}: {auto_min:.3f}")
                
                # Trigger full update to recreate color function and scalar bar
                self.apply_parameter_changes()
            else:
                print(f"Scalar '{scalar_name}' not found in mesh data")
                    
        except Exception as e:
            print(f"Error auto-detecting minimum: {e}")

    def auto_detect_scalar_max(self):
        """Auto-detect maximum value for current scalar and update the UI"""
        if not self.vtk_files or not self.active_scalars:
            print("No data loaded or no active scalars selected")
            return
            
        try:
            # Get current frame
            current_frame = self.control_panel.get_current_frame()
            if current_frame not in self.vtk_files:
                print(f"Frame {current_frame} not found")
                return
            
            # Load the current mesh
            file_path = os.path.join(self.data_location, self.vtk_files[current_frame])
            mesh = pv.read(file_path)
            
            # Handle cell data vs point data for selected scalars
            scalar_name = self.active_scalars
            if "(cell)" in scalar_name:
                scalar_name = scalar_name.replace(" (cell)", "")
                if scalar_name in mesh.cell_data:
                    mesh = mesh.cell_data_to_point_data()
            
            # Get the scalar data
            if scalar_name in mesh.point_data:
                data = mesh.point_data[scalar_name]
                auto_max = float(np.nanmax(data))
                
                # Update the control panel spinbox
                self.control_panel.data_max_spinbox.setValue(auto_max)
                print(f"Auto-detected maximum for {self.active_scalars}: {auto_max:.3f}")
                
                # Trigger full update to recreate color function and scalar bar
                self.apply_parameter_changes()
            else:
                print(f"Scalar '{scalar_name}' not found in mesh data")
                    
        except Exception as e:
            print(f"Error auto-detecting maximum: {e}")

    def auto_detect_scalar_range(self):
        """Auto-detect both min and max values for current scalar and update the UI"""
        if not self.vtk_files or not self.active_scalars:
            print("No data loaded or no active scalars selected")
            return
            
        try:
            # Get current frame
            current_frame = self.control_panel.get_current_frame()
            if current_frame not in self.vtk_files:
                print(f"Frame {current_frame} not found")
                return
            
            # Load the current mesh
            file_path = os.path.join(self.data_location, self.vtk_files[current_frame])
            mesh = pv.read(file_path)
            
            # Handle cell data vs point data for selected scalars
            scalar_name = self.active_scalars
            if "(cell)" in scalar_name:
                scalar_name = scalar_name.replace(" (cell)", "")
                if scalar_name in mesh.cell_data:
                    mesh = mesh.cell_data_to_point_data()
            
            # Get the scalar data
            if scalar_name in mesh.point_data:
                data = mesh.point_data[scalar_name]
                auto_min = float(np.nanmin(data))
                auto_max = float(np.nanmax(data))
                
                # Update the control panel spinboxes
                self.control_panel.data_min_spinbox.setValue(auto_min)
                self.control_panel.data_max_spinbox.setValue(auto_max)
                print(f"Auto-detected range for {self.active_scalars}: [{auto_min:.3f}, {auto_max:.3f}]")
                
                # Trigger full update to recreate color function and scalar bar
                self.apply_parameter_changes()
            else:
                print(f"Scalar '{scalar_name}' not found in mesh data")
                    
        except Exception as e:
            print(f"Error auto-detecting range: {e}")


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