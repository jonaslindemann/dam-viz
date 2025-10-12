#!/usr/bin/env python3
"""Test script for VTK axes and scalar bar functionality"""

import sys
import vtk
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt

try:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    print("VTK Qt integration not available")
    sys.exit(1)

class VTKAxesScalarBarTest(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VTK Axes and Scalar Bar Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        layout.addWidget(self.vtk_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.toggle_axes_btn = QPushButton("Toggle Axes")
        self.toggle_scalar_bar_btn = QPushButton("Toggle Scalar Bar")
        self.add_sample_data_btn = QPushButton("Add Sample Data")
        
        button_layout.addWidget(self.toggle_axes_btn)
        button_layout.addWidget(self.toggle_scalar_bar_btn)
        button_layout.addWidget(self.add_sample_data_btn)
        layout.addLayout(button_layout)
        
        central_widget.setLayout(layout)
        
        # VTK setup
        self.renderer = vtk.vtkRenderer()
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        self.renderer.SetBackground(0.2, 0.3, 0.4)
        
        # Initialize actors
        self.axes_actor = None
        self.scalar_bar_actor = None
        self.sample_actor = None
        self.color_function = None
        
        # Connect buttons
        self.toggle_axes_btn.clicked.connect(self.toggle_axes)
        self.toggle_scalar_bar_btn.clicked.connect(self.toggle_scalar_bar)
        self.add_sample_data_btn.clicked.connect(self.add_sample_data)
        
        # Initialize
        self.interactor.Initialize()
        self.interactor.Start()
        
        # Add initial content
        self.setup_initial_scene()
    
    def setup_initial_scene(self):
        """Set up initial scene with sample data"""
        self.add_sample_data()
        self.add_axes()
        self.add_scalar_bar()
        self.renderer.ResetCamera()
        self.render_window.Render()
    
    def add_axes(self):
        """Add coordinate axes"""
        if self.axes_actor:
            self.renderer.RemoveActor(self.axes_actor)
        
        # Create axes actor
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(2.0, 2.0, 2.0)
        axes.SetShaftType(0)  # Line shaft
        axes.SetTipType(0)    # Cone tip
        
        # Set axis labels
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y") 
        axes.SetZAxisLabelText("Z")
        
        self.axes_actor = axes
        self.renderer.AddActor(self.axes_actor)
        self.render_window.Render()
    
    def add_scalar_bar(self):
        """Add scalar bar"""
        if self.scalar_bar_actor:
            self.renderer.RemoveActor(self.scalar_bar_actor)
        
        if self.color_function:
            # Create scalar bar
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(self.color_function)
            scalar_bar.SetTitle("Sample Data Values")
            scalar_bar.SetNumberOfLabels(5)
            
            # Position and size
            scalar_bar.SetPosition(0.85, 0.1)
            scalar_bar.SetWidth(0.12)
            scalar_bar.SetHeight(0.8)
            
            # Style
            scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
            scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
            scalar_bar.GetTitleTextProperty().SetFontSize(12)
            scalar_bar.GetLabelTextProperty().SetFontSize(10)
            
            self.scalar_bar_actor = scalar_bar
            self.renderer.AddActor2D(self.scalar_bar_actor)
            self.render_window.Render()
    
    def add_sample_data(self):
        """Add sample 3D data"""
        if self.sample_actor:
            self.renderer.RemoveActor(self.sample_actor)
        
        # Create a simple sphere with scalar data
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(1.0)
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(20)
        
        # Add scalar data based on elevation
        elevate = vtk.vtkElevationFilter()
        elevate.SetInputConnection(sphere.GetOutputPort())
        elevate.SetLowPoint(0, -1, 0)
        elevate.SetHighPoint(0, 1, 0)
        elevate.SetScalarRange(-1, 1)
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(elevate.GetOutputPort())
        mapper.SetScalarRange(-1, 1)
        
        # Create color transfer function
        self.color_function = vtk.vtkColorTransferFunction()
        self.color_function.AddRGBPoint(-1.0, 0.0, 0.0, 1.0)  # Blue
        self.color_function.AddRGBPoint(0.0, 1.0, 1.0, 0.0)   # Yellow
        self.color_function.AddRGBPoint(1.0, 1.0, 0.0, 0.0)   # Red
        
        mapper.SetLookupTable(self.color_function)
        
        # Create actor
        self.sample_actor = vtk.vtkActor()
        self.sample_actor.SetMapper(mapper)
        
        self.renderer.AddActor(self.sample_actor)
        self.render_window.Render()
        
        print("Sample sphere data added")
    
    def toggle_axes(self):
        """Toggle axes visibility"""
        if self.axes_actor:
            visible = self.axes_actor.GetVisibility()
            self.axes_actor.SetVisibility(not visible)
            self.render_window.Render()
            print(f"Axes {'hidden' if visible else 'shown'}")
    
    def toggle_scalar_bar(self):
        """Toggle scalar bar visibility"""
        if self.scalar_bar_actor:
            visible = self.scalar_bar_actor.GetVisibility()
            self.scalar_bar_actor.SetVisibility(not visible)
            self.render_window.Render()
            print(f"Scalar bar {'hidden' if visible else 'shown'}")

def main():
    app = QApplication(sys.argv)
    
    window = VTKAxesScalarBarTest()
    window.show()
    
    print("VTK Axes and Scalar Bar Test")
    print("- Click 'Toggle Axes' to show/hide coordinate axes")
    print("- Click 'Toggle Scalar Bar' to show/hide color scale")
    print("- Click 'Add Sample Data' to refresh the sphere")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()