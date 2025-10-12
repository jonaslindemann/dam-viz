#!/usr/bin/env python3
"""Test script for VTK Cube Axes (ParaView-style grid)"""

import sys
import vtk
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt

try:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    print("VTK Qt integration not available")
    sys.exit(1)

class VTKCubeAxesTest(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VTK Cube Axes Grid Test (ParaView Style)")
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
        self.toggle_grid_btn = QPushButton("Toggle Axes Grid")
        self.toggle_scalar_bar_btn = QPushButton("Toggle Scalar Bar")
        self.add_sample_data_btn = QPushButton("Add Sample Data")
        
        button_layout.addWidget(self.toggle_grid_btn)
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
        self.cube_axes_actor = None
        self.scalar_bar_actor = None
        self.sample_actor = None
        self.color_function = None
        
        # Connect buttons
        self.toggle_grid_btn.clicked.connect(self.toggle_axes_grid)
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
        self.add_cube_axes_grid()
        self.add_scalar_bar()
        self.renderer.ResetCamera()
        self.render_window.Render()
    
    def add_cube_axes_grid(self):
        """Add ParaView-style cube axes grid"""
        if self.cube_axes_actor:
            self.renderer.RemoveActor(self.cube_axes_actor)
        
        # Define bounds for the grid
        bounds = [-2, 2, -2, 2, -2, 2]
        
        # Create cube axes actor
        cube_axes = vtk.vtkCubeAxesActor()
        cube_axes.SetBounds(bounds)
        cube_axes.SetCamera(self.renderer.GetActiveCamera())
        
        # Set axes properties
        cube_axes.GetTitleTextProperty(0).SetColor(1.0, 1.0, 1.0)  # X axis title
        cube_axes.GetTitleTextProperty(1).SetColor(1.0, 1.0, 1.0)  # Y axis title  
        cube_axes.GetTitleTextProperty(2).SetColor(1.0, 1.0, 1.0)  # Z axis title
        
        cube_axes.GetLabelTextProperty(0).SetColor(0.8, 0.8, 0.8)  # X axis labels
        cube_axes.GetLabelTextProperty(1).SetColor(0.8, 0.8, 0.8)  # Y axis labels
        cube_axes.GetLabelTextProperty(2).SetColor(0.8, 0.8, 0.8)  # Z axis labels
        
        # Set font sizes
        for i in range(3):
            cube_axes.GetTitleTextProperty(i).SetFontSize(10)
            cube_axes.GetLabelTextProperty(i).SetFontSize(8)
        
        # Set axis titles
        cube_axes.SetXTitle("X Coordinate")
        cube_axes.SetYTitle("Y Coordinate") 
        cube_axes.SetZTitle("Z Coordinate")
        
        # Configure tick marks and grid
        cube_axes.SetTickLocationToBoth()
        cube_axes.SetFlyModeToOuterEdges()
        cube_axes.SetNumberOfLabels(5)
        
        # Grid lines properties
        cube_axes.SetGridLineLocation(vtk.vtkCubeAxesActor.VTK_GRID_LINES_ALL)
        cube_axes.GetXAxesGridlinesProperty().SetColor(0.3, 0.3, 0.3)
        cube_axes.GetYAxesGridlinesProperty().SetColor(0.3, 0.3, 0.3)
        cube_axes.GetZAxesGridlinesProperty().SetColor(0.3, 0.3, 0.3)
        
        # Main axes lines properties  
        cube_axes.GetXAxesLinesProperty().SetColor(0.8, 0.8, 0.8)
        cube_axes.GetYAxesLinesProperty().SetColor(0.8, 0.8, 0.8)
        cube_axes.GetZAxesLinesProperty().SetColor(0.8, 0.8, 0.8)
        
        # Enable grid lines
        cube_axes.SetDrawXGridlines(True)
        cube_axes.SetDrawYGridlines(True) 
        cube_axes.SetDrawZGridlines(True)
        
        self.cube_axes_actor = cube_axes
        self.renderer.AddActor(self.cube_axes_actor)
        self.render_window.Render()
    
    def add_scalar_bar(self):
        """Add compact scalar bar"""
        if self.scalar_bar_actor:
            self.renderer.RemoveActor(self.scalar_bar_actor)
        
        if self.color_function:
            # Create scalar bar
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(self.color_function)
            scalar_bar.SetTitle("Sample Values")
            scalar_bar.SetNumberOfLabels(4)
            
            # Position and size - compact
            scalar_bar.SetPosition(0.92, 0.15)
            scalar_bar.SetWidth(0.06)
            scalar_bar.SetHeight(0.4)
            
            # Style - small fonts
            scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
            scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
            scalar_bar.GetTitleTextProperty().SetFontSize(8)
            scalar_bar.GetLabelTextProperty().SetFontSize(6)
            
            self.scalar_bar_actor = scalar_bar
            self.renderer.AddActor2D(self.scalar_bar_actor)
            self.render_window.Render()
    
    def add_sample_data(self):
        """Add sample 3D data"""
        if self.sample_actor:
            self.renderer.RemoveActor(self.sample_actor)
        
        # Create a simple sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(1.0)
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(20)
        
        # Add scalar data
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
        
        print("Sample sphere data added with ParaView-style axes grid")
    
    def toggle_axes_grid(self):
        """Toggle axes grid visibility"""
        if self.cube_axes_actor:
            visible = self.cube_axes_actor.GetVisibility()
            self.cube_axes_actor.SetVisibility(not visible)
            self.render_window.Render()
            print(f"Axes grid {'hidden' if visible else 'shown'}")
    
    def toggle_scalar_bar(self):
        """Toggle scalar bar visibility"""
        if self.scalar_bar_actor:
            visible = self.scalar_bar_actor.GetVisibility()
            self.scalar_bar_actor.SetVisibility(not visible)
            self.render_window.Render()
            print(f"Scalar bar {'hidden' if visible else 'shown'}")

def main():
    app = QApplication(sys.argv)
    
    window = VTKCubeAxesTest()
    window.show()
    
    print("VTK Cube Axes Grid Test (ParaView Style)")
    print("- Click 'Toggle Axes Grid' to show/hide ParaView-style coordinate grid")
    print("- Click 'Toggle Scalar Bar' to show/hide compact color scale")
    print("- Rotate, zoom, and pan to see how the grid adapts to camera position")
    print("- Notice tick marks, grid lines, and coordinate labels")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()