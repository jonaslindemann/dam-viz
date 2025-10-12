#!/usr/bin/env python3
"""Simple test script to verify volume rendering works"""

import sys
import os
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyvista as pv
import damvis_utils as dvu

class SimpleVolumeTest(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Simple Volume Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        layout.addWidget(self.vtk_widget)
        central_widget.setLayout(layout)
        
        # VTK setup
        self.renderer = vtk.vtkRenderer()
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        self.renderer.SetBackground(0.1, 0.1, 0.2)
        
        # Load and display a test volume
        self.load_test_volume()
        
        self.interactor.Initialize()
        self.interactor.Start()
    
    def load_test_volume(self):
        """Load a test volume"""
        # Check if test data exists
        data_location = "/home/bmjl/lu2023-17-17/Inversion_RealData/Results"
        if not os.path.exists(data_location):
            print("Test data location not found")
            return
        
        # Find first VTK file
        vtk_file = None
        for file in os.listdir(data_location):
            if file.startswith("dcinv") and file.endswith(".vtk"):
                vtk_file = os.path.join(data_location, file)
                break
        
        if not vtk_file:
            print("No VTK files found")
            return
        
        print(f"Loading test file: {vtk_file}")
        
        try:
            # Load mesh
            mesh = pv.read(vtk_file)
            mesh.set_active_scalars("Resistivity(log10)")
            
            print(f"Mesh bounds: {mesh.bounds}")
            print(f"Mesh n_points: {mesh.n_points}")
            print(f"Mesh n_cells: {mesh.n_cells}")
            
            # Clip mesh
            bounds = [2, 17, 2, 22, 22, 27]
            clipped = mesh.clip_box(bounds=bounds, invert=False)
            
            # Resample
            resampled = dvu.resample_to_uniform_grid(clipped, target_cells=100_000)
            resampled.set_active_scalars('Resistivity(log10)')
            
            print(f"Resampled bounds: {resampled.bounds}")
            print(f"Resampled type: {type(resampled)}")
            
            # Create simple volume
            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputData(resampled)
            
            # Simple volume property
            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetInterpolationTypeToLinear()
            
            # Simple color function
            color_func = vtk.vtkColorTransferFunction()
            color_func.AddRGBPoint(-0.189, 0.0, 0.0, 1.0)  # Blue
            color_func.AddRGBPoint(2.5, 1.0, 1.0, 0.0)     # Yellow
            color_func.AddRGBPoint(4.970, 1.0, 0.0, 0.0)   # Red
            volume_property.SetColor(color_func)
            
            # Simple opacity function
            opacity_func = vtk.vtkPiecewiseFunction()
            opacity_func.AddPoint(-0.189, 0.0)
            opacity_func.AddPoint(0.0, 0.1)
            opacity_func.AddPoint(2.5, 1.0)
            opacity_func.AddPoint(4.970, 0.5)
            volume_property.SetScalarOpacity(opacity_func)
            
            # Create volume
            volume = vtk.vtkVolume()
            volume.SetMapper(mapper)
            volume.SetProperty(volume_property)
            
            # Add to renderer
            self.renderer.AddVolume(volume)
            
            # Reset camera
            self.renderer.ResetCamera()
            self.render_window.Render()
            
            print("Volume added successfully")
            
        except Exception as e:
            print(f"Error loading test volume: {e}")
            import traceback
            traceback.print_exc()

def main():
    app = QApplication(sys.argv)
    window = SimpleVolumeTest()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()