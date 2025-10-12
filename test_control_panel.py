#!/usr/bin/env python3
"""Simple test script for the Qt control panel without VTK dependencies"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import pyqtSlot

# Import just the control panel
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the damvis_utils module to avoid import errors
class MockDamvisUtils:
    @staticmethod
    def resample_to_uniform_grid(mesh, target_cells=500_000):
        return mesh

sys.modules['damvis_utils'] = MockDamvisUtils()

# Mock VTK and PyVista to avoid import errors
class MockVTK:
    class vtkRenderer:
        def SetBackground(self, r, g, b): pass
        def GetActiveCamera(self): return MockVTK.vtkCamera()
        def AddVolume(self, actor): pass
        def AddActor(self, actor): pass
        def RemoveAllViewProps(self): pass
        def ResetCamera(self): pass
    
    class vtkCamera:
        def SetPosition(self, x, y, z): pass
        def SetFocalPoint(self, x, y, z): pass
        def SetViewUp(self, x, y, z): pass
    
    class QVTKRenderWindowInteractor:
        def __init__(self, parent): 
            self.parent = parent
        def GetRenderWindow(self): return MockVTK.vtkRenderWindow()
    
    class vtkRenderWindow:
        def AddRenderer(self, renderer): pass
        def GetInteractor(self): return MockVTK.vtkInteractor()
        def Render(self): pass
    
    class vtkInteractor:
        def Initialize(self): pass
        def Start(self): pass

sys.modules['vtk'] = MockVTK()
sys.modules['vtk.qt'] = MockVTK()
sys.modules['vtk.qt.QVTKRenderWindowInteractor'] = MockVTK()
sys.modules['pyvista'] = MockVTK()

# Now import the control panel
from qt_dam_visualizer import ControlPanel

class TestControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Control Panel Test")
        self.setGeometry(100, 100, 400, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Add info label
        info_label = QLabel("Test Control Panel\nAdjust parameters and click Apply")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Add control panel
        self.control_panel = ControlPanel()
        layout.addWidget(self.control_panel)
        
        # Connect signals for testing
        self.connect_test_signals()
        
        central_widget.setLayout(layout)
    
    def connect_test_signals(self):
        """Connect control panel signals for testing"""
        self.control_panel.apply_changes.connect(self.on_apply_changes)
    
    @pyqtSlot()
    def on_apply_changes(self):
        """Handle apply button click"""
        frame = self.control_panel.get_current_frame()
        opacity = self.control_panel.get_opacity_values()
        bounds = self.control_panel.get_bounds_values()
        colormap = self.control_panel.get_colormap()
        show_bounds = self.control_panel.is_show_bounds_enabled()
        show_colorbar = self.control_panel.is_show_colorbar_enabled()
        
        print("Apply Changes clicked!")
        print(f"Frame: {frame}")
        print(f"Opacity values: {opacity}")
        print(f"Bounds: {bounds}")
        print(f"Colormap: {colormap}")
        print(f"Show bounds: {show_bounds}")
        print(f"Show color bar: {show_colorbar}")
        print(f"Dirty state before apply: {self.control_panel.is_dirty()}")
        
        # The control panel will automatically clear the dirty flag

def main():
    app = QApplication(sys.argv)
    
    # Apply dark theme
    app.setStyle('Fusion')
    
    window = TestControlPanel()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()