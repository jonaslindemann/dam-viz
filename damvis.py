#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualization of dam resistivity data from VTK files"""

import os
import sys

import vtk
print(f"VTK has OpenGL support: {vtk.vtkRenderWindow().SupportsOpenGL()}")

import pyvista as pv

import damvis_utils as dvu

def check_version_and_renderer():
    """Check PyVista version and set appropriate rendering backend"""

    print(f"PyVista version: {pv.__version__}")
    print(f"VTK version: {pv.vtk_version_info}")

    # Create a simple plotter to check GPU info
    p = pv.Plotter()
    print(f"Renderer: {p.renderer}")
    print(f"Render window: {p.render_window}")

    # Try to get GPU info
    p.show(auto_close=False)
    print(f"GPU Info: {p.render_window.ReportCapabilities()}")
    p.close()

class DamVisualization:
    """Class to handle visualization of dam resistivity data from VTK files"""

    def __init__(self, data_location):
        """Initialize with data location and default parameters"""

        self.data_location = data_location
        self.vtk_files = {}
        self.opacity = [0.1, 0.2, 1.0, 0.9, 0.2, 0.1, 0.0, 0.0, 0.0]
        self.global_min = None
        self.global_max = None
        self.bounds = [2, 17, 2, 22, 22, 27]  # Default bounds for clipping
        self.cmap = 'RdYlBu_r'  # Default colormap
        self.show_bounds = True
        self.movie_filename = 'resistivity_animation.mp4'
        self.plot_theme = 'dark'
        self.window_size = (1920, 1080)
        self.target_cells = 5_000_000  # Target number of cells for resampling

    def find_vtk_files(self):
        """Load and sort VTK files from the data location"""

        print(f"Fiding VTK files in {self.data_location}...")

        self.vtk_files = {}

        for file in os.listdir(self.data_location):
            if file.startswith("dcinv") and file.endswith(".vtk"):
                number = int(file.split('_')[-1].split('.')[0])
                self.vtk_files[number] = file

        sorted_keys = sorted(self.vtk_files.keys())
        self.vtk_files = {key: self.vtk_files[key] for key in sorted_keys}

        print(f"Found {len(self.vtk_files)} VTK files.")

    def print_mesh_info(self, mesh):
        """Print detailed information about the mesh and its data arrays"""

        # Print basic mesh info
        print("Mesh type:", type(mesh))
        print("Number of points:", mesh.n_points)
        print("Number of cells:", mesh.n_cells)
        print("Mesh bounds:", mesh.bounds)

        # Check ALL data locations
        print("\n=== POINT DATA ===")
        print("Point data keys:", list(mesh.point_data.keys()))
        print("Number of point arrays:", len(mesh.point_data))

        print("\n=== CELL DATA ===") 
        print("Cell data keys:", list(mesh.cell_data.keys()))
        print("Number of cell arrays:", len(mesh.cell_data))

        print("\n=== FIELD DATA ===")
        print("Field data keys:", list(mesh.field_data.keys()))

        print("\n=== ACTIVE ARRAYS ===")
        print("Active scalars:", mesh.active_scalars_name)
        print("Active vectors:", mesh.active_vectors_name)
        print("Active tensors:", mesh.active_tensors_name)

    def calculate_global_range(self):
        """Calculate global min and max of "Resistivity(log10)" across all VTK files"""
        global_min = float('inf')
        global_max = float('-inf')

        for key, filename in self.vtk_files.items():
            file_path = os.path.join(self.data_location, filename)
            mesh = pv.read(file_path)
            mesh.set_active_scalars("Resistivity(log10)")
            
            resistivity_data = mesh["Resistivity(log10)"]
            current_min = resistivity_data.min()
            current_max = resistivity_data.max()
            
            global_min = min(global_min, current_min)
            global_max = max(global_max, current_max)
            
            print(f"File {filename}: min={current_min:.3f}, max={current_max:.3f}")

        self.global_min = global_min
        self.global_max = global_max

    def create_video(self):
        """Create video by actually showing the window for each frame"""
        import imageio
        import time
        
        if self.global_min is None or self.global_max is None:
            raise ValueError("Global min and max must be calculated before creating video.")

        pv.set_plot_theme(self.plot_theme)
        
        frames = []
        camera_position = None
        

        for key, filename in self.vtk_files.items():
            print(f"Processing frame {key} for {filename}...")
            
            # Create COMPLETELY FRESH plotter for each frame
            plotter = pv.Plotter(window_size=self.window_size)
            
            # Load mesh
            file_path = os.path.join(self.data_location, filename)
            mesh = pv.read(file_path)
            mesh.set_active_scalars("Resistivity(log10)")
            
            # Process
            clipped = mesh.clip_box(bounds=self.bounds, invert=False)
            resampled = dvu.resample_to_uniform_grid(clipped)
            resampled.set_active_scalars('Resistivity(log10)')
            
            # Add volume
            plotter.add_volume(
                resampled, 
                cmap='RdYlBu_r', 
                opacity=self.opacity, 
                shade=True,
                clim=[self.global_min, self.global_max]
            )
            
            if self.show_bounds:
                plotter.show_bounds(location='outer', all_edges=True)
            
            plotter.add_text(f"Time Step: {key}", position='upper_left', font_size=10)
            
            # Set camera with exact parameters
            if camera_position is None:
                # First frame - set camera and save all parameters
                #plotter.camera_position = 'iso'
                #plotter.camera.zoom(1.5)
                
                # Save complete camera state
                camera_position = plotter.camera.position
                camera_focal_point = plotter.camera.focal_point
                #camera_view_up = plotter.camera.view_up
                
                print(f"Camera position: {camera_position}")
                print(f"Camera focal point: {camera_focal_point}")
                #print(f"Camera view up: {camera_view_up}")
            else:
                # Subsequent frames - set exact same camera parameters
                plotter.camera.position = camera_position
                plotter.camera.focal_point = camera_focal_point
                #plotter.camera.view_up = camera_view_up
            
            # CRITICAL: Actually show the window to force proper GPU rendering
            plotter.show(auto_close=False, interactive=False, interactive_update=False)
            
            # Give GPU time to complete rendering
            time.sleep(0.3)
            
            # Now capture screenshot
            img = plotter.screenshot(return_img=True)
            frames.append(img)
            
            # Close plotter
            plotter.close()
            del plotter

        
        # Combine frames into video
        print(f"\nCombining {len(frames)} frames into video...")
        imageio.mimsave(
            self.movie_filename,
            frames,
            fps=10,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p'
        )
        
        print(f"Video saved as {self.movie_filename}") 


    def plot_interactive(self, frame_index=0):
        """Plot an interactive frame for the specified VTK file"""

        pv.set_plot_theme(self.plot_theme)

        file_path = os.path.join(self.data_location, self.vtk_files[frame_index])
        mesh = pv.read(file_path)
        mesh.set_active_scalars("Resistivity(log10)")
        
        clipped = mesh.clip_box(bounds=(2, 17, 2, 22, 22, 27), invert=False)

        resampled = dvu.resample_to_uniform_grid_with_cleanup(clipped, target_cells=self.target_cells)

        # Check what arrays exist
        print(f"Available point arrays: {resampled.point_data.keys()}")

        # Set the active scalars (replace 'your_array_name' with actual name)
        if len(resampled.point_data.keys()) > 0:
            resampled.set_active_scalars('Resistivity(log10)')
        else:
            print("ERROR: No point data arrays found!")        

        p = pv.Plotter(window_size=self.window_size)
        p.add_volume(resampled, cmap='RdYlBu_r', opacity=self.opacity, shade=True,
                    clim=[self.global_min, self.global_max])
        
        p.show_bounds(location='outer', all_edges=True)
        p.add_text(f"Interactive View: {self.vtk_files[frame_index]}", position='upper_left', font_size=16)
        p.show()


if __name__ == "__main__":

    #check_version_and_renderer()

    data_location = "/home/bmjl/lu2023-17-17/Inversion_RealData/Results"
    
    dam_vis = DamVisualization(data_location)
    dam_vis.find_vtk_files()
    dam_vis.global_max = 4.970
    dam_vis.global_min = -0.189

    #dam_vis.opacity = [0.1, 0.2, 1.0, 0.9, 0.2, 0.1, 0.0, 0.0, 0.0]
    #dam_vis.opacity = [0.4, 0.5, 1.0, 1.0, 0.5, 0.3, 0.1, 0.1, 0.0]
    #dam_vis.opacity = [0.3, 0.6, 1.0, 0.8, 0.4, 0.2, 0.1, 0.0, 0.0]    
    dam_vis.opacity = [0.1, 0.2, 1.0, 0.9, 0.2, 0.1, 0.0, 0.0, 0.0]

    dam_vis.show_bounds = True
    #dam_vis.calculate_global_range()
    #dam_vis.create_video()
    dam_vis.plot_interactive(frame_index=100)