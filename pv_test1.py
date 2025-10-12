import os
import sys

import pyvista as pv

def print_mesh_info(mesh):
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

def calculate_global_range(vtk_files, data_location):
    global_min = float('inf')
    global_max = float('-inf')

    for key, filename in vtk_files.items():
        file_path = os.path.join(data_location, filename)
        mesh = pv.read(file_path)
        mesh.set_active_scalars("Resistivity(log10)")
        
        resistivity_data = mesh["Resistivity(log10)"]
        current_min = resistivity_data.min()
        current_max = resistivity_data.max()
        
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
        
        print(f"File {filename}: min={current_min:.3f}, max={current_max:.3f}")

    return global_min, global_max

def create_video(data_location, vtk_files, global_min, global_max, opacity):
    # Set up plotter for animation
    pv.set_plot_theme('dark')
    plotter = pv.Plotter()
    plotter.open_movie('resistivity_animation.mp4')

    # Create animation frames
    for key, filename in vtk_files.items():
        print(f"Processing frame for {filename}...")
        
        # Load mesh for current time step
        file_path = os.path.join(data_location, filename)
        mesh = pv.read(file_path)
        mesh.set_active_scalars("Resistivity(log10)")
        
        # Clip mesh to region of interest
        clipped = mesh.clip_box(bounds=(2, 17, 2, 22, 22, 27), invert=False)
        
        # Clear previous actors
        plotter.clear()
        
        # Add volume with consistent color range
        plotter.add_volume(clipped, cmap='RdYlBu_r', opacity=opacity, shade=True, 
                          clim=[global_min, global_max])
        plotter.show_bounds(location='outer', all_edges=True)
        
        # Add title showing current time step
        plotter.add_text(f"Time Step: {key}", position='upper_left', font_size=16)
        
        # Write frame
        plotter.write_frame()

    # Close the movie and plotter
    plotter.close()

def plot_interactive_frame(vtk_file, data_location, global_min, global_max, opacity):
    file_path = os.path.join(data_location, vtk_file)
    mesh = pv.read(file_path)
    mesh.set_active_scalars("Resistivity(log10)")
    
    clipped = mesh.clip_box(bounds=(2, 17, 2, 22, 22, 27), invert=False)
    
    p = pv.Plotter()
    p.add_volume(clipped, cmap='RdYlBu_r', opacity=opacity, shade=True,
                 clim=[global_min, global_max])
    p.show_bounds(location='outer', all_edges=True)
    p.add_text(f"Interactive View: {vtk_file}", position='upper_left', font_size=16)
    p.show()

data_location = "/home/bmjl/lu2023-17-17/Inversion_RealData/Results"
vtk_filename = "dcinv.result_201.vtk"
vtk_full_filename = os.path.join(data_location, vtk_filename) 

os.listdir(data_location)

vtk_files = {}

for file in os.listdir(data_location):
    if file.startswith("dcinv") and file.endswith(".vtk"):
        number = int(file.split('_')[-1].split('.')[0])
        print(f"Found file: {file} with number: {number}")
        vtk_files[number] = file

sorted_keys = sorted(vtk_files.keys())
vtk_files = {key: vtk_files[key] for key in sorted_keys}

for key, value in vtk_files.items():
    print(f"Number: {key}, File: {value}")

# First pass: determine global min/max for consistent color scale

#global_min, global_max = calculate_global_range(vtk_files, data_location)
global_min=-0.189
global_max=4.970
print(f"Global min: {global_min}, Global max: {global_max}")

# Set up plotter for animation
pv.set_plot_theme('dark')

opacity_alt1 = [0.8, 0.6, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
opacity_alt2 = [0.3, 0.2, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
opacity_alt3 = [1.0, 0.8, 0.1, 0.0, 0.1, 0.3, 0.5, 0.8, 1.0]

opacity_sharp_core = [0.1, 0.2, 1.0, 0.9, 0.2, 0.1, 0.0, 0.0, 0.0]
opacity_core_context = [0.4, 0.5, 1.0, 1.0, 0.5, 0.3, 0.1, 0.1, 0.0]
opacity_smooth_core = [0.3, 0.6, 1.0, 0.8, 0.4, 0.2, 0.1, 0.0, 0.0]

#plot_interactive_frame(vtk_files[0], data_location, global_min, global_max, opacity_smooth_core)
create_video(data_location, vtk_files, global_min, global_max, opacity_sharp_core)
