#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyvista as pv
import numpy as np

def resample_to_uniform_grid(ugrid, target_cells=1_000_000):
    """
    Resample unstructured grid to uniform grid with approximate target cell count.
    """
    bounds = ugrid.bounds
    extents = [
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4]
    ]
    
    # Calculate dimensions maintaining aspect ratio
    volume = extents[0] * extents[1] * extents[2]
    cell_size = (volume / target_cells) ** (1/3)
    
    dimensions = [
        int(np.ceil(extents[0] / cell_size)) + 1,
        int(np.ceil(extents[1] / cell_size)) + 1,
        int(np.ceil(extents[2] / cell_size)) + 1
    ]
    
    # Create uniform grid
    uniform_grid = pv.ImageData(
        dimensions=dimensions,
        spacing=[
            extents[0] / (dimensions[0] - 1),
            extents[1] / (dimensions[1] - 1),
            extents[2] / (dimensions[2] - 1)
        ],
        origin=(bounds[0], bounds[2], bounds[4])
    )
    
    # Check what arrays are available
    print(f"Available arrays: {ugrid.array_names}")
    
    # Resample - this will interpolate all point data
    resampled = uniform_grid.sample(ugrid)
    
    print(f"Resampled dimensions: {dimensions}")
    print(f"Total cells: {resampled.n_cells}")
    print(f"Resampled arrays: {resampled.array_names}")
    
    return resampled

def resample_to_uniform_grid_with_cleanup(ugrid, target_cells=1_000_000):
    """Resample with artifact cleanup"""
    import numpy as np
    
    bounds = ugrid.bounds
    extents = [
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4]
    ]
    
    volume = extents[0] * extents[1] * extents[2]
    cell_size = (volume / target_cells) ** (1/3)
    
    dimensions = [
        int(np.ceil(extents[0] / cell_size)) + 1,
        int(np.ceil(extents[1] / cell_size)) + 1,
        int(np.ceil(extents[2] / cell_size)) + 1
    ]
    
    # Ensure input has point data
    if ugrid.active_scalars_name in ugrid.cell_data:
        ugrid = ugrid.cell_data_to_point_data()
    
    uniform_grid = pv.ImageData(
        dimensions=dimensions,
        spacing=[
            extents[0] / (dimensions[0] - 1),
            extents[1] / (dimensions[1] - 1),
            extents[2] / (dimensions[2] - 1)
        ],
        origin=(bounds[0], bounds[2], bounds[4])
    )
    
    # Resample
    resampled = uniform_grid.sample(ugrid)
    
    # CRITICAL: Clean up the resampled data
    if resampled.active_scalars_name:
        data = resampled[resampled.active_scalars_name].copy()
        
        # Replace NaN and Inf values
        nan_mask = np.isnan(data) | np.isinf(data)
        if nan_mask.any():
            print(f"  Replacing {nan_mask.sum()} NaN/Inf values")
            # Use a value outside your data range or minimum value
            data[nan_mask] = np.nanmin(data) - 1.0  # or use 0.0
        
        # Threshold: remove values that are clearly artifacts
        # (values significantly outside the expected range)
        valid_min = ugrid[ugrid.active_scalars_name].min()
        valid_max = ugrid[ugrid.active_scalars_name].max()
        
        # Add small buffer for interpolation
        buffer = (valid_max - valid_min) * 0.1
        artifact_mask = (data < valid_min - buffer) | (data > valid_max + buffer)
        if artifact_mask.any():
            print(f"  Removing {artifact_mask.sum()} out-of-range values")
            data[artifact_mask] = valid_min - 1.0
        
        # Update the resampled data
        resampled[resampled.active_scalars_name] = data
    
    print(f"Resampled dimensions: {dimensions}")
    print(f"Total cells: {resampled.n_cells}")
    
    return resampled