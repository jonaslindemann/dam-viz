#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import OrderedDict

import pyvista as pv
import numpy as np
from scipy.ndimage import distance_transform_edt

def warm_up():
    """No-op used to start a worker process (and its imports) ahead of time"""
    return None

def load_scalar_mesh(file_path, scalar_name):
    """Read a mesh keeping only scalar_name, as point data (as it is rendered)"""
    mesh = pv.read(file_path)

    # Keep only the selected array to reduce clipping, resampling and transfer cost
    if scalar_name in mesh.cell_data:
        values = mesh.cell_data[scalar_name]
        mesh.clear_data()
        mesh.cell_data[scalar_name] = values
        mesh = mesh.cell_data_to_point_data()
    elif scalar_name in mesh.point_data:
        values = mesh.point_data[scalar_name]
        mesh.clear_data()
        mesh.point_data[scalar_name] = values
    else:
        raise KeyError(f"Scalar '{scalar_name}' not found in {os.path.basename(file_path)}")
    mesh.set_active_scalars(scalar_name)
    return mesh

def sample_points(mesh, points, scalar_name):
    """Interpolate scalar_name at points; returns (values, valid) where valid is False outside the mesh"""
    probe = pv.PolyData(np.asarray(points, dtype=float)).sample(mesh)
    valid = np.asarray(probe.point_data['vtkValidPointMask']).astype(bool)
    return np.asarray(probe.point_data[scalar_name], dtype=float), valid

def probe_frame(file_path, scalar_name, point):
    """Value of scalar_name at a point of one frame, NaN outside the mesh (runs in a worker process)"""
    mesh = load_scalar_mesh(file_path, scalar_name)
    values, valid = sample_points(mesh, [point], scalar_name)
    return float(values[0]) if valid[0] else float('nan')

def prepare_frame(file_path, scalar_name, bounds, resample, target_cells):
    """
    Load, clip and optionally resample one frame for rendering.

    This is the expensive part of showing a frame. It is meant to run in a
    worker process (VTK holds the GIL, so a thread would still block the UI),
    so the result is returned as plain numpy arrays. Use FrameData to turn it
    back into PyVista objects.
    """
    mesh = load_scalar_mesh(file_path, scalar_name)
    clipped = mesh.clip_box(bounds=list(bounds), invert=False)

    result = {
        'scalar_name': scalar_name,
        'points': np.asarray(clipped.points),
        'cells': np.asarray(clipped.cells),
        'celltypes': np.asarray(clipped.celltypes),
        'values': np.asarray(clipped.point_data[scalar_name]),
    }

    if resample and clipped.n_cells > 0:
        grid = resample_to_uniform_grid(clipped, target_cells=target_cells)
        values = np.asarray(grid.point_data[scalar_name], dtype=np.float32)
        valid = np.asarray(grid.point_data['vtkValidPointMask']).astype(bool)
        result.update({
            'dimensions': grid.dimensions,
            'spacing': grid.spacing,
            'origin': grid.origin,
            'volume_values': fill_from_nearest_valid(values, valid, grid.dimensions),
            'volume_valid': valid,
        })

    return result

def fill_from_nearest_valid(values, valid, dimensions):
    """
    Replace values of voxels outside the mesh by the nearest valid value.

    The outside voxels are hidden with a mask when rendering, but they are
    still used when interpolating samples near the boundary. Filling them
    with nearby data keeps that interpolation smooth instead of blending the
    data with an arbitrary fill value, which shows up as a shell with
    moire-like stripes around the data.
    """
    if valid.all() or not valid.any():
        return values

    # ImageData point arrays are ordered with x varying fastest
    shape = tuple(reversed(dimensions))
    _, indices = distance_transform_edt(~valid.reshape(shape), return_indices=True)
    filled = values.reshape(shape)[tuple(indices)]
    return filled.ravel()

class FrameData:
    """Render-ready data of one frame, rebuilt from the output of prepare_frame"""

    def __init__(self, arrays):
        self.scalar_name = arrays['scalar_name']

        # Clipped mesh, used for isosurfaces and range detection
        self.mesh = pv.UnstructuredGrid(arrays['cells'], arrays['celltypes'], arrays['points'])
        self.mesh.point_data[self.scalar_name] = arrays['values']
        self.mesh.set_active_scalars(self.scalar_name)

        # Volume to render: the resampled image if available, otherwise the mesh itself.
        # For the image, mask marks the voxels inside the mesh (255) that are rendered.
        self.mask = None
        self._histogram_values = None
        if 'volume_values' in arrays:
            self.volume = pv.ImageData(dimensions=arrays['dimensions'],
                                       spacing=arrays['spacing'],
                                       origin=arrays['origin'])
            self.volume.point_data[self.scalar_name] = arrays['volume_values']
            self.volume.set_active_scalars(self.scalar_name)

            self.valid = arrays['volume_valid']
            self.mask = pv.ImageData(dimensions=arrays['dimensions'],
                                     spacing=arrays['spacing'],
                                     origin=arrays['origin'])
            self.mask.point_data['mask'] = self.valid.astype(np.uint8) * 255
            self.mask.set_active_scalars('mask')
        else:
            self.volume = self.mesh

    def histogram_values(self):
        """Scalar values as rendered: the valid voxels, or the mesh points"""
        if self._histogram_values is None:
            if self.mask is not None:
                self._histogram_values = self.volume.point_data[self.scalar_name][self.valid]
            else:
                self._histogram_values = self.mesh.point_data[self.scalar_name]
        return self._histogram_values

    def robust_range(self, low_percentile=2.0, high_percentile=98.0):
        """Return a (min, max) of the scalar that ignores extreme outliers.

        The inversion results contain a few extreme values that would otherwise
        squeeze almost all data into a narrow part of the transfer functions.
        Returns None if the clipped mesh is empty.
        """
        data = self.mesh.point_data[self.scalar_name]
        if data.size == 0:
            return None

        data_min, data_max = np.nanpercentile(data, [low_percentile, high_percentile])
        if data_max <= data_min:
            data_min, data_max = np.nanmin(data), np.nanmax(data)
        return float(data_min), float(data_max)

class FrameCache:
    """Small least-recently-used cache of FrameData objects"""

    def __init__(self, max_size):
        self.max_size = max_size
        self._items = OrderedDict()

    def get(self, key):
        item = self._items.get(key)
        if item is not None:
            self._items.move_to_end(key)
        return item

    def put(self, key, item):
        self._items[key] = item
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def items(self):
        """(key, item) pairs, without changing the LRU order"""
        return list(self._items.items())

    def clear(self):
        self._items.clear()

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