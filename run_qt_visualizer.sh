#!/bin/bash
# Launch script for Qt Dam Visualizer

echo "Starting Qt Dam Visualizer..."

export QT_PLUGIN_PATH=$CONDA_PREFIX/plugins:$CONDA_PREFIX/lib/qt6/plugins


# Check if required packages are installed
python3 -c "import PyQt5; import vtk; import pyvista; print('All dependencies found')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Missing dependencies. Please install:"
    echo "pip install PyQt5 vtk pyvista numpy"
    exit 1
fi

# Launch the application
vglrun python3 qt_dam_visualizer.py

echo "Application closed."