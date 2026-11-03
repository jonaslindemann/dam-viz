# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller configuration for the Dam Resistivity Visualizer.

Build:   pyinstaller dam_viz.spec
Output:  dist/DamViz/DamViz.exe  (copy or zip the whole dist/DamViz folder)

A one-folder build is used on purpose: frames are prepared in worker
processes, and with a one-file build every worker would first have to unpack
the whole application again.

Set DAMVIZ_CONSOLE=1 when building to get a console window with the log
output, which helps when debugging the packaged application.
"""
import os

console = os.environ.get("DAMVIZ_CONSOLE") == "1"

a = Analysis(
    ["qt_dam_visualizer.py"],
    pathex=[],
    binaries=[],
    datas=[("icons", "icons")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Only PySide6 is used (through qtpy); bundling several Qt bindings breaks the build.
    # The rest are optional dependencies of pyvista/pyqtgraph that the application
    # doesn't use and that would only add size (matplotlib and PIL can't be excluded,
    # pyvista imports them unconditionally).
    excludes=["PyQt5", "PyQt6", "PySide2", "tkinter", "IPython", "jupyter", "notebook", "pytest",
              "scipy", "pandas"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DamViz",
    icon="icons/app.ico",
    console=console,
    upx=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="DamViz",
    upx=False,
)
