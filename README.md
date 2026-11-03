# dam-viz
Source for dam visualization tool

## Running

```
pip install -r requirements.txt
python qt_dam_visualizer.py [data folder]
```

Video export needs `ffmpeg` on the `PATH`.

## Building a Windows executable

```
pip install -r requirements-dev.txt
pyinstaller dam_viz.spec
```

The application is written to `dist/DamViz/`; distribute the whole folder
(e.g. as a zip) and start `DamViz.exe`. Build with `DAMVIZ_CONSOLE=1` set to get
a console window showing the log output, which helps when debugging the build.

When building from a conda environment, avoid conda's scipy: its BLAS/LAPACK
libraries load Intel MKL at runtime, which PyInstaller can't detect. The
application doesn't need scipy, and `dam_viz.spec` excludes it.
