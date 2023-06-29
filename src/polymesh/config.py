import toml
import os
from os.path import dirname, abspath

try:
    import vtk

    __hasvtk__ = True
except Exception:
    __hasvtk__ = False

try:
    import pyvista as pv

    __haspyvista__ = True
except Exception:
    __haspyvista__ = False

try:
    import matplotlib as mpl

    __hasmatplotlib__ = True
except Exception:
    __hasmatplotlib__ = False

try:
    import plotly.express as px
    import plotly.graph_objects as go

    __hasplotly__ = True
except Exception:
    __hasplotly__ = False

try:
    import networkx as nx

    __hasnx__ = True
except Exception:
    __hasnx__ = False

try:
    import k3d

    __hask3d__ = True
except Exception:
    __hask3d__ = False


try:
    import tetgen

    __has_tetgen__ = True
except Exception:
    __has_tetgen__ = False


def load_pyproject_config():
    config_path = os.path.join(
        dirname(dirname(dirname(abspath(__file__)))), "pyproject.toml"
    )
    with open(config_path, "r") as f:
        config_toml = toml.load(f)
    config = config_toml.get("polymesh", {})
    return config
