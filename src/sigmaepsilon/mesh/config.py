try:
    import vtk

    __hasvtk__ = True
except Exception:  # pragma: no cover
    __hasvtk__ = False

try:
    import pyvista as pv

    __haspyvista__ = True
except Exception:  # pragma: no cover
    __haspyvista__ = False

try:
    import matplotlib as mpl

    __hasmatplotlib__ = True
except Exception:  # pragma: no cover
    __hasmatplotlib__ = False

try:
    import plotly.express as px
    import plotly.graph_objects as go

    __hasplotly__ = True
except Exception:  # pragma: no cover
    __hasplotly__ = False

try:
    import networkx as nx

    __hasnx__ = True
except Exception:  # pragma: no cover
    __hasnx__ = False

try:
    import k3d

    __hask3d__ = True
except Exception:  # pragma: no cover
    __hask3d__ = False


try:
    import tetgen

    __has_tetgen__ = True
except Exception:  # pragma: no cover
    __has_tetgen__ = False
