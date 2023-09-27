from os.path import dirname, abspath
from importlib.metadata import metadata

from sigmaepsilon.core.config import namespace_package_name

from .space import PointCloud, CartesianFrame
from .data.polydata import PolyData, PointData
from .data.linedata import LineData
from .data.linedata import LineData as PolyData1d
from .utils import k_nearest_neighbours as KNN
from .topoarray import TopologyArray
from .trimesh import TriMesh
from .tetmesh import TetMesh
from .triang import triangulate
from .grid import grid, Grid
from .tetrahedralize import tetrahedralize
from .cellapproximator import LagrangianCellApproximator

__all__ = [
    "PointCloud",
    "CartesianFrame",
    "PolyData",
    "LineData",
    "PolyData1d",
    "PointData",
    "KNN",
    "TopologyArray",
    "TriMesh",
    "TetMesh",
    "triangulate",
    "grid",
    "Grid",
    "tetrahedralize",
    "LagrangianCellApproximator",
]

__pkg_name__ = namespace_package_name(dirname(abspath(__file__)), 10)
__pkg_metadata__ = metadata(__pkg_name__)
__version__ = __pkg_metadata__["version"]
__description__ = __pkg_metadata__["summary"]
del __pkg_metadata__
