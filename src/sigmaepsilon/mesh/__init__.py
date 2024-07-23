from os.path import dirname, abspath
from importlib.metadata import metadata

from sigmaepsilon.core.config import namespace_package_name

from .space import PointCloud, CartesianFrame
from .data import PolyData, PointData, LineData, TriMesh
from .data import LineData as PolyData1d
from .utils import k_nearest_neighbours as KNN
from .topoarray import TopologyArray
from .triang import triangulate
from .grid import grid
from .cellapproximator import LagrangianCellApproximator
from .io import from_pv, from_meshio, to_vtk, to_pv, to_k3d
from .plotting import pvplot

__all__ = [
    "PointCloud",
    "CartesianFrame",
    "PolyData",
    "LineData",
    "PolyData1d",
    "PointData",
    "TriMesh",
    #
    "TopologyArray",
    "LagrangianCellApproximator",
    #
    "KNN",
    "triangulate",
    "grid",
    #
    "from_pv",
    "from_meshio",
    "to_pv",
    "to_vtk",
    "to_k3d",
    #
    "pvplot",
]

__pkg_name__ = namespace_package_name(dirname(abspath(__file__)), 10)
__pkg_metadata__ = metadata(__pkg_name__)
__version__ = __pkg_metadata__["version"]
__description__ = __pkg_metadata__["summary"]
del __pkg_metadata__
