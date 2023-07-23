import importlib.metadata

from .space import PointCloud
from .space import CartesianFrame
from .polydata import PolyData
from .linedata import LineData
from .linedata import LineData as PolyData1d
from .pointdata import PointData
from .utils import k_nearest_neighbours as KNN
from .topoarray import TopologyArray
from .trimesh import TriMesh
from .tetmesh import TetMesh
from .triang import triangulate
from .grid import grid, Grid
from .tetrahedralize import tetrahedralize
from .config import load_pyproject_config


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
]

# _config = load_pyproject_config()

__pkg_name__ = "sigmaepsilon.mesh"
__version__ = importlib.metadata.version(__pkg_name__)
__description__ = "A Python package to build, manipulate and analyze polygonal meshes."
