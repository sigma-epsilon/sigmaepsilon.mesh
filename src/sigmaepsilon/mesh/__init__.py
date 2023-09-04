from os.path import dirname, abspath

from sigmaepsilon.core.config import find_pyproject_toml, load_pyproject_config

from .space import PointCloud
from .space import CartesianFrame
from .polydata import PolyData
from .linedata import LineData
from .linedata import LineData as PolyData1d
from .core.pointdata import PointData
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

pyproject_toml_path = find_pyproject_toml(dirname(abspath(__file__)), 10)
project_config = load_pyproject_config(filepath=pyproject_toml_path, section="project")

__pkg_name__ = project_config["name"]
__version__ = project_config["version"]
__description__ = project_config["description"]
