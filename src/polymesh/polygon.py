import numpy as np

from .utils.utils import points_of_cells
from .utils.tri import area_tri_bulk
from .utils.topology import T6_to_T3, Q4_to_T3, Q9_to_Q4, T3_to_T6
from .cell import PolyCell2d


class PolyGon(PolyCell2d):
    """
    Base class for polygons.
    """


class Triangle(PolyGon):
    """
    Base class for triangles.
    """

    NNODE = 3
    vtkCellType = 5
    __label__ = "T3"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def trimap(cls) -> np.ndarray:
        return np.array([[0, 1, 2]], dtype=int)

    def to_triangles(self):
        return self.topology().to_numpy()

    @classmethod
    def from_TriMesh(cls, *args, coords=None, topo=None, **kwargs):
        from polymesh.trimesh import TriMesh

        if len(args) > 0 and isinstance(args[0], TriMesh):
            mesh = args[0]
            return mesh.coords(), mesh.topology().to_numpy()
        elif coords is not None and topo is not None:
            return coords, topo
        else:
            raise NotImplementedError

    def areas(self, *args, **kwargs):
        coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        ec = points_of_cells(coords, topo, local_axes=self.frames)
        return area_tri_bulk(ec)


class QuadraticTriangle(PolyGon):
    """Quadratic 6-noded triangle."""

    NNODE = 6
    vtkCellType = 22
    __label__ = "T6"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def trimap(cls, subdivide: bool = True):
        if subdivide:
            return np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [5, 3, 4]], dtype=int)
        else:
            return np.array([[0, 1, 2]], dtype=int)

    def to_triangles(self):
        return T6_to_T3(None, self.topology().to_numpy())[1]

    @classmethod
    def from_TriMesh(cls, *args, coords=None, topo=None, **kwargs):
        from polymesh.trimesh import TriMesh

        if len(args) > 0 and isinstance(args[0], TriMesh):
            return T3_to_T6(TriMesh.coords(), TriMesh.topology())
        elif coords is not None and topo is not None:
            return T3_to_T6(coords, topo)
        else:
            raise NotImplementedError


class Quadrilateral(PolyGon):
    """Basic 4-noded bilinear quadrilateral."""

    NNODE = 4
    vtkCellType = 9
    __label__ = "Q4"

    @classmethod
    def trimap(cls):
        return np.array([[0, 1, 2], [0, 2, 3]], dtype=int)

    def to_triangles(self):
        return Q4_to_T3(None, self.topology().to_numpy())[1]


class BiQuadraticQuadrilateral(PolyGon):
    """9-noded biquadratic quadrilateral."""

    NNODE = 9
    vtkCellType = 28
    __label__ = "Q9"

    @classmethod
    def trimap(cls):
        return np.array(
            [
                [0, 4, 8],
                [0, 8, 7],
                [4, 1, 5],
                [4, 5, 8],
                [8, 5, 2],
                [8, 2, 6],
                [7, 8, 6],
                [7, 6, 3],
            ],
            dtype=int,
        )

    def to_triangles(self):
        return Q4_to_T3(*Q9_to_Q4(None, self.topology().to_numpy()))[1]
