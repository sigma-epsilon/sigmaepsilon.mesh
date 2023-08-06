from typing import Union, MutableMapping, Iterable, Tuple, List, Callable

import numpy as np
from numpy import ndarray
from sympy import Matrix, lambdify

from sigmaepsilon.math import atleast1d, atleast2d, ascont
from sigmaepsilon.math.utils import to_range_1d
from sigmaepsilon.math.linalg import ReferenceFrame as FrameLike

from sigmaepsilon.mesh.space import PointCloud, CartesianFrame
from .cell import PolyCell
from ...celldata import CellData
from ...utils.utils import (
    jacobian_matrix_bulk,
    jacobian_matrix_bulk_1d,
    jacobian_det_bulk_1d,
    points_of_cells,
    pcoords_to_coords,
    pcoords_to_coords_1d,
    cells_coords,
    lengths_of_lines,
    global_shape_function_derivatives,
)
from ...utils.cells.utils import (
    _loc_to_glob_bulk_,
    _find_first_hits_,
    _find_first_hits_knn_,
    _ntet_to_loc_bulk_,
)
from ...utils.tri import area_tri_bulk, _pip_tri_bulk_
from ...utils.tet import (
    vol_tet_bulk,
    _pip_tet_bulk_knn_,
    _pip_tet_bulk_,
    _glob_to_nat_tet_bulk_,
    _glob_to_nat_tet_bulk_knn_,
    __pip_tet_bulk__,
)
from ...utils.space import index_of_closest_point
from ...vtkutils import mesh_to_UnstructuredGrid as mesh_to_vtk
from ...utils.topology.topo import detach_mesh_bulk, rewire
from ...utils.topology import transform_topology
from ...utils.tri import triangulate_cell_coords
from ...utils import cell_center, cell_center_2d, cell_centers_bulk
from ...utils.knn import k_nearest_neighbours
from ...topoarray import TopologyArray
from ...space import CartesianFrame
from ...triang import triangulate
from ...config import __haspyvista__

if __haspyvista__:
    import pyvista as pv

MapLike = Union[ndarray, MutableMapping]


class PolyCell2d(PolyCell):
    """Base class for 2d cells"""

    NDIM = 2

    def area(self) -> float:
        """
        Returns the total area of the cells in the block.
        """
        return np.sum(self.areas())

    @classmethod
    def trimap(cls) -> Iterable:
        """
        Returns a mapper to transform topology and other data to
        a collection of T3 triangles.
        """
        _, t, _ = triangulate(points=cls.lcoords())
        return t

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Ought to return the local coordinates of the center of the
        master element.

        Returns
        -------
        numpy.ndarray
        """
        return cell_center_2d(cls.lcoords())

    def to_triangles(self) -> ndarray:
        """
        Returns the topology as a collection of T3 triangles.
        """
        t = self.topology().to_numpy()
        return transform_topology(t, self.trimap())

    def areas(self) -> ndarray:
        """
        Returns the areas of the cells.
        """
        nE = len(self)
        coords = self.source_coords()
        topo = self.topology().to_numpy()
        frames = self.frames
        ec = points_of_cells(coords, topo, local_axes=frames)
        trimap = self.__class__.trimap()
        ec_tri = triangulate_cell_coords(ec, trimap)
        areas_tri = area_tri_bulk(ec_tri)
        res = np.sum(areas_tri.reshape(nE, int(len(areas_tri) / nE)), axis=1)
        return res

    def volumes(self) -> ndarray:
        """
        Returns the volumes of the cells.
        """
        areas = self.areas()
        t = self.thickness()
        return areas * t

    def measures(self) -> ndarray:
        """
        Returns the areas of the cells.
        """
        return self.areas()

    def local_coordinates(self, *_, target: CartesianFrame = None) -> ndarray:
        """
        Returns the local coordinates of the cells of the block.
        """
        ec = super(PolyCell2d, self).local_coordinates(target=target)
        return ascont(ec[:, :, :2])

    def thickness(self) -> ndarray:
        """
        Returns the thicknesses of the cells. If not set, a thickness
        of 1.0 is returned for each cell.
        """
        dbkey = self._dbkey_thickness_
        if dbkey in self.fields:
            t = self.db[dbkey].to_numpy()
        else:
            t = np.ones(len(self), dtype=float)
        return t

    def pip(
        self, x: Union[Iterable, ndarray], tol: float = 1e-12
    ) -> Union[bool, ndarray]:
        """
        Returns an 1d boolean integer array that tells if the points specified by 'x'
        are included in any of the cells in the block.

        Parameters
        ----------
        x: Iterable or numpy.ndarray
            The coordinates of the points that we want to investigate.

        Returns
        -------
        bool or numpy.ndarray
            A single or NumPy array of booleans for every input point.
        """
        x = atleast2d(x, front=True)
        coords = self.source_coords()
        topo = self.to_triangles()
        ecoords = points_of_cells(coords, topo, centralize=False)
        pips = _pip_tri_bulk_(x, ecoords, tol)
        return np.squeeze(np.any(pips, axis=1))