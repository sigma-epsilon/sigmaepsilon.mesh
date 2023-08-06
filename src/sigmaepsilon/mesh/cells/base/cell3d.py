from typing import Union, MutableMapping, Iterable, Tuple, Any

import numpy as np
from numpy import ndarray

from sigmaepsilon.math import atleast2d

from .cell import PolyCell
from ...utils.utils import (
    points_of_cells,
    cells_coords,
)
from ...utils.cells.utils import (
    _find_first_hits_,
    _find_first_hits_knn_,
    _ntet_to_loc_bulk_,
)
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
from ...utils import cell_centers_bulk
from ...utils.knn import k_nearest_neighbours
from ...config import __haspyvista__

if __haspyvista__:
    import pyvista as pv

MapLike = Union[ndarray, MutableMapping]


class PolyCell3d(PolyCell):
    """Base class for 3d cells"""

    NDIM = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def measures(self, *args, **kwargs) -> ndarray:
        """
        Returns the measures of the block.
        """
        return self.volumes(*args, **kwargs)

    @classmethod
    def tetmap(cls) -> Iterable:
        """
        Returns a mapper to transform topology and other data to
        a collection of T3 triangles.
        """
        raise NotImplementedError

    def to_tetrahedra(self, flatten: bool = True) -> ndarray:
        """
        Returns the topology as a collection of TET4 tetrahedra.

        Parameters
        ----------
        flatten: bool, Optional
            If True, the topology is returned as a 2d array. If False, the
            length of the first axis equals the number of cells in the block,
            the length of the second axis equals the number of tetrahedra per
            cell.
        """
        t = self.topology().to_numpy()
        tetmap = self.tetmap()
        tetra = transform_topology(t, tetmap)
        if flatten:
            return tetra
        else:
            nE = len(t)
            nT = len(tetmap)
            return tetra.reshape(nE, nT, 4)

    def to_vtk(self, detach: bool = False) -> Any:
        """
        Returns the block as a VTK object.
        """
        coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        vtkid = self.__class__.vtkCellType
        if detach:
            ugrid = mesh_to_vtk(*detach_mesh_bulk(coords, topo), vtkid)
        else:
            ugrid = mesh_to_vtk(coords, topo, vtkid)
        return ugrid

    if __haspyvista__:

        def to_pv(
            self, detach: bool = False
        ) -> Union[pv.UnstructuredGrid, pv.PolyData]:
            """
            Returns the block as a pyVista object.
            """
            return pv.wrap(self.to_vtk(detach=detach))

    def extract_surface(self, detach: bool = False) -> Tuple[ndarray]:
        """
        Extracts the surface of the object.
        """
        coords = self.source_coords()
        pvs = self.to_pv(detach=False).extract_surface()
        s = pvs.triangulate().cast_to_unstructured_grid()
        topo = s.cells_dict[5]
        if detach:
            return s.points, topo
        else:
            coords = self.source_coords()
            imap = index_of_closest_point(coords, np.array(s.points, dtype=float))
            topo = rewire(topo, imap)
            return coords, topo

    def boundary(self, detach: bool = False) -> Tuple[ndarray]:
        """
        Returns the boundary of the block as 2 NumPy arrays.
        """
        return self.extract_surface(detach=detach)

    def volumes(self) -> ndarray:
        """
        Returns the volumes of the block as an 1d float array.
        """
        coords = self.source_coords()
        topo = self.topology().to_numpy()
        topo_tet = self.to_tetrahedra()
        volumes = vol_tet_bulk(cells_coords(coords, topo_tet))
        res = np.sum(
            volumes.reshape(topo.shape[0], int(len(volumes) / topo.shape[0])), axis=1
        )
        return res

    def locate(
        self,
        x: Union[Iterable, ndarray],
        lazy: bool = True,
        tol: float = 1e-12,
        k: int = 4,
    ) -> Tuple[ndarray]:
        """
        Locates a set of points inside the cells of the block.

        Parameters
        ----------
        x: Iterable or numpy.ndarray
            The coordinates of the points that we want to investigate.
        tol: float, Optional
            Floating point tolerance for detecting boundaries. Default is 1e-12.
        lazy: bool, Optional
            If False, the ckeck is performed for all cells in the block. If True,
            it is used in combination with parameter 'k' and the check is only performed
            for the k nearest neighbours of the input points. Default is True.
        k: int, Optional
            The number of neighbours for the case when 'lazy' is true. Default is 4.

        Returns
        -------
        numpy.ndarray
            The indices of 'x' that are inside one of the cells of the block.
        numpy.ndarray
            The block-local indices of the cells that include the points with
            the returned indices.
        numpy.ndarray
            The master coordinates of the located points inside the including cells.
        """
        x = atleast2d(x, front=True)

        coords = self.source_coords()
        topo = self.topology()
        topo_tet = self.to_tetrahedra(flatten=True)
        ecoords_tet = points_of_cells(coords, topo_tet, centralize=False)
        tetmap = self.tetmap()

        # perform point-in-polygon test for tetrahedra
        if lazy:
            centers_tet = cell_centers_bulk(coords, topo_tet)
            k_tet = min(k, len(centers_tet))
            neighbours_tet = k_nearest_neighbours(centers_tet, x, k=k_tet)
            nat_tet = _glob_to_nat_tet_bulk_knn_(
                x, ecoords_tet, neighbours_tet
            )  # (nP, kTET, 4)
            pips_tet = __pip_tet_bulk__(nat_tet, tol)  # (nP, kTET)
        else:
            nat_tet = _glob_to_nat_tet_bulk_(x, ecoords_tet)  # (nP, nTET, 4)
            pips_tet = __pip_tet_bulk__(nat_tet, tol)  # (nP, nTET)

        # locate the points that are inside any of the cells
        pip = np.squeeze(np.any(pips_tet, axis=1))  # (nP)
        i_source = np.where(pip)[0]  # (nP_)
        if lazy:
            points_to_tets, points_to_neighbours = _find_first_hits_knn_(
                pips_tet[i_source], neighbours_tet[i_source]
            )
        else:
            points_to_tets, points_to_neighbours = _find_first_hits_(pips_tet[i_source])
        tets_to_cells = np.floor(np.arange(len(topo_tet)) / len(tetmap)).astype(int)
        i_target = tets_to_cells[points_to_tets]  # (nP_)

        # locate the cells that contain the points
        cell_tet_indices = np.tile(np.arange(tetmap.shape[0]), len(topo))[
            points_to_tets
        ]
        nat_tet = nat_tet[i_source]  # (nP_, nTET, 4)
        locations_target = _ntet_to_loc_bulk_(
            self.lcoords(), nat_tet, tetmap, cell_tet_indices, points_to_neighbours
        )

        return i_source, i_target, locations_target

    def pip(
        self,
        x: Union[Iterable, ndarray],
        tol: float = 1e-12,
        lazy: bool = True,
        k: int = 4,
    ) -> Union[bool, ndarray]:
        """
        Returns an 1d boolean integer array that tells if the points specified by 'x'
        are included in any of the cells in the block.

        Parameters
        ----------
        x: Iterable or numpy.ndarray
            The coordinates of the points that we want to investigate.
        tol: float, Optional
            Floating point tolerance for detecting boundaries. Default is 1e-12.
        lazy: bool, Optional
            If False, the ckeck is performed for all cells in the block. If True,
            it is used in combination with parameter 'k' and the check is only performed
            for the k nearest neighbours of the input points. Default is True.
        k: int, Optional
            The number of neighbours for the case when 'lazy' is true. Default is 4.

        Returns
        -------
        bool or numpy.ndarray
            A single or NumPy array of booleans for every input point.
        """
        x = atleast2d(x, front=True)
        coords = self.source_coords()
        tetra = self.to_tetrahedra(flatten=True)
        ecoords = points_of_cells(coords, tetra, centralize=False)
        if lazy:
            centers = cell_centers_bulk(coords, tetra)
            k = min(k, len(centers))
            knn = k_nearest_neighbours(centers, x, k=k)
            pips = _pip_tet_bulk_knn_(x, ecoords, knn, tol)
        else:
            pips = _pip_tet_bulk_(x, ecoords, tol)
        return np.squeeze(np.any(pips, axis=1))