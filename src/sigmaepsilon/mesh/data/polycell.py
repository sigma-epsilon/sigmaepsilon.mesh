from typing import (
    Union,
    MutableMapping,
    Iterable,
    Tuple,
    Any,
    ClassVar,
    Optional,
    TypeVar,
    Generic,
)

import numpy as np
from numpy import ndarray

from sigmaepsilon.math import atleast1d, atleast2d, ascont
from sigmaepsilon.math.linalg import ReferenceFrame as FrameLike
from sigmaepsilon.math.utils import to_range_1d

from ..typing import ABC_PolyCell, PolyDataProtocol, PointDataProtocol, GeometryProtocol
from .celldata import CellData
from ..space import PointCloud, CartesianFrame
from ..utils.utils import (
    jacobian_matrix_bulk,
    jacobian_matrix_bulk_1d,
    jacobian_det_bulk_1d,
    points_of_cells,
    pcoords_to_coords,
    pcoords_to_coords_1d,
    cells_coords,
    lengths_of_lines,
)
from ..utils.cells.utils import (
    _loc_to_glob_bulk_,
)
from ..utils.tet import (
    vol_tet_bulk,
    _pip_tet_bulk_knn_,
    _pip_tet_bulk_,
    _glob_to_nat_tet_bulk_,
    _glob_to_nat_tet_bulk_knn_,
    __pip_tet_bulk__,
)
from ..utils.cells.utils import (
    _find_first_hits_,
    _find_first_hits_knn_,
    _ntet_to_loc_bulk_,
)
from ..utils.topology.topo import detach_mesh_bulk, rewire
from ..utils.topology import transform_topology
from ..utils.tri import triangulate_cell_coords, area_tri_bulk, _pip_tri_bulk_
from ..utils.knn import k_nearest_neighbours
from ..utils.space import index_of_closest_point, frames_of_lines, frames_of_surfaces
from ..utils import cell_centers_bulk
from ..vtkutils import mesh_to_UnstructuredGrid as mesh_to_vtk
from ..topoarray import TopologyArray
from ..space import CartesianFrame
from ..config import __haspyvista__

if __haspyvista__:
    import pyvista as pv

MapLike = Union[ndarray, MutableMapping]
PointDataLike = TypeVar("PointDataLike", bound=PointDataProtocol)
MeshDataLike = TypeVar("MeshDataLike", bound=PolyDataProtocol)


__all__ = ["PolyCell"]


class PolyCell(
    Generic[MeshDataLike, PointDataLike],
    CellData[MeshDataLike, PointDataLike],
    ABC_PolyCell,
):
    """
    A subclass of :class:`~sigmaepsilon.mesh.data.celldata.CellData` as a base class
    for all cell containers.
    """

    label: ClassVar[Optional[str]] = None
    Geometry: ClassVar[GeometryProtocol]

    @CellData.frames.getter
    def frames(self) -> ndarray:
        """Returns local coordinate frames of the cells."""
        if not self.has_frames:
            if (nD := self.Geometry.number_of_spatial_dimensions) == 1:
                coords = self.source_coords()
                topo = self.topology().to_numpy()
                self.frames = frames_of_lines(coords, topo)
            elif nD == 2:
                coords = self.source_coords()
                topo = self.topology().to_numpy()
                self.frames = frames_of_surfaces(coords, topo)
            elif nD == 3:
                self.frames = self.source_frame()
            else:  # pragma: no cover
                raise TypeError(
                    "Invalid Geometry class. The 'number of spatial dimensions'"
                    " must be 1, 2 or 3."
                )
        return super().frames

    def to_triangles(self) -> ndarray:
        """
        Returns the topology as a collection of T3 triangles.
        """
        if self.Geometry.number_of_spatial_dimensions == 2:
            t = self.topology().to_numpy()
            return transform_topology(t, self.Geometry.trimap())
        else:
            raise NotImplementedError("This is only for 2d cells")

    def to_tetrahedra(self, flatten: Optional[bool] = True) -> ndarray:
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
        if self.Geometry.number_of_spatial_dimensions == 3:
            t = self.topology().to_numpy()
            tetmap = self.Geometry.tetmap()
            tetra = transform_topology(t, tetmap)
            if flatten:
                return tetra
            else:
                nE = len(t)
                nT = len(tetmap)
                return tetra.reshape(nE, nT, 4)
        else:
            raise NotImplementedError("This is only for 3d cells")

    def to_simplices(self) -> Tuple[ndarray]:
        """
        Returns the cells of the block, refactorized into simplices.
        """
        NDIM: int = self.Geometry.number_of_spatial_dimensions
        if NDIM == 1:
            raise NotImplementedError
        elif NDIM == 2:
            return self.to_triangles()
        elif NDIM == 3:
            return self.to_tetrahedra()
        else:
            raise NotImplementedError

    def jacobian_matrix(
        self, *, pcoords: Iterable[float] = None, dshp: ndarray = None, **__
    ) -> ndarray:
        """
        Returns the jacobian matrices of the cells in the block. The evaluation
        of the matrix is governed by the inputs in the following way:
        - if `dshp` is provided, it must be a matrix of shape function derivatives
          evaluated at the desired locations
        - the desired locations are specified through `pcoords`

        Parameters
        ----------
        pcoords: Iterable[float], Optional
            Locations of the evaluation points.
        dshp: numpy.ndarray, Optional
            3d array of shape function derivatives for the master cell,
            evaluated at some points. The array must have a shape of
            (nG, nNE, nD), where nG, nNE and nD are the number of evaluation
            points, nodes per cell and spatial dimensions.

        Returns
        -------
        numpy.ndarray
            A 4d array of shape (nE, nP, nD, nD), where nE, nP and nD
            are the number of elements, evaluation points and spatial
            dimensions. The number of evaluation points in the output
            is governed by the parameter 'dshp' or 'pcoords'.
        """
        ecoords = self.local_coordinates()

        if dshp is None:
            x = (
                np.array(pcoords)
                if pcoords is not None
                else self.Geometry.master_coordinates()
            )
            dshp = self.Geometry.shape_function_derivatives(x)

        if self.Geometry.number_of_spatial_dimensions == 1:
            return jacobian_matrix_bulk_1d(dshp, ecoords)
        else:
            return jacobian_matrix_bulk(dshp, ecoords)

    def jacobian(self, *, jac: ndarray = None, **kwargs) -> Union[float, ndarray]:
        """
        Returns the jacobian determinant for one or more cells.

        Parameters
        ----------
        jac: numpy.ndarray, Optional
            One or more Jacobian matrices. Default is None.
        **kwargs: dict
            Forwarded to :func:`jacobian_matrix` if the jacobian
            is not provided by the parameter 'jac'.

        Returns
        -------
        float or numpy.ndarray
            Value of the Jacobian for one or more cells.

        See Also
        --------
        :func:`jacobian_matrix`
        """
        if jac is None:
            jac = self.jacobian_matrix(**kwargs)

        if self.Geometry.number_of_spatial_dimensions == 1:
            return jacobian_det_bulk_1d(jac)
        else:
            return np.linalg.det(jac)

    def flip(self) -> "PolyCell":
        """
        Reverse the order of nodes of the topology.
        """
        topo = self.topology().to_numpy()
        self.nodes = np.flip(topo, axis=1)
        return self

    def measures(self, *args, **kwargs) -> ndarray:
        """Ought to return measures for each cell in the database."""
        NDIM: int = self.Geometry.number_of_spatial_dimensions
        if NDIM == 1:
            return self.lengths()
        elif NDIM == 2:
            return self.areas()
        elif NDIM == 3:
            return self.volumes()
        else:
            raise NotImplementedError

    def measure(self, *args, **kwargs) -> float:
        """Ought to return the net measure for the cells in the
        database as a group."""
        return np.sum(self.measures(*args, **kwargs))

    def thickness(self) -> ndarray:
        """
        Returns the thicknesses of the cells. If not set, a thickness
        of 1.0 is returned for each cell. Only for 2d cells.
        """
        if self.Geometry.number_of_spatial_dimensions == 2:
            dbkey = self._dbkey_thickness_
            if dbkey in self.fields:
                t = self.db[dbkey].to_numpy()
            else:
                t = np.ones(len(self), dtype=float)
            return t
        else:
            raise NotImplementedError("This is only for 2d cells")

    def length(self) -> float:
        """Returns the total length of the cells in the block."""
        if self.Geometry.number_of_spatial_dimensions == 1:
            return np.sum(self.lengths())
        else:
            raise NotImplementedError("This is only for 1d cells")

    def lengths(self) -> ndarray:
        """
        Returns the lengths as a NumPy array.
        """
        if self.Geometry.number_of_spatial_dimensions == 1:
            coords = self.container.source().coords()
            topo = self.topology().to_numpy()
            return lengths_of_lines(coords, topo)
        else:
            raise NotImplementedError("This is only for 1d cells")

    def area(self, *args, **kwargs) -> float:
        """
        Returns the total area of the cells in the database. Only for 2d entities.
        """
        if self.Geometry.number_of_spatial_dimensions == 2:
            return np.sum(self.areas(*args, **kwargs))
        else:
            raise NotImplementedError("This is only for 2d cells")

    def areas(self, *args, **kwargs) -> ndarray:
        """Ought to return the areas of the individuall cells in the database."""
        NDIM: int = self.Geometry.number_of_spatial_dimensions
        if NDIM == 1:
            areakey = self._dbkey_areas_
            if areakey in self.fields:
                return self[areakey].to_numpy()
            else:
                return np.ones((len(self)))
        elif NDIM == 2:
            nE = len(self)
            coords = self.source_coords()
            topo = self.topology().to_numpy()
            frames = self.frames
            ec = points_of_cells(coords, topo, local_axes=frames)
            trimap = self.__class__.Geometry.trimap()
            ec_tri = triangulate_cell_coords(ec, trimap)
            areas_tri = area_tri_bulk(ec_tri)
            res = np.sum(areas_tri.reshape(nE, int(len(areas_tri) / nE)), axis=1)
            return res
        else:
            raise NotImplementedError("This is only for 2d cells")

    def volume(self, *args, **kwargs) -> float:
        """Returns the volume of the cells in the database."""
        return np.sum(self.volumes(*args, **kwargs))

    def volumes(self, *args, **kwargs) -> ndarray:
        """Returns the volumes of the cells in the database."""
        NDIM: int = self.Geometry.number_of_spatial_dimensions
        if NDIM == 1:
            return self.lengths() * self.areas()
        elif NDIM == 2:
            areas = self.areas()
            t = self.thickness()
            return areas * t
        elif NDIM == 3:
            coords = self.source_coords()
            topo = self.topology().to_numpy()
            topo_tet = self.to_tetrahedra()
            volumes = vol_tet_bulk(cells_coords(coords, topo_tet))
            res = np.sum(
                volumes.reshape(topo.shape[0], int(len(volumes) / topo.shape[0])),
                axis=1,
            )
            return res
        else:
            raise NotImplementedError

    def extract_surface(self, detach: bool = False):
        """Extracts the surface of the mesh. Only for 3d meshes."""
        raise NotImplementedError

    def source_points(self) -> PointCloud:
        """
        Returns the hosting pointcloud.
        """
        return self.container.source().points()

    def source_coords(self) -> ndarray:
        """
        Returns the coordinates of the hosting pointcloud.
        """
        if self.pointdata is not None:
            coords = self.pointdata.x
        else:
            coords = self.container.source().coords()
        return coords

    def source_frame(self) -> FrameLike:
        """
        Returns the frame of the hosting pointcloud.
        """
        return self.container.source().frame

    def points_of_cells(
        self,
        *,
        points: Union[float, Iterable] = None,
        cells: Union[int, Iterable] = None,
        target: Union[str, CartesianFrame] = "global",
        rng: Iterable = None,
    ) -> ndarray:
        """
        Returns the points of selected cells as a NumPy array.
        """
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            assert len(cells) > 0, "Length of cells is zero!"
        else:
            cells = np.s_[:]

        if isinstance(target, str):
            assert target.lower() in ["global", "g"]
        else:
            raise NotImplementedError

        NDIM: int = self.Geometry.number_of_spatial_dimensions
        coords = self.source_coords()
        topo = self.topology().to_numpy()[cells]
        ecoords = points_of_cells(coords, topo, centralize=False)

        if points is None:
            return ecoords
        else:
            if NDIM == 1:
                rng = np.array([-1, 1]) if rng is None else np.array(rng)
                points = atleast1d(np.array(points))
                points = to_range_1d(points, source=rng, target=[0, 1])
            else:
                points = np.array(points)

        if NDIM == 1:
            res = pcoords_to_coords_1d(points, ecoords)  # (nE * nP, nD)
            nE = ecoords.shape[0]
            nP = points.shape[0]
            res = res.reshape(nE, nP, res.shape[-1])  # (nE, nP, nD)
            return res
        else:
            shp = self.Geometry.shape_function_values(points)
            if len(shp) == 3:  # variable metric cells
                shp = shp if len(shp) == 2 else shp[cells]
            return pcoords_to_coords(points, ecoords, shp)  # (nE, nP, nD)

    def local_coordinates(self, *, target: CartesianFrame = None) -> ndarray:
        """
        Returns local coordinates of the cells as a 3d float
        numpy array.

        Parameters
        ----------
        target: CartesianFrame, Optional
            A target frame. If provided, coordinates are returned in
            this frame, otherwise they are returned in the local frames
            of the cells. Default is None.
        """
        if isinstance(target, CartesianFrame):
            frames = target.show()
        else:
            frames = self.frames
        topo = self.topology().to_numpy()
        if self.pointdata is not None:
            coords = self.pointdata.x
        else:
            coords = self.container.source().coords()
        res = points_of_cells(coords, topo, local_axes=frames, centralize=True)

        if self.Geometry.number_of_spatial_dimensions == 2:
            return ascont(res[:, :, :2])
        else:
            return res

    def coords(self, *args, **kwargs) -> ndarray:
        """
        Returns the coordinates of the cells in the database as a 3d
        numpy array.
        """
        return self.points_of_cells(*args, **kwargs)

    def topology(self) -> Union[TopologyArray, None]:
        """
        Returns the numerical representation of the topology of
        the cells.
        """
        key = self._dbkey_nodes_
        if key in self.fields:
            return TopologyArray(self.nodes)
        else:
            return None

    def rewire(self, imap: MapLike = None, invert: bool = False) -> "PolyCell":
        """
        Rewires the topology of the block according to the mapping
        described by the argument `imap`. The mapping of the j-th node
        of the i-th cell happens the following way:

        topology_new[i, j] = imap[topology_old[i, j]]

        The object is returned for continuation.

        Parameters
        ----------
        imap: MapLike
            Mapping from old to new node indices (global to local).
        invert: bool, Optional
            If `True` the argument `imap` describes a local to global
            mapping and an inversion takes place. In this case,
            `imap` must be a `numpy` array. Default is False.
        """
        if imap is None:
            imap = self.source().pointdata.id
        topo = self.topology().to_array().astype(int)
        topo = rewire(topo, imap, invert=invert).astype(int)
        self._wrapped[self._dbkey_nodes_] = topo
        return self

    def glob_to_loc(self, x: Union[Iterable, ndarray]) -> ndarray:
        """
        Returns the local coordinates of the input points for each
        cell in the block. The input 'x' can describe a single (1d array),
        or several positions at once (2d array).

        Notes
        -----
        This function is useful when detecting if two bodies touch each other or not,
        and if they do, where.

        Parameters
        ----------
        x: Iterable or numpy.ndarray
            A single point in 3d space as an 1d array, or a collection of points
            as a 2d array.

        Returns
        -------
        numpy.ndarray
            A NumPy array of shape (nE, nP, nD), where nP is the number of points in 'x',
            nE is the number of cells in the block and nD is the number of spatial dimensions.
        """
        raise NotImplementedError

    def loc_to_glob(
        self, x: Union[Iterable, ndarray], ec: Optional[Union[ndarray, None]] = None
    ) -> ndarray:
        """
        Returns the global coordinates of the input points for each
        cell in the block. The input 'x' can describe a single (1d array),
        or several local positions at once (2d array).

        Notes
        -----
        This function is useful when detecting if two bodies touch each other or not,
        and if they do, where.

        Parameters
        ----------
        x: Iterable or numpy.ndarray
            A single point as an 1d array, or a collection of points
            as a 2d array.
        ec: numpy.ndarray, Optional
            Element coordinates as a 3d array of shape (nE, nNE, nD).
            Default is None, in which case the global coordinates of the
            cells are used.

        Returns
        -------
        numpy.ndarray
            A NumPy array of shape (nE, nP, nD), where nP is the number of points in 'x',
            nE is the number of cells in the block and nD is the number of spatial dimensions.
        """
        x = atleast2d(x, front=True)
        shp = self.Geometry.shape_function_values(x)  # (nP, nNE)
        if ec is None:
            ec = self.points_of_cells()
        return _loc_to_glob_bulk_(shp, ec)

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
        NDIM: int = self.Geometry.number_of_spatial_dimensions
        if NDIM == 2:
            x = atleast2d(x, front=True)
            coords = self.source_coords()
            topo = self.to_triangles()
            ecoords = points_of_cells(coords, topo, centralize=False)
            if lazy:
                raise NotImplementedError
            else:
                pips = _pip_tri_bulk_(x, ecoords, tol)
            return np.squeeze(np.any(pips, axis=1))
        elif NDIM == 3:
            x = atleast2d(x, front=True)
            coords = self.source_coords()
            topo = self.to_tetrahedra(flatten=True)
            ecoords = points_of_cells(coords, topo, centralize=False)
            if lazy:
                centers = cell_centers_bulk(coords, topo)
                k = min(k, len(centers))
                knn = k_nearest_neighbours(centers, x, k=k)
                pips = _pip_tet_bulk_knn_(x, ecoords, knn, tol)
            else:
                pips = _pip_tet_bulk_(x, ecoords, tol)
            return np.squeeze(np.any(pips, axis=1))
        else:
            raise NotImplementedError

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
            The indices of 'x' that are inside a cell of the block.
        numpy.ndarray
            The block-local indices of the cells that include the points with
            the returned indices.
        numpy.ndarray
            The parametric coordinates of the located points inside the including cells.
        """
        NDIM: int = self.Geometry.number_of_spatial_dimensions
        if NDIM == 3:
            x = atleast2d(x, front=True)

            coords = self.source_coords()
            topo = self.topology()
            topo_tet = self.to_tetrahedra(flatten=True)
            ecoords_tet = points_of_cells(coords, topo_tet, centralize=False)
            tetmap = self.Geometry.tetmap()

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
                points_to_tets, points_to_neighbours = _find_first_hits_(
                    pips_tet[i_source]
                )
            tets_to_cells = np.floor(np.arange(len(topo_tet)) / len(tetmap)).astype(int)
            i_target = tets_to_cells[points_to_tets]  # (nP_)

            # locate the cells that contain the points
            cell_tet_indices = np.tile(np.arange(tetmap.shape[0]), len(topo))[
                points_to_tets
            ]
            nat_tet = nat_tet[i_source]  # (nP_, nTET, 4)
            locations_target = _ntet_to_loc_bulk_(
                self.Geometry.master_coordinates(),
                nat_tet,
                tetmap,
                cell_tet_indices,
                points_to_neighbours,
            )

            return i_source, i_target, locations_target
        else:
            raise NotImplementedError

    def centers(self, target: FrameLike = None) -> ndarray:
        """Returns the centers of the cells of the block."""
        coords = self.source_coords()
        t = self.topology().to_numpy()
        centers = cell_centers_bulk(coords, t)
        if target:
            pc = PointCloud(centers, frame=self.source_frame())
            centers = pc.show(target)
        return centers

    def unique_indices(self) -> ndarray:
        """
        Returns the indices of the points involved in the cells of the block.
        """
        return np.unique(self.topology())

    def points_involved(self) -> PointCloud:
        """Returns the points involved in the cells of the block."""
        return self.source_points()[self.unique_indices()]

    def detach_points_cells(self) -> Tuple[ndarray]:
        """
        Returns the detached coordinate and topology array of the block.
        """
        coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        return detach_mesh_bulk(coords, topo)

    def to_vtk(self, detach: bool = False) -> Any:
        """
        Returns the block as a VTK object.
        """
        coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        vtkid: int = self.Geometry.vtk_cell_id
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

        if self.Geometry.number_of_spatial_dimensions == 3:
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
        else:
            raise NotImplementedError

    def boundary(self, detach: bool = False) -> Tuple[ndarray]:
        """
        Returns the boundary of the block as 2 NumPy arrays.
        """
        if self.Geometry.number_of_spatial_dimensions == 3:
            return self.extract_surface(detach=detach)
        else:
            raise NotImplementedError

    def _rotate_(self, *args, **kwargs):
        # this is triggered upon transformations performed on the hosting pointcloud
        if self.has_frames:
            source_frame = self.container.source().frame
            new_frames = (
                CartesianFrame(self.frames, assume_cartesian=True)
                .rotate(*args, **kwargs)
                .show(source_frame)
            )
            self.frames = new_frames
