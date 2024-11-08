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
    Hashable,
    Callable,
)
from numbers import Number
from copy import deepcopy

import numpy as np
from numpy import ndarray
from numpy.lib.index_tricks import IndexExpression

from sigmaepsilon.core.typing import issequence
from sigmaepsilon.math import atleast1d, atleast2d, atleastnd, ascont
from sigmaepsilon.math.linalg import ReferenceFrame as FrameLike
from sigmaepsilon.math.utils import to_range_1d
from sigmaepsilon.math.linalg.sparse import csr_matrix

from ..typing import (
    ABC_PolyCell,
    PolyDataProtocol,
    PointDataProtocol,
    GeometryProtocol,
    CellDataProtocol,
)
from .celldata import CellData
from ..space import PointCloud, CartesianFrame
from ..utils import (
    distribute_nodal_data_bulk,
    distribute_nodal_data_sparse,
)
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
    _pip_tet_bulk_knn_,
    _pip_tet_bulk_,
    _glob_to_nat_tet_bulk_,
    _glob_to_nat_tet_bulk_knn_,
    _pip_tet_bulk_nat_,
)
from ..utils.cells.utils import (
    _find_first_hits_,
    _find_first_hits_knn_,
    _ntet_to_loc_bulk_,
    cell_measures,
)
from ..utils.topology.topo import detach_mesh_bulk, rewire
from ..utils.topology import transform_topology
from ..utils.tri import _pip_tri_bulk_
from ..utils.knn import k_nearest_neighbours
from ..utils.space import index_of_closest_point, frames_of_lines, frames_of_surfaces
from ..utils import cell_centers_bulk
from ..vtkutils import mesh_to_UnstructuredGrid as mesh_to_vtk
from ..topoarray import TopologyArray
from ..space import CartesianFrame
from ..utils.numint import Quadrature
from ..config import __haspyvista__

if __haspyvista__:
    import pyvista as pv

MapLike = Union[ndarray, MutableMapping]
PointDataLike = TypeVar("PointDataLike", bound=PointDataProtocol)
MeshDataLike = TypeVar("MeshDataLike", bound=PolyDataProtocol)
T = TypeVar("T", bound="PolyCell")

__all__ = ["PolyCell"]


class PolyCell(Generic[MeshDataLike, PointDataLike], ABC_PolyCell):
    """
    A subclass of :class:`~sigmaepsilon.mesh.data.celldata.CellData` as a base class
    for all cell containers. The class should not be used directly, the main purpose
    here is encapsulation of common behaviour for all kinds of cells.
    """

    label: ClassVar[Optional[str]] = None
    Geometry: ClassVar[GeometryProtocol]

    data_class: ClassVar[type] = CellData[MeshDataLike, PointDataLike]

    def __init__(
        self,
        *args,
        db: Optional[Union[CellData[MeshDataLike, PointDataLike], None]] = None,
        pointdata: Optional[Union[PointDataLike, None]] = None,
        container: Optional[Union[MeshDataLike, None]] = None,
        **kwargs,
    ):
        if db is None:
            db_class = self.__class__.data_class
            db = db_class(*args, **kwargs)

        self._db = db
        self._pointdata = pointdata
        self._container = container

        super().__init__()

    @property
    def db(self) -> CellDataProtocol[MeshDataLike, PointDataLike]:
        """
        Returns the database of the block.
        """
        return self._db

    @db.setter
    def db(self, value: CellDataProtocol[MeshDataLike, PointDataLike]) -> None:
        """
        Sets the database of the block.
        """
        self._db = value

    @property
    def pointdata(self) -> PointDataLike:
        """
        Returns the hosting point database. This is what
        the topology of the cells are referring to.
        """
        return self._pointdata

    @pointdata.setter
    def pointdata(self, value: PointDataLike) -> None:
        """
        Sets the hosting point database. This is what
        the topology of the cells are referring to.
        """
        if value is not None:
            if not isinstance(value, PointDataProtocol):
                raise TypeError("'value' must be a PointData instance")
        self._pointdata = value

    @property
    def pd(self) -> PointDataLike:
        """
        Returns the attached point database. This is what
        the topology of the cells are referring to.
        """
        return self.pointdata

    @pd.setter
    def pd(self, value: PointDataLike) -> None:
        """
        Sets the attached pointdata.
        """
        self.pointdata = value

    @property
    def container(self) -> MeshDataLike:
        """
        Returns the container of the block.
        """
        return self._container

    @container.setter
    def container(self, value: MeshDataLike) -> None:
        """
        Sets the container of the block.
        """
        if not isinstance(value, PolyDataProtocol):
            raise TypeError("'value' must be a PolyData instance")
        self._container = value

    def root(self) -> MeshDataLike:
        """
        Returns the top level container of the model the block is
        the part of.
        """
        c = self.container
        return None if c is None else c.root

    def source(self) -> MeshDataLike:
        """
        Retruns the source of the cells. This is the PolyData block
        that stores the PointData object the topology of the cells
        are referring to.
        """
        c = self.container
        return None if c is None else c.source()

    def pull(
        self, data: Union[str, ndarray], ndf: Union[ndarray, csr_matrix] = None
    ) -> ndarray:
        """
        Pulls data from the attached pointdata. The pulled data is either copied or
        distributed according to a measure.

        Parameters
        ----------
        data: str or numpy.ndarray
            Either a field key to identify data in the database of the attached
            PointData, or a NumPy array.

        See Also
        --------
        :func:`~sigmaepsilon.mesh.utils.utils.distribute_nodal_data_bulk`
        :func:`~sigmaepsilon.mesh.utils.utils.distribute_nodal_data_sparse`
        """
        if isinstance(data, str):
            pd = self.source().pd
            nodal_data = pd[data].to_numpy()
        else:
            assert isinstance(
                data, ndarray
            ), "'data' must be a string or a NumPy array."
            nodal_data = data
        topo = self.nodes
        if ndf is None:
            ndf = np.ones_like(topo).astype(float)
        if len(nodal_data.shape) == 1:
            nodal_data = atleast2d(nodal_data, back=True)
        if isinstance(ndf, ndarray):
            d = distribute_nodal_data_bulk(nodal_data, topo, ndf)
        else:
            d = distribute_nodal_data_sparse(nodal_data, topo, self.id, ndf)
        # nE, nNE, nDATA
        return d

    def _get_cell_slicer(
        self, cells: Optional[Union[int, Iterable[int]]] = None
    ) -> Union[Iterable[int], IndexExpression]:
        if isinstance(cells, Iterable):
            cells = atleast1d(cells)
            conds = np.isin(cells, self.db.id)
            cells = atleast1d(cells[conds])
            assert (
                len(cells) > 0
            ), "Length of cells is zero. At least one cell must be requested"
        else:
            cells = np.s_[:]
        return cells

    def _get_points_and_range(
        self,
        points: Optional[Union[None, Iterable[Number]]] = None,
        rng: Optional[Union[None, Iterable[Number]]] = None,
    ) -> Tuple[ndarray, ndarray]:
        """
        Returns the points and the domain in which they are defined.
        Currently the range is only a matter of interest for 1d cells. Then,
        the points are returned in the range [-1, 1]. In the future it is planned
        to make the target range a parameter of the function.
        """
        nDIM = self.Geometry.number_of_spatial_dimensions
        if nDIM == 1:
            if points is None:
                points = np.array(self.Geometry.master_coordinates()).flatten()
                rng = [-1, 1]
            else:
                points = atleast1d(np.array(points))
                rng = np.array([-1, 1]) if rng is None else np.array(rng)
            points = to_range_1d(points, source=rng, target=[-1, 1]).flatten()
            rng = [-1, 1]
        else:
            if points is None:
                points = np.array(self.Geometry.master_coordinates())

        points, rng = np.array(points, dtype=float), np.array(rng, dtype=float)

        if nDIM > 1:
            points = atleastnd(points, 2, front=True)

        return points, rng

    def _parse_gauss_data(self, quad_dict: dict, key: Hashable) -> Iterable[Quadrature]:
        value: Union[Callable, str, dict] = quad_dict[key]

        if isinstance(value, Callable):
            qpos, qweight = value()
            quad = Quadrature(x=qpos, w=qweight)
            yield quad
        elif isinstance(value, str):
            for v in self._parse_gauss_data(quad_dict, value):
                yield v
        else:
            qpos, qweight = value
            quad = Quadrature(x=qpos, w=qweight)
            yield quad

    @property
    def frames(self) -> ndarray:
        """
        Returns local coordinate frames of the cells as a 3d NumPy float array,
        where the first axis runs along the cells of the block.
        """
        if not self.db.has_frames:
            if (nD := self.Geometry.number_of_spatial_dimensions) == 1:
                coords = self.source_coords()
                topo = self.topology().to_numpy()
                self.db.frames = frames_of_lines(coords, topo)
            elif nD == 2:
                coords = self.source_coords()
                topo = self.topology().to_numpy()
                self.db.frames = frames_of_surfaces(coords, topo)
            elif nD == 3:
                self.db.frames = self.source_frame()
            else:  # pragma: no cover
                raise TypeError(
                    "Invalid Geometry class. The 'number of spatial dimensions'"
                    " must be 1, 2 or 3."
                )
        return self.db.frames

    @frames.setter
    def frames(self, value: Union[FrameLike, ndarray]) -> None:
        self.db.frames = value

    def to_parquet(
        self, path: str, *args, fields: Iterable[str] = None, **kwargs
    ) -> None:
        """
        Saves the data of the database to a parquet file.

        Parameters
        ----------
        *args: tuple, Optional
            Positional arguments to specify fields.
        path: str
            Path of the file being created.
        fields: Iterable[str], Optional
            Valid field names to include in the parquet files.
        **kwargs: dict, Optional
            Keyword arguments forwarded to :func:`awkward.to_parquet`.
        """
        self.db.to_parquet(path, *args, fields=fields, **kwargs)

    @classmethod
    def from_parquet(cls, path: str) -> "PolyCell":
        """
        Saves the data of the database to a parquet file.

        Parameters
        ----------
        path: str
            Path of the file being created.
        """
        db_class = cls.data_class
        return cls(db=db_class.from_parquet(path))

    def to_triangles(self) -> ndarray:
        """
        Returns the topology as a collection of T3 triangles, represented
        as a 2d NumPy integer array, where the first axis runs along the
        triangles, and the second along the nodes.

        Only for 2d cells.
        """
        if self.Geometry.number_of_spatial_dimensions == 2:
            t = self.topology().to_numpy()
            return transform_topology(t, self.Geometry.trimap())
        else:
            raise NotImplementedError("This is only for 2d cells")

    def to_tetrahedra(self, flatten: Optional[bool] = True) -> ndarray:
        """
        Returns the topology as a collection of TET4 tetrahedra, represented
        as a 2d NumPy integer array, where the first axis runs along the
        tetrahedra, and the second along the nodes.

        Only for 3d cells.

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
        Returns the cells of the block, refactorized into simplices. For cells
        of dimension 2, the returned 2d NumPy integer array represents 3-noded
        triangles, for 3d cells it is a collection of 4-noded tetrahedra.
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
        self,
        *,
        pcoords: Optional[Union[Iterable[float], None]] = None,
        dshp: Optional[Union[ndarray, None]] = None,
        **kwargs,
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

        Note
        ----
        For 1d cells, the returned array is also 4 dimensional, with the last two
        axes being dummy.
        """
        ecoords = kwargs.get("_ec", self.local_coordinates())

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

    def jacobian(
        self, *, jac: Optional[Union[ndarray, None]] = None, **kwargs
    ) -> Union[float, ndarray]:
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
        self.db.nodes = np.flip(topo, axis=1)
        return self

    def measures(self, *args, **kwargs) -> ndarray:
        """
        Returns generalized measures for each cell in the block.
        The generalized measure is length for 1d cells,
        area for 2d cells and volume for 3d cells.
        """
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
        """
        Returns the net generalized measure for the cells in the
        block. The generalized measure is length for 1d cells,
        area for 2d cells and volume for 3d cells.
        """
        return np.sum(self.measures(*args, **kwargs))

    def thickness(self) -> ndarray:
        """
        Returns the thicknesses of the cells. If not set, a thickness
        of 1.0 is returned for each cell. Only for 2d cells.
        """
        if self.Geometry.number_of_spatial_dimensions == 2:
            if self.db.has_thickness:
                t = self.db.t
            else:
                t = np.ones(len(self), dtype=float)
            return t
        else:
            raise NotImplementedError("This is only for 2d cells")

    def length(self) -> float:
        """
        Returns the total length of the cells in the block.
        """
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
        Returns the total area of the cells in the database.
        Only for 2d entities.
        """
        if self.Geometry.number_of_spatial_dimensions == 2:
            return np.sum(self.areas(*args, **kwargs))
        else:
            raise NotImplementedError("This is only for 2d cells")

    def areas(self, *args, **kwargs) -> ndarray:
        """
        Returns the areas of the individuall cells in the database as
        a 1d float NumPy array.

        Note
        ----
        For 1d cells, the cross sectional areas are returned.
        """
        NDIM: int = self.Geometry.number_of_spatial_dimensions
        if NDIM == 1:
            if self.db.has_areas:
                return self.db.A
            else:
                return np.ones((len(self)))
        elif NDIM == 2:
            coords = self.source_coords()
            topo = self.topology().to_numpy()
            ecoords = cells_coords(coords[:, :2], topo)
            quad: Quadrature = next(
                self._parse_gauss_data(self.Geometry.quadrature, "geometry")
            )
            dshp = self.Geometry.shape_function_derivatives(quad.pos)
            return cell_measures(ecoords, dshp, quad.weight)
        else:
            raise NotImplementedError("This is only for 2d cells")

    def volume(self, *args, **kwargs) -> float:
        """
        Returns the volume of the cells in the database.
        """
        return np.sum(self.volumes(*args, **kwargs))

    def volumes(self, *args, **kwargs) -> ndarray:
        """
        Returns the volumes of the cells in the database.
        """
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
            ecoords = cells_coords(coords, topo)
            quad: Quadrature = next(
                self._parse_gauss_data(self.Geometry.quadrature, "geometry")
            )
            dshp = self.Geometry.shape_function_derivatives(quad.pos)
            return cell_measures(ecoords, dshp, quad.weight)
        else:  # pragma: no cover
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
            coords = self.source().coords()
        return coords

    def source_frame(self) -> FrameLike:
        """
        Returns the frame of the hosting pointcloud.
        """
        return self.source().frame

    def points_of_cells(
        self,
        *,
        points: Optional[Union[float, Iterable, None]] = None,
        cells: Optional[Union[int, Iterable, None]] = None,
        rng: Optional[Union[Iterable, None]] = None,
    ) -> ndarray:
        """
        Returns the points of selected cells as a NumPy array. The returned
        array is three dimensional with a shape of (nE, nNE, nD), where `nE` is
        the number of cells in the block, `nNE` is the number of nodes per cell or
        the number of the points (if 'points' is specified) and nD stands for the
        number of spatial dimensions.

        Parameters
        ----------
        points: Optional[Union[float, Iterable, None]]
            Points defined in the domain of the master cell. If specified, global
            coordinates for each cell are calculated and returned for each cell.
            Default is `None`, in which case the locations of the nodes of the cells
            are used.
        cells: Optional[Union[int, Iterable, None]]
            BLock-local indices of the cells of interest, or `None` if all of the
            cells in the block are of interest. Default is `None`.
        rng: Optional[Union[Iterable, None]]
            For 1d cells only, it is possible to provide an iterable of length 2
            as an interval (or range) in which the argument `points` is to be understood.
            Default is `None`, in which case the `points` are expected in the range [-1, 1].
        """
        cells = self._get_cell_slicer(cells)

        NDIM: int = self.Geometry.number_of_spatial_dimensions
        coords = self.source_coords()
        topo = self.topology().to_numpy()[cells]
        ecoords = points_of_cells(coords, topo, centralize=False)

        if points is None:
            return ecoords

        points, rng = self._get_points_and_range(points, rng)

        if NDIM == 1:
            points = to_range_1d(points, source=rng, target=[0, 1]).flatten()
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

    def local_coordinates(
        self, *, target: Optional[Union[str, CartesianFrame, None]] = None
    ) -> ndarray:
        """
        Returns local coordinates of the cells as a 3d float NumPy array.
        The returned array is three dimensional with a shape of (nE, nNE, 2),
        where `nE` is the number of cells in the block, `nNE` is the number of
        nodes per cell and 2 stands for the 2 spatial dimensions. The coordinates
        are centralized to the centers for each cell.

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
        coords = self.source_coords()
        centers = self.loc_to_glob(self.Geometry.master_center())
        res = points_of_cells(
            coords, topo, local_axes=frames, centralize=True, centers=centers
        )

        if self.Geometry.number_of_spatial_dimensions == 2:
            return ascont(res[:, :, :2])
        else:
            return res

    def coords(self, *args, **kwargs) -> ndarray:
        """
        Alias for :func:`points_of_cells`, all arguments are forwarded.
        """
        return self.points_of_cells(*args, **kwargs)

    def topology(self) -> TopologyArray | None:
        """
        Returns the numerical representation of the topology of
        the cells as either a :class:`~sigmaepsilon.mesh.topoarray.TopologyArray`
        or `None` if the topology is not specified yet.
        """
        if self.db.has_nodes:
            return TopologyArray(self.db.nodes)
        else:  # pragma: no cover
            return None

    def rewire(
        self,
        imap: Optional[Union[MapLike, None]] = None,
        invert: Optional[bool] = False,
    ) -> "PolyCell":
        """
        Rewires the topology of the block according to the mapping
        described by the argument `imap`. The mapping of the j-th node
        of the i-th cell happens the following way:

        topology_new[i, j] = imap[topology_old[i, j]]

        The object is returned for continuation.

        Parameters
        ----------
        imap: MapLike, Optional
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
        self.db.nodes = topo
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
        nD = self.Geometry.number_of_spatial_dimensions
        if nD > 1:
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
                pips_tet = _pip_tet_bulk_nat_(nat_tet, tol)  # (nP, kTET)
            else:
                nat_tet = _glob_to_nat_tet_bulk_(x, ecoords_tet)  # (nP, nTET, 4)
                pips_tet = _pip_tet_bulk_nat_(nat_tet, tol)  # (nP, nTET)

            # locate the points that are inside any of the cells
            pip = np.squeeze(np.any(pips_tet, axis=1))  # (nP)
            i_source = np.nonzero(np.atleast_1d(pip))  # (nP_)
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

    def centers(self, target: Optional[Union[CartesianFrame, None]] = None) -> ndarray:
        """
        Returns the centers of the cells of the block as a 1d float
        NumPy array.

        Parameters
        ----------
        target: CartesianFrame, Optional
            A target frame. If provided, coordinates are returned in
            this frame, otherwise they are returned in the global frame.
            Default is None.
        """
        coords = self.source_coords()
        t = self.topology().to_numpy()
        centers = cell_centers_bulk(coords, t)
        if target:
            pc = PointCloud(centers, frame=self.source_frame())
            centers = pc.show(target)
        return centers

    def normals(self) -> ndarray:
        """
        Returns the normals of the cells as a 2d NumPy array.
        """
        if self.Geometry.number_of_spatial_dimensions == 2:
            return ascont(self.frames[:, 2, :])
        else:
            raise NotImplementedError("This is only available for 2d cells.")

    def unique_indices(self) -> ndarray:
        """
        Returns the indices of the points involved in the cells of the block
        as a 1d integer NumPy array.
        """
        return np.unique(self.topology())

    def points_involved(self) -> PointCloud:
        """
        Returns the points involved in the cells of the block.
        """
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

        Parameters
        ----------
        detach: bool, Optional
            Wether to detach the mesh or not. Default is False.
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
            Returns the block as a PyVista object.

            Parameters
            ----------
            detach: bool, Optional
                Wether to detach the mesh or not. Default is False.
            """
            return pv.wrap(self.to_vtk(detach=detach))

    def extract_surface(self, detach: bool = False) -> Tuple[ndarray]:
        """
        Extracts the surface of the object as a 2-tuple of NumPy arrays
        representing the coordinates and the topology of a triangulation.

        Parameters
        ----------
        detach: bool, Optional
            Wether to detach the mesh or not. Default is False.
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
        Alias for :func:`extract_surface`.
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

    def __len__(self) -> int:
        return len(self.db)

    def __hasattr__(self, attr: str) -> Any:
        return attr in self.__dict__ or hasattr(self.db, attr)

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dict__:
            return self.__dict__[attr]
        try:
            return getattr(self.db, attr)
        except Exception:
            raise AttributeError(
                "'{}' object has no attribute "
                "called {}".format(self.__class__.__name__, attr)
            )

    def __getitem__(self, index: str) -> Any:
        try:
            return super().__getitem__(index)
        except Exception:
            try:
                return self.db.__getitem__(index)
            except Exception:
                raise TypeError(
                    "'{}' object is not "
                    "subscriptable".format(self.__class__.__name__)
                )

    def __setitem__(self, index: str, value: Any) -> None:
        if not isinstance(index, str):
            raise TypeError(f"Expected a string, got {type(index)}")

        if not (issequence(value) and len(value) == len(self.db)):
            raise ValueError(
                "The length of the provided data must match the number of cells in the block"
            )

        self.db[index] = value

    def __deepcopy__(self, memo: dict) -> "PolyCell":
        return self.__copy__(memo)

    def __copy__(self, memo: Optional[Union[dict, None]] = None) -> "PolyCell":
        cls = type(self)
        is_deep = memo is not None

        if is_deep:
            copy_function = lambda x: deepcopy(x, memo)
        else:
            copy_function = lambda x: x

        db = copy_function(self.db)

        pd = self.pointdata
        pd_copy = None
        if pd is not None:
            if is_deep:
                pd_copy = memo.get(id(pd), None)
            if pd_copy is None:
                pd_copy = copy_function(pd)

        result = cls(db=db, pointdata=pd_copy)
        if is_deep:
            memo[id(self)] = result

        result_dict = result.__dict__
        for k, v in self.__dict__.items():
            if not k in result_dict:
                setattr(result, k, copy_function(v))

        return result
