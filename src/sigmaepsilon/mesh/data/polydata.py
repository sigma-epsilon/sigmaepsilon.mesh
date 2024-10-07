from copy import copy, deepcopy
from typing import (
    Iterator,
    Union,
    Hashable,
    Iterable,
    Tuple,
    Any,
    Generic,
    Optional,
    Callable,
    Generator,
)
from types import NoneType
from collections import defaultdict
import functools
import warnings
import importlib

from numpy import ndarray
import numpy as np
from scipy.sparse import spmatrix, csr_matrix as csr_scipy, diags as scipy_diags
import awkward as ak
from meshio import Mesh as MeshioMesh

from sigmaepsilon.deepdict import DeepDict
from sigmaepsilon.core.warning import SigmaEpsilonPerformanceWarning
from sigmaepsilon.math.linalg.sparse import csr_matrix, JaggedArray
from sigmaepsilon.math.linalg import Vector, ReferenceFrame as FrameLike
from sigmaepsilon.math import atleast1d, minmax

from ..typing import (
    PolyDataProtocol as PDP,
    PolyCellProtocol,
    PolyDataLike,
    PointDataLike,
    PolyCellLike,
)

from .akwrapper import AkWrapper
from .pointdata import PointData
from .polycell import PolyCell
from .polycell import PolyCell
from ..space import CartesianFrame, PointCloud
from ..indexmanager import IndexManager
from ..topoarray import TopologyArray

from ..utils.topology.topo import inds_to_invmap_as_dict, remap_topo_1d
from ..utils.utils import (
    cells_coords,
    cells_around,
    cell_centers_bulk,
    explode_mesh_data_bulk,
    nodal_distribution_factors,
)
from ..utils.knn import k_nearest_neighbours as KNN
from ..cells import T3 as Triangle
from ..utils.space import (
    index_of_closest_point,
    index_of_furthest_point,
    frames_of_surfaces,
)
from ..utils.topology import (
    nodal_adjacency,
    detach_mesh_data_bulk,
    detach_mesh_bulk,
    cells_at_nodes,
)
from ..helpers import importers, exporters, plotters
from ..config import __hasvtk__, __haspyvista__, __hask3d__, __hasmatplotlib__

VectorLike = Union[Vector, ndarray]

__all__ = ["PolyData"]


class PolyData(DeepDict[Hashable, PDP | Any], Generic[PointDataLike, PolyCellLike]):
    """
    A class to handle complex polygonal meshes.

    The `PolyData` class is the most important class in the library
    and a backbone of all mesh classes.

    The implementation is based on the `Awkward` library, which provides
    memory-efficient, numba-jittable data classes to deal with dense, sparse,
    complete or incomplete data. These data structures are managed in pure
    Python by the `DeepDict` class.

    The class accepts several kinds of inputs, allowing for a wide range of
    possible use cases. The fastes way to create a PolyData is from predefined
    `PointData` and `CellData` instances, defined separately.

    Parameters
    ----------
    pd: Union[PointData, CellData], Optional
        A PolyData or a CellData instance. Dafault is None.
    cd: CellData, Optional
        A CellData instance, if the first argument is provided. Dafault is None.

    Examples
    --------
    To create a simple cube:

    >>> from sigmaepsilon.mesh import PolyData, PointData
    >>> from sigmaepsilon.mesh.grid import grid
    >>> from sigmaepsilon.mesh.space import StandardFrame
    >>> from sigmaepsilon.mesh.cells import H27
    >>> size = Lx, Ly, Lz = 100, 100, 100
    >>> shape = nx, ny, nz = 10, 10, 10
    >>> coords, topo = grid(size=size, shape=shape, eshape='H27')
    >>> frame = StandardFrame(dim=3)
    >>> mesh = PolyData(pd=PointData(coords=coords, frame=frame))
    >>> mesh['A']['Part1'] = PolyData(cd=H27(topo=topo[:10], frames=frame))
    >>> mesh['A']['Part2'] = PolyData(cd=H27(topo=topo[10:-10], frames=frame))
    >>> mesh['A']['Part3'] = PolyData(cd=H27(topo=topo[-10:], frames=frame))
    >>> mesh.plot()  # doctest: +SKIP

    Load a mesh from a PyVista object:

    >>> from pyvista import examples
    >>> from sigmaepsilon.mesh import PolyData
    >>> bunny = examples.download_bunny_coarse()
    >>> mesh = PolyData.from_pv(bunny)

    Read from a .vtk file:

    >>> from sigmaepsilon.mesh import PolyData
    >>> from sigmaepsilon.mesh.downloads import download_stand
    >>> vtkpath = download_stand()
    >>> mesh = PolyData.read(vtkpath)

    See also
    --------
    :class:`~sigmaepsilon.mesh.data.trimesh.TriMesh`
    :class:`~sigmaepsilon.mesh.data.pointdata.PointData`
    :class:`~sigmaepsilon.mesh.data.celldata.CellData`
    """

    _point_array_class_ = PointCloud
    _point_class_ = PointData
    _frame_class_ = CartesianFrame
    _pv_config_key_ = ("pv", "default")
    _k3d_config_key_ = ("k3d", "default")

    def __init__(
        self,
        pd: Optional[Union[PointData, PolyCell, None]] = None,
        cd: Optional[Union[PolyCell, None]] = None,
        *args,
        **kwargs,
    ):
        self._reset_point_data()
        self._reset_cell_data()
        self._parent = None
        self._config = None
        self._cid2bid = None  # maps cell indices to block indices
        self._bid2b = None  # maps block indices to block addresses
        self._pointdata = None
        self._celldata = None
        self._init_config_()

        self.point_index_manager = IndexManager()
        self.cell_index_manager = IndexManager()

        if isinstance(pd, PointData):
            self.pointdata = pd
            if isinstance(cd, PolyCell):
                self.celldata = cd
        elif isinstance(pd, PolyCell):
            self.celldata = pd
            if isinstance(cd, PointData):
                self.pointdata = cd
        elif isinstance(cd, PolyCell):
            self.celldata = cd

        pidkey = self.__class__._point_class_._dbkey_id_

        if self.pointdata is not None:
            if self.pd.has_id:
                if self.celldata is not None:
                    imap = self.pd.id
                    self.cd.rewire(imap=imap, invert=True)
            N = len(self.pointdata)
            GIDs = self.root.pim.generate_np(N)
            self.pd[pidkey] = GIDs
            self.pd.container = self

        if self.celldata is not None:
            N = len(self.celldata.db)
            GIDs = self.root.cim.generate_np(N)
            self.cd.db.id = GIDs
            try:
                pd = self.source().pd
            except Exception:
                pd = None
            self.cd.pd = pd
            self.cd.container = self

        super().__init__(*args, **kwargs)

        if self.celldata is not None:
            self.celltype = self.celldata.__class__
            self.celldata.container = self

    def __deepcopy__(self, memo):
        return self.__copy__(memo)

    def __copy__(self, memo=None):
        cls = type(self)
        is_deep = memo is not None

        if is_deep:
            copy_function = lambda x: deepcopy(x, memo)
        else:
            copy_function = lambda x: x

        frame_cls = self._frame_class_

        # initialize result
        if self.frame is not None:
            f = self.frame
            ax = copy_function(f.axes)
            if is_deep:
                memo[id(f.axes)] = ax
            frame = frame_cls(ax)
        else:
            frame = None
        result = cls(frame=frame)
        if is_deep:
            memo[id(self)] = result

        # self
        if self.pointdata is not None:
            result.pointdata = copy_function(self.pointdata)

        if self.celldata is not None:
            result.celldata = copy_function(self.celldata)

        for k, v in self.items():
            if not isinstance(v, PolyData):
                result[k] = copy_function(v)

        result_dict = result.__dict__

        for k, v in self.__dict__.items():
            if not k in result_dict:
                setattr(result, k, copy_function(v))

        # children
        l0 = len(self.address)
        for b in self.blocks(inclusive=False, deep=True):
            pd, cd, bframe = None, None, None
            addr = b.address

            if len(addr) > l0:
                # pointdata
                if b.pointdata is not None:
                    pd = copy_function(b.pd)
                    # block frame
                    f = b.frame
                    ax = copy_function(f.axes)

                    if is_deep:
                        memo[id(f.axes)] = ax

                    bframe = frame_cls(ax)

                # celldata
                if b.celldata is not None:
                    cd = copy_function(b.cd)

                # mesh object
                pd_result = PolyData(pd, cd, frame=bframe)
                result[addr[l0:]] = pd_result

                # other data
                for k, v in b.items():
                    if not isinstance(v, PolyData):
                        pd_result[k] = copy_function(v)

                pd_result_dict = pd_result.__dict__

                for k, v in b.__dict__.items():
                    if not k in pd_result_dict:
                        setattr(pd_result, k, copy_function(v))

        return result

    def copy(self: PolyDataLike) -> PolyDataLike:
        """
        Returns a shallow copy.
        """
        return copy(self)

    def deepcopy(self: PolyDataLike) -> PolyDataLike:
        """
        Returns a deep copy.
        """
        return deepcopy(self)

    def __getitem__(self: PolyDataLike, key) -> PolyDataLike:
        return super().__getitem__(key)

    @property
    def pointdata(self) -> PointDataLike:
        """
        Returns the attached pointdata.
        """
        return self._pointdata

    @pointdata.setter
    def pointdata(self, pd: PointData | NoneType) -> NoneType:
        """
        Returns the attached pointdata.
        """
        if pd is not None and not isinstance(pd, PointData):
            raise TypeError("Value must be a PointData instance.")
        self._pointdata = pd
        if isinstance(pd, PointData):
            self._pointdata.container = self

    @property
    def pd(self) -> PointDataLike:
        """
        Returns the attached pointdata.
        """
        return self.pointdata

    @property
    def celldata(self) -> PolyCellLike:
        """
        Returns the attached celldata.
        """
        return self._celldata

    @celldata.setter
    def celldata(self, cd: PolyCell | NoneType) -> NoneType:
        """
        Returns the attached celldata.
        """
        if cd is not None and not isinstance(cd, PolyCell):
            raise TypeError("Value must be a PolyCell instance.")
        self._celldata = cd
        if isinstance(cd, PolyCell):
            self._celldata.container = self

    @property
    def cd(self) -> PolyCellLike:
        """
        Returns the attached celldata.
        """
        return self.celldata

    def lock(self: PolyDataLike, create_mappers: bool = False) -> PolyDataLike:
        """
        Locks the layout. If a `PolyData` instance is locked,
        missing keys are handled the same way as they would've been handled
        if it was a `dict`. Also, setting or deleting items in a locked
        dictionary and not possible and you will experience an error upon
        trying.

        The object is returned for continuation.

        Parameters
        ----------
        create_mappers: bool, Optional
            If True, some mappers are generated to speed up certain types of
            searches, like finding a block containing cells based on their
            indices.
        """
        if create_mappers and self._cid2bid is None:
            bid2b, cid2bid = self._create_mappers_()
            self._cid2bid = cid2bid  # maps cell indices to block indices
            self._bid2b = bid2b  # maps block indices to block addresses
        self._locked = True
        return self

    def unlock(self: PolyDataLike) -> PolyDataLike:
        """
        Releases the layout. If a `sigmaepsilon.mesh` instance is not locked,
        a missing key creates a new level in the layout, also setting and
        deleting items becomes an option. Additionally, mappers created with
        the call `generate_cell_mappers` are deleted.

        The object is returned for continuation.
        """
        self._locked = False
        self._cid2bid = None  # maps cell indices to block indices
        self._bid2b = None  # maps block indices to block addresses
        return self

    def blocks_of_cells(self, i: int | Iterable | NoneType = None) -> dict:
        """
        Returns a dictionary that maps cell indices to blocks.
        """
        assert self.is_root(), "This must be called on the root object."

        if self._cid2bid is None:
            warnings.warn(
                "Calling 'obj.lock(create_mappers=True)' creates additional"
                " mappers that make lookups like this much more efficient. "
                "See the doc of the sigmaepsilon.mesh library for more details.",
                SigmaEpsilonPerformanceWarning,
            )
            bid2b, cid2bid = self._create_mappers_()
        else:
            cid2bid = self._cid2bid
            bid2b = self._bid2b

        if i is None:
            return {cid: bid2b[bid] for cid, bid in cid2bid.items()}

        cids = atleast1d(i)
        bids = [cid2bid[cid] for cid in cids]
        cid2b = {cid: bid2b[bid] for cid, bid in zip(cids, bids)}

        return cid2b

    def _create_mappers_(self) -> Tuple[dict, dict]:
        """
        Generates mappers between cells and blocks to speed up some
        queries. This can only be called on the root object.
        The object is returned for continuation.
        """
        assert self.is_root(), "This must be called on the root object."
        bid2b = {}  # block index to block address
        cids = []  # cell indices
        bids = []  # block infices of cells
        for bid, b in enumerate(self.cellblocks(inclusive=True)):
            b.id = bid
            bid2b[bid] = b
            cids.append(b.cd.id)
            bids.append(np.full(len(b.cd), bid))
        cids = np.concatenate(cids)
        bids = np.concatenate(bids)
        cid2bid = {cid: bid for cid, bid in zip(cids, bids)}
        return bid2b, cid2bid

    @classmethod
    def read(cls: PolyDataLike, *args, **kwargs) -> PolyDataLike:
        """
        Reads from a file using PyVista.

        Example
        -------
        Download a .vtk file and read it:

        >>> from sigmaepsilon.mesh import PolyData
        >>> from sigmaepsilon.mesh.downloads import download_stand
        >>> vtkpath = download_stand(read=False)
        >>> mesh = PolyData.read(vtkpath)
        """
        if not __haspyvista__:
            raise ImportError("PyVista is not available.")
        pv = importlib.import_module("pyvista")
        return cls.from_pv(pv.read(*args, **kwargs))

    @classmethod
    def from_meshio(cls: PolyDataLike, mesh: MeshioMesh) -> PolyDataLike:
        """
        Returns a :class:`~sigmaepsilon.mesh.polydata.PolyData` instance from
        a :class:`meshio.Mesh` instance.

        .. note::
            See https://github.com/nschloe/meshio for formats supported by
            ``meshio``. Be sure to install ``meshio`` with ``pip install
            meshio`` if you wish to use it.
        """
        importer: Callable = importers["meshio"]
        return importer(mesh)

    @classmethod
    def from_pv(cls: PolyDataLike, pvobj) -> PolyDataLike:
        """
        Returns a :class:`~sigmaepsilon.mesh.polydata.PolyData` instance from
        a :class:`pyvista.PolyData` or a :class:`pyvista.UnstructuredGrid`
        instance.

        .. note::
            See https://github.com/pyvista/pyvista for more examples with
            ``pyvista``. Be sure to install ``pyvista`` with ``pip install
            pyvista`` if you wish to use it.

        Example
        -------
        >>> from pyvista import examples
        >>> from sigmaepsilon.mesh import PolyData
        >>> bunny = examples.download_bunny_coarse()
        >>> mesh = PolyData.from_pv(bunny)
        """
        importer: Callable = importers["PyVista"]
        return importer(pvobj)

    def to_dataframe(
        self,
        *args,
        point_fields: Optional[Union[Iterable[str], None]] = None,
        cell_fields: Optional[Union[Iterable[str], None]] = None,
        **kwargs,
    ) -> Any:
        """
        Returns the data contained within the mesh to pandas dataframes.

        Parameters
        ----------
        point_fields: Iterable[str], Optional
            A list of keys that might identify data in a database for the
            points in the mesh. Default is None.
        cell_fields: Iterable[str], Optional
            A list of keys that might identify data in a database for the
            cells in the mesh. Default is None.

        Example
        -------
        >>> from sigmaepsilon.mesh.downloads import download_stand
        >>> mesh = download_stand(read=True)
        >>> _ = mesh.to_dataframe(point_fields=mesh.pd.fields)
        """
        ak_pd, ak_cd = self.to_ak(
            *args, point_fields=point_fields, cell_fields=cell_fields
        )
        return ak.to_dataframe(ak_pd, **kwargs), ak.to_dataframe(ak_cd, **kwargs)

    def to_parquet(
        self,
        path_pd: str,
        path_cd: str,
        *args,
        point_fields: Optional[Union[Iterable[str], None]] = None,
        cell_fields: Optional[Union[Iterable[str], None]] = None,
        **kwargs,
    ) -> None:
        """
        Saves the data contained within the mesh to parquet files.

        Parameters
        ----------
        path_pd: str
            File path for point-related data.
        path_cd: str
            File path for cell-related data.
        point_fields: Iterable[str], Optional
            A list of keys that might identify data in a database for the
            points in the mesh. Default is None.
        cell_fields: Iterable[str], Optional
            A list of keys that might identify data in a database for the
            cells in the mesh. Default is None.

        Example
        -------
        >>> from sigmaepsilon.mesh.downloads import download_stand
        >>> mesh = download_stand(read=True)
        >>> _ = mesh.to_parquet('pd.parquet', 'cd.parquet', point_fields=mesh.pd.fields)
        """
        ak_pd, ak_cd = self.to_ak(
            *args, point_fields=point_fields, cell_fields=cell_fields
        )
        ak.to_parquet(ak_pd, path_pd, **kwargs)
        ak.to_parquet(ak_cd, path_cd, **kwargs)

    def to_ak(
        self,
        *args,
        point_fields: Optional[Union[Iterable[str], None]] = None,
        cell_fields: Optional[Union[Iterable[str], None]] = None,
        **__,
    ) -> Tuple[ak.Array]:
        """
        Returns the data contained within the mesh as a tuple of two
        Awkward arrays.

        Parameters
        ----------
        point_fields: Iterable[str], Optional
            A list of keys that might identify data in a database for the
            points in the mesh. Default is None.
        cell_fields: Iterable[str], Optional
            A list of keys that might identify data in a database for the
            cells in the mesh. Default is None.

        Example
        -------
        >>> from sigmaepsilon.mesh.downloads import download_stand
        >>> mesh = download_stand(read=True)
        >>> _ = mesh.to_ak(point_fields=mesh.pd.fields)
        """
        lp, lc = self.to_lists(
            *args, point_fields=point_fields, cell_fields=cell_fields
        )
        return ak.from_iter(lp), ak.from_iter(lc)

    def to_lists(
        self,
        *,
        point_fields: Optional[Union[Iterable[str], None]] = None,
        cell_fields: Optional[Union[Iterable[str], None]] = None,
    ) -> Tuple[list]:
        """
        Returns data of the object as a tuple of lists. The first is a list
        of point-related, the other one is cell-related data. Unless specified
        by 'fields', all data is returned from the pointcloud and the related
        cells of the mesh.

        Parameters
        ----------
        point_fields: Iterable[str], Optional
            A list of keys that might identify data in a database for the
            points in the mesh. Default is None.
        cell_fields: Iterable[str], Optional
            A list of keys that might identify data in a database for the
            cells in the mesh. Default is None.

        Example
        -------
        >>> from sigmaepsilon.mesh.downloads import download_stand
        >>> mesh = download_stand(read=True)
        >>> _ = mesh.to_lists(point_fields=mesh.pd.fields)
        """
        # handle points
        blocks = self.pointblocks(inclusive=True, deep=True)
        if point_fields is not None:

            def foo(b):
                pdb = b.pd.db
                db = {}
                for f in point_fields:
                    if f in pdb.fields:
                        db[f] = pdb[f]
                    else:
                        raise KeyError(f"Point field {f} not found.")
                w = AkWrapper(fields=db)
                return w.db.to_list()

        else:

            def foo(b):
                return b.pd.db.to_list()

        lp = list(map(foo, blocks))
        lp = functools.reduce(lambda a, b: a + b, lp)
        # handle cells
        blocks = self.cellblocks(inclusive=True, deep=True)
        if cell_fields is not None:

            def foo(b):
                cdb = b.cd.db
                db = {}
                for f in cell_fields:
                    if f in cdb.fields:
                        db[f] = cdb[f]
                    else:
                        raise KeyError(f"Cell field {f} not found.")
                cd = AkWrapper(fields=db)
                return cd.db.to_list()

        else:

            def foo(b):
                return b.cd.db.to_list()

        lc = list(map(foo, blocks))
        lc = functools.reduce(lambda a, b: a + b, lc)
        return lp, lc

    @property
    def config(self) -> DeepDict:
        """
        Returns the configuration object.

        Returns
        -------
        :class:`linkeddeepdict.LinkedDeepDict`
            The configuration object.

        Example
        -------
        >>> from sigmaepsilon.mesh.downloads import download_stand
        >>> mesh = download_stand(read=True)

        To set configuration values related to plotting with `pyVista`,
        do the following:

        >>> mesh.config['pyvista', 'plot', 'color'] = 'red'
        >>> mesh.config['pyvista', 'plot', 'style'] = 'wireframe'

        Then, when it comes to plotting, you can specify your configuration
        with the `config_key` keyword argument:

        >>> mesh.pvplot(config_key=('pyvista', 'plot'))  # doctest: +SKIP

        This way, you can store several different configurations for
        different scenarios.
        """
        return self._config

    def _init_config_(self):
        self._config = DeepDict()
        key = self.__class__._pv_config_key_
        self.config[key]["show_edges"] = True

    @property
    def pim(self) -> "IndexManager":
        return self.point_index_manager

    @property
    def cim(self) -> "IndexManager":
        return self.cell_index_manager

    @property
    def parent(self: PolyDataLike) -> PolyDataLike:
        """Returns the parent of the object."""
        return self._parent

    @parent.setter
    def parent(self, value: PolyDataLike) -> None:
        """Sets the parent."""
        self._parent = value

    def is_source(self, key: str | NoneType = None) -> bool:
        """
        Returns `True`, if the object is a valid source of data
        specified by `key`.

        Parameters
        ----------
        key: str
            A valid key to the PointData of the mesh. If not specified
            the key is the key used for storing coorindates.
        """
        key = PointData._dbkey_x_ if key is None else key
        return self.pointdata is not None and key in self.pointdata.fields

    def source(
        self, key: str | NoneType = None
    ) -> Union[PDP[PointDataLike, PolyCellLike], None]:
        """
        Returns the closest (going upwards in the hierarchy) block that holds
        on to data with a certain field name. If called without arguments,
        it is looking for a block with a valid pointcloud, definition, otherwise
        the field specified by the argument `key`.

        Parameters
        ----------
        key: str
            A valid key in any of the blocks with data. Default is None.
        """
        if self.is_source(key):
            return self
        else:
            if self.is_root():
                return None
            else:
                return self.parent.source(key=key)

    def blocks(
        self: PolyDataLike,
        *,
        inclusive: bool = False,
        blocktype: Any = None,
        deep: bool = True,
        **__,
    ) -> Generator[PolyDataLike, None, None]:
        """
        Returns an iterable over nested blocks.

        Parameters
        ----------
        inclusive: bool, Optional
            Whether to include the object the call was made upon.
            Default is False.
        blocktype: Any, Optional
            A required type. Default is None, which means theat all
            subclasses of the PolyData class are accepted. Default is None.
        deep: bool, Optional
            If True, parsing goes into deep levels. If False, only the level
            of the current object is handled.

        Yields
        ------
        PolyDataLike
            A PolyData instance. The actual type depends on the 'blocktype'
            parameter.
        """
        dtype = PolyData if blocktype is None else blocktype
        return self.containers(inclusive=inclusive, dtype=dtype, deep=deep)

    def pointblocks(
        self, *args, **kwargs
    ) -> Generator[PDP[PointDataLike, PolyCellLike], None, None]:
        """
        Returns an iterable over blocks with PointData. All arguments
        are forwarded to :func:`blocks`.

        Yields
        ------
        Any
            A PolyData instance with a PointData.

        See also
        --------
        :func:`blocks`
        :class:`~sigmaepsilon.mesh.data.pointdata.PointData`
        """
        return filter(lambda i: i.pd is not None, self.blocks(*args, **kwargs))

    def cellblocks(
        self, *args, **kwargs
    ) -> Generator[PDP[PointDataLike, PolyCellLike], None, None]:
        """
        Returns an iterable over blocks with CellData. All arguments
        are forwarded to :func:`blocks`.

        Yields
        ------
        Any
            A CellData instance with a CellData.

        See also
        --------
        :func:`blocks`
        :class:`~sigmaepsilon.mesh.data.celldata.CellData`
        """
        return filter(lambda i: i.cd is not None, self.blocks(*args, **kwargs))

    @property
    def point_fields(self) -> Iterable[str]:
        """
        Returns the fields of all the pointdata of the object.

        Returns
        -------
        numpy.ndarray
            NumPy array of data keys.
        """
        pointblocks = list(self.pointblocks())
        m = map(lambda pb: pb.pointdata.fields, pointblocks)
        return np.unique(np.array(list(m)).flatten())

    @property
    def cell_fields(self) -> Iterable[str]:
        """
        Returns the fields of all the celldata of the object.

        Returns
        -------
        numpy.ndarray
            NumPy array of data keys.
        """
        cellblocks = list(self.cellblocks())
        m = map(lambda cb: cb.celldata.fields, cellblocks)
        return np.unique(np.array(list(m)).flatten())

    @property
    def frame(self) -> FrameLike:
        """Returns the frame of the underlying pointcloud."""
        result = None

        if self.pd is not None:
            if self.pd.has_x:
                result = self.pd.frame

        if result is None:
            if self.parent is not None:
                result = self.parent.frame

        # If the frame is still None, it means that the entire mesh
        # has no frame, not even the root object. In this case we assign
        # a default frame to the root.
        if result is None:
            result = CartesianFrame()

        return result

    def _reset_point_data(self):
        self.pointdata = None
        self.cell_index_manager = None

    def _reset_cell_data(self):
        self.celldata = None
        self.celltype = None

    def rewire(
        self: PolyDataLike,
        deep: bool = True,
        imap: ndarray | NoneType = None,
        invert: bool = False,
    ) -> PolyDataLike:
        """
        Rewires topology according to the index mapping of the source object.

        Parameters
        ----------
        deep: bool, Optional
            If `True`, the action propagates down. Default is True.
        imap: numpy.ndarray, Optional
            Index mapper. Either provided as a numpy array, or it gets
            fetched from the database. Default is None.
        invert: bool, Optional
            A flag to indicate wether the provided index map should be
            inverted or not. Default is False.

        Notes
        -----
        Unless node numbering was modified, subsequent executions have
        no effect after once called.

        Returns
        -------
        :class:`~sigmaepsilon.mesh.polydata.PolyData`
            Returnes the object instance for continuitation.
        """
        if not deep:
            if self.cd is not None:
                if imap is not None:
                    self.cd.rewire(imap=imap, invert=invert)
                else:
                    imap = self.source().pointdata.id
                    self.cd.rewire(imap=imap, invert=False)
        else:
            if imap is not None:
                [
                    cb.rewire(imap=imap, deep=False, invert=invert)
                    for cb in self.cellblocks(inclusive=True)
                ]
            else:
                [
                    cb.rewire(deep=False, invert=invert)
                    for cb in self.cellblocks(inclusive=True)
                ]
        return self

    def to_standard_form(
        self: PolyDataLike,
        inplace: bool = True,
        default_point_fields: dict | NoneType = None,
        default_cell_fields: dict | NoneType = None,
    ) -> PolyDataLike:
        """
        Transforms the problem to standard form, which means
        a centralized pointdata and regular cell indices.

        Notes
        -----
        Some operation might work better if the layout of the mesh
        admits the standard form.

        Parameters
        ----------
        inplace: bool, Optional
            Performs the operations inplace. Default is True.
        default_point_fields: dict, Optional
            A dictionary to define default values for missing fields
            for point related data. If not specified, the default
            is `numpy.nan`.
        default_cell_fields: dict, Optional
            A dictionary to define default values for missing fields
            for cell related data. If not specified, the default
            is `numpy.nan`.
        """
        assert self.is_root(), "This must be called on he root object!"

        if not inplace:
            return deepcopy(self).to_standard_form(inplace=True)

        # merge points and point related data
        # + decorate the points with globally unique ids
        dpf = defaultdict(lambda: np.nan)
        if isinstance(default_point_fields, dict):
            dpf.update(default_point_fields)

        pim = IndexManager()
        pointtype = self.__class__._point_class_
        pointblocks = list(self.pointblocks(inclusive=True, deep=True))
        m = map(lambda pb: pb.pointdata.fields, pointblocks)
        fields = set(np.concatenate(list(m)))
        frame = self.frame

        data = {f: [] for f in fields}
        for pb in pointblocks:
            id = pim.generate_np(len(pb.pointdata))
            pb.pointdata.id = id
            pb.pd.x = PointCloud(pb.pd.x, frame=pb.frame).show(frame)
            for f in fields:
                if f in pb.pd.fields:
                    data[f].append(pb.pointdata[f].to_numpy())
                else:
                    data[f].append(np.full(len(pb.pd), dpf[f]))

        X = np.concatenate(data.pop(PointData._dbkey_x_), axis=0)

        point_fields = {}
        for f in data.keys():
            point_fields[f] = np.concatenate(data[f], axis=0)

        self.pointdata = pointtype(
            coords=X, frame=frame, fields=point_fields, container=self
        )

        # merge cells and cell related data
        # + rewire the topology based on the ids set in the previous block
        dcf = defaultdict(lambda: np.nan)
        if isinstance(default_cell_fields, dict):
            dcf.update(default_cell_fields)

        cim = IndexManager()
        cellblocks = list(self.cellblocks(inclusive=True, deep=True))
        m = map(lambda pb: pb.celldata.fields, cellblocks)
        fields = set(np.concatenate(list(m)))
        for cb in cellblocks:
            id = cb.source().pd.id
            cb.rewire(deep=False, imap=id)
            cb.cd.id = atleast1d(cim.generate_np(len(cb.celldata)))
            for f in fields:
                if f not in cb.celldata.fields:
                    cb.celldata[f] = np.full(len(cb.cd), dcf[f])
            cb.cd.pointdata = None

        # free resources
        for pb in self.pointblocks(inclusive=False, deep=True):
            pb._reset_point_data()

        return self

    def points(
        self, *, return_inds: bool = False, from_cells: bool = False
    ) -> PointCloud:
        """
        Returns the points as a :class:`~sigmaepsilon.mesh.space.pointcloud.PointCloud` instance.

        Notes
        -----
        Opposed to :func:`coords`, which returns the coordiantes, it returns
        the points of a mesh as vectors.

        See Also
        --------
        :func:`coords`

        Returns
        -------
        :class:`~sigmaepsilon.mesh.space.pointcloud.PointCloud`
        """
        global_frame = self.root.frame

        if from_cells:
            inds_ = np.unique(self.topology())
            x, inds = self.root.points(from_cells=False, return_inds=True)
            imap = inds_to_invmap_as_dict(inds)
            inds = remap_topo_1d(inds_, imap)
            coords, inds = x[inds, :], inds_
        else:
            __cls__ = self.__class__._point_array_class_
            coords, inds = [], []
            for pb in self.pointblocks(inclusive=True):
                x = pb.pd.x
                fr = pb.frame
                i = pb.pd.id
                v = PointCloud(x, frame=fr)
                coords.append(v.show(global_frame))
                inds.append(i)

            if len(coords) == 0:  # pragma: no cover
                raise Exception("There are no points belonging to this block")

            coords = np.vstack(list(coords))
            inds = np.concatenate(inds).astype(int)

        __cls__ = self.__class__._point_array_class_
        points = __cls__(coords, frame=global_frame, inds=inds)

        if return_inds:
            return points, inds
        return points

    def coords(
        self,
        *args,
        return_inds: bool = False,
        from_cells: bool = False,
        **kwargs,
    ) -> ndarray:
        """
        Returns the coordinates as an array.

        Parameters
        ----------
        return_inds: bool, Optional
            Returns the indices of the points. Default is False.
        from_cells: bool, Optional
            If there is no pointdata attached to the current block, the
            points of the sublevels of the mesh can be gathered from cell
            information. Default is False.

        Returns
        -------
        numpy.ndarray
        """
        if return_inds:
            p, inds = self.points(return_inds=True, from_cells=from_cells)
            return p.show(*args, **kwargs), inds
        else:
            return self.points(from_cells=from_cells).show(*args, **kwargs)

    def bounds(self, *args, **kwargs) -> list:
        """
        Returns the bounds of the mesh.

        Example
        -------
        >>> from sigmaepsilon.mesh.downloads import download_stand
        >>> pd = download_stand(read=True)
        >>> bounds = pd.bounds()
        """
        c = self.coords(*args, **kwargs)
        return [minmax(c[:, 0]), minmax(c[:, 1]), minmax(c[:, 2])]

    def is_2d_mesh(self) -> bool:
        """
        Returns true if the mesh is a 2-dimensional, ie. it only contains 2 dimensional
        cells.
        """
        blocks = self.cellblocks(inclusive=True)
        m = map(lambda b: b.cd.Geometry.number_of_spatial_dimensions, blocks)
        return np.all(np.array(list(m)) == 2)

    def surface_normals(self, *args, **kwargs) -> ndarray:
        """
        Retuns the surface normals as a 2d numpy array.

        .. versionadded:: 2.3.0

        Note
        ----
        It only works in cases where the call to `surface` returns a mesh
        with a `normals` method, like a `Trimesh` instance.
        """
        return self.surface(*args, **kwargs).cd.normals()

    def surface_centers(self, *args, **kwargs) -> ndarray:
        """
        Retuns the surface centers as a 3d numpy array.

        .. versionadded:: 2.3.0

        Note
        ----
        It only works in cases where the call to `surface` returns a mesh
        with a `normals` method, like a `Trimesh` instance.
        """
        return self.surface(*args, **kwargs).centers()

    @property
    def is_surface(self: PolyDataLike) -> bool:
        blocks: Iterable[PolyData] = list(self.cellblocks(inclusive=True))
        if not len(blocks) == 1:
            return False
        cell_data: PolyCellProtocol = blocks[0].cd
        if not cell_data.Geometry.number_of_spatial_dimensions == 2:
            return False
        return True

    def surface(
        self: PolyDataLike, mesh_class: PolyDataLike | NoneType = None
    ) -> PolyDataLike:
        """
        Returns the surface of the mesh as another `PolyData` instance.

        Parameters
        ----------
        mesh_class: PolyDataLike, Optional
            The class of the resulting mesh instance.
            The default is :class:`sigmaepsilon.mesh.PolyData`.

            .. versionadded:: 2.3.0
        """
        if self.is_surface:
            return self

        if mesh_class is None:
            mesh_class = PolyData

        blocks = list(self.cellblocks(inclusive=True))
        source = self.source()
        coords = source.coords()
        frame = source.frame

        triangles = []
        for block in blocks:
            NDIM = block.celldata.Geometry.number_of_spatial_dimensions
            assert NDIM == 3, "This is only for 3d cells."
            triangles.append(block.cd.extract_surface(detach=False)[-1])
        triangles = np.vstack(triangles)

        if len(blocks) > 1:
            _, indices = np.unique(triangles, axis=0, return_index=True)
            triangles = triangles[indices]

        frames = frames_of_surfaces(coords, triangles)

        pointtype = self.__class__._point_class_
        pd = pointtype(coords=coords, frame=frame)
        cd = Triangle(topo=triangles, pointdata=pd, frames=frames)

        return mesh_class(pd, cd)

    def topology(self, *args, return_inds: bool = False, **kwargs) -> TopologyArray:
        """
        Returns the topology.

        Parameters
        ----------
        return_inds: bool, Optional
            Returns the indices of the points. Default is False.

        Returns
        -------
        :class:`~sigmaepsilon.mesh.topoarray.TopologyArray`
        """
        blocks = list(self.cellblocks(*args, inclusive=True, **kwargs))
        topo = list(map(lambda i: i.celldata.topology(), blocks))
        topo = np.vstack(topo)

        if return_inds:
            inds = list(map(lambda i: i.celldata.id, blocks))
            return topo, np.concatenate(inds)
        else:
            return topo

    def cell_indices(self) -> ndarray:
        """
        Returns the indices of the cells along the walk.
        """
        blocks = self.cellblocks(inclusive=True)
        m = map(lambda b: b.cd.id, blocks)
        return np.concatenate(list(m))

    def detach(self: PolyDataLike, nummrg: bool = False) -> PolyDataLike:
        """
        Returns a detached version of the mesh.

        Parameters
        ----------
        nummrg: bool, Optional
            If True, merges node numbering. Default is False.
        """
        s: PolyData = self.source()
        polydata = PolyData(s.pd, frame=s.frame)
        l0 = len(self.address)

        if self.celldata is not None:
            db = deepcopy(self.cd.db)
            cd = self.celltype(container=polydata, db=db)
            polydata.celldata = cd
            polydata.celltype = self.celltype

        for cb in self.cellblocks(inclusive=False):
            addr = cb.address
            if len(addr) > l0:
                db = deepcopy(cb.cd.db)
                cd = cb.celltype(container=polydata, db=db)
                assert cd is not None
                polydata[addr[l0:]] = PolyData(None, cd)
                assert polydata[addr[l0:]].celldata is not None

        if nummrg:
            polydata.nummrg()

        return polydata

    def nummrg(self: PolyDataLike) -> PolyDataLike:
        """
        Merges node numbering.
        """
        assert self.is_root(), "This must be called on he root object!"
        topo = self.topology()
        inds = np.unique(topo)
        pointtype = self.__class__._point_class_
        self.pointdata = pointtype(db=self.pd[inds])
        imap = inds_to_invmap_as_dict(self.pd.id)
        [cb.rewire(imap=imap) for cb in self.cellblocks(inclusive=True)]
        self.pointdata.id = np.arange(len(self.pd))
        return self

    def move(
        self: PolyDataLike,
        v: VectorLike,
        frame: FrameLike | NoneType = None,
        inplace: bool = True,
    ) -> PolyDataLike:
        """
        Moves and returns the object or a deep copy of it.

        Parameters
        ----------
        v: VectorLike, Optional
            A vector describing a translation.
        frame: :class:`~sigmaepsilon.math.linalg.FrameLike`, Optional
            If `v` is only an array, this can be used to specify
            a frame in which the components should be understood.
        inplace: bool, Optional
            If True, the transformation is done on the instance, otherwise
            a deep copy is created first. Default is True.

        Examples
        --------
        Download the Stanford bunny and move it along global X:

        >>> from sigmaepsilon.mesh.downloads import download_bunny
        >>> import numpy as np
        >>> bunny = download_bunny(tetra=False, read=True)
        >>> bunny.move([0.2, 0, 0])
        PolyData({5: PolyData({})})
        """
        subject = self if inplace else self.deepcopy()
        if subject.is_source():
            pc = subject.points()
            pc.move(v, frame)
            subject.pointdata.x = pc.array
        else:  # pragma: no cover
            raise Exception("This is only for blocks with a point source.")
        return subject

    def rotate(
        self: PolyDataLike, *args, inplace: bool = True, **kwargs
    ) -> PolyDataLike:
        """
        Rotates and returns the object. Positional and keyword arguments
        not listed here are forwarded to :class:`sigmaepsilon.math.linalg.frame.ReferenceFrame`

        Parameters
        ----------
        *args
            Forwarded to :class:`sigmaepsilon.math.linalg.frame.ReferenceFrame`.
        inplace: bool, Optional
            If True, the transformation is done on the instance, otherwise
            a deep copy is created first. Default is True.
        **kwargs
            Forwarded to :class:`sigmaepsilon.math.linalg.frame.ReferenceFrame`.

        Examples
        --------
        Download the Stanford bunny and rotate it about global Z with 90 degrees:

        >>> from sigmaepsilon.mesh.downloads import download_bunny
        >>> import numpy as np
        >>> bunny = download_bunny(tetra=False, read=True)
        >>> bunny.rotate("Space", [0, 0, np.pi/2], "xyz")
        PolyData({5: PolyData({})})

        """
        subject = self if inplace else self.deepcopy()

        if subject.is_source():
            pc = subject.points()
            source = subject
        else:  # pragma: no cover
            raise Exception("This is only for blocks with a point source.")

        pc.rotate(*args, **kwargs)
        subject._rotate_attached_cells_(*args, **kwargs)
        source.pointdata.x = pc.show(subject.frame)

        return subject

    def spin(self: PolyDataLike, *args, inplace: bool = True, **kwargs) -> PolyDataLike:
        """
        Like rotate, but rotation happens around centroidal axes. Positional and keyword
        arguments not listed here are forwarded to :class:`sigmaepsilon.math.linalg.frame.ReferenceFrame`

        Parameters
        ----------
        *args
            Forwarded to :class:`sigmaepsilon.math.linalg.frame.ReferenceFrame`.
        inplace: bool, Optional
            If True, the transformation is done on the instance, otherwise
            a deep copy is created first. Default is True.
        **kwargs
            Forwarded to :class:`sigmaepsilon.math.linalg.frame.ReferenceFrame`.

        Examples
        --------
        Download the Stanford bunny and spin it about global Z with 90 degrees:

        >>> from sigmaepsilon.mesh.downloads import download_bunny
        >>> import numpy as np
        >>> bunny = download_bunny(tetra=False, read=True)
        >>> bunny.spin("Space", [0, 0, np.pi/2], "xyz")
        PolyData({5: PolyData({})})

        """
        subject = self if inplace else self.deepcopy()

        if subject.is_source():
            pc = subject.points()
            source = subject
        else:  # pragma: no cover
            raise Exception("This is only for blocks with a point source.")

        center = pc.center()
        pc.centralize()
        pc.rotate(*args, **kwargs)
        pc.move(center)
        subject._rotate_attached_cells_(*args, **kwargs)
        source.pointdata.x = pc.show(subject.frame)

        return subject

    def cells_at_nodes(self, *args, **kwargs) -> Iterable:
        """
        Returns the neighbouring cells of nodes.

        Returns
        -------
        object
            Some kind of iterable, depending on the inputs.
            See the docs below for further details.

        See Also
        --------
        :func:`cells_at_nodes`
        """
        topo = self.topology()

        if isinstance(topo, TopologyArray):
            if topo.is_jagged():
                topo = topo.to_csr()
            else:
                topo = topo.to_numpy()

        return cells_at_nodes(topo, *args, **kwargs)

    def cells_around_cells(
        self, radius: float, frmt: str = "dict"
    ) -> Union[JaggedArray, csr_matrix, dict]:
        """
        Returns the neares cells to cells.

        Parameters
        ----------
        radius: float
            The influence radius of a point.
        frmt: str, Optional
            A string specifying the type of the result. Valid
            options are 'jagged', 'csr' and 'dict'.

        See Also
        --------
        :func:`cells_around`
        """
        return cells_around(self.centers(), radius, frmt=frmt)

    def nodal_adjacency(self, *args, **kwargs) -> Any:
        """
        Returns the nodal adjecency matrix.

        Parameters
        ----------
        frmt: str
            A string specifying the output format. Valid options are
            'jagged', 'csr', 'nx' and 'scipy-csr'. See below for the details on the
            returned object.
        assume_regular: bool
            If the topology is regular, you can gain some speed with providing
            it as `True`. Default is `False`.
        """
        # FIXME This doesn't work with Awkward arrays.
        # topo = self.topology(jagged=True).to_array()
        topo = self.topology(jagged=True).to_numpy()

        if isinstance(topo, ak.Array):
            topo = ak.values_astype(topo, "int64")
        else:
            assert isinstance(topo, ndarray)
            topo = topo.astype(np.int64)

        return nodal_adjacency(topo, *args, **kwargs)

    def nodal_adjacency_matrix(self, assume_regular: bool = False) -> spmatrix:
        """
        Returns the nodal adjecency information as a SciPy CSR matrix.

        Parameters
        ----------
        assume_regular: bool
            If the topology is regular, you can gain some speed with providing
            it as `True`. Default is `False`.

            .. versionadded:: 2.3.0

        Returns
        -------
        scipy.sparse.spmatrix
        """
        return self.nodal_adjacency(frmt="scipy-csr", assume_regular=assume_regular)

    def nodal_neighbourhood_matrix(self) -> csr_scipy:
        """
        Returns a sparse SciPy CSR matrix as a representation of the first order
        neighbourhood structure of the mesh.

        The [i, j] entry of the returned matrix is 1 if points i and j are
        neighbours (they share a cell) 0 if they are not. Points are not considered
        to be neighbours of themselfes, therefore entries in the main diagonal are zero.

        .. versionadded:: 2.3.0
        """
        adj: spmatrix = self.nodal_adjacency_matrix()
        adj_csr = csr_scipy(adj - scipy_diags(adj.diagonal()))
        adj_csr.sum_duplicates()
        adj_csr.eliminate_zeros()
        return adj_csr

    def number_of_cells(self) -> int:
        """Returns the number of cells."""
        blocks = self.cellblocks(inclusive=True)
        return np.sum(list(map(lambda i: len(i.celldata), blocks)))

    def number_of_points(self) -> int:
        """Returns the number of points."""
        return len(self.source().pointdata)

    def cells_coords(self) -> ndarray:
        """Returns the coordiantes of the cells in explicit format."""
        return cells_coords(self.source().coords(), self.topology().to_numpy())

    def center(self, target: FrameLike | NoneType = None) -> ndarray:
        """
        Returns the center of the pointcloud of the mesh.

        Parameters
        ----------
        target: :class:`~sigmaepsilon.math.linalg.FrameLike`, Optional
            The target frame in which the returned coordinates are to be understood.
            A `None` value means the frame the mesh is embedded in. Default is None.

        Returns
        -------
        numpy.ndarray
            A one dimensional float array.
        """
        centers = self.centers(target)
        return np.array(
            [np.mean(centers[:, i]) for i in range(centers.shape[1])],
            dtype=centers.dtype,
        )

    def centers(self, target: FrameLike | NoneType = None) -> ndarray:
        """
        Returns the centers of the cells.

        Parameters
        ----------
        target: :class:`~sigmaepsilon.math.linalg.FrameLike`, Optional
            The target frame in which the returned coordinates are to be understood.
            A `None` value means the frame the mesh is embedded in. Default is None.

        Returns
        -------
        numpy.ndarray
            A 2 dimensional float array.
        """
        source = self.source()
        coords = source.coords()
        blocks = self.cellblocks(inclusive=True)

        def foo(b: PolyData[PointData, PolyCell]):
            t = b.cd.topology().to_numpy()
            return cell_centers_bulk(coords, t)

        centers = np.vstack(list(map(foo, blocks)))

        if target:
            pc = PointCloud(centers, frame=source.frame)
            centers = pc.show(target)

        return centers

    def centralize(
        self: PolyDataLike,
        target: FrameLike | NoneType = None,
        inplace: bool = True,
        axes: Iterable[int] | NoneType = None,
    ) -> PolyDataLike:
        """
        Moves all the meshes that belong to the same source such that the current object's
        center will be at the origin of its embedding frame.

        Parameters
        ----------
        target: :class:`~sigmaepsilon.math.linalg.FrameLike`, Optional
            The target frame the mesh should be central to. A `None` value
            means the frame the mesh is embedded in. Default is True.
        inplace: bool, Optional
            If True, the transformation is done on the instance, otherwise
            a deep copy is created first. Default is True.
        axes: Iterable[int], Optional
            The axes on which centralization is to be performed. A `None` value
            means all axes. For instance providing `axes=[2]` would only centralize
            coordinates in Z direction. Default is None.

        Notes
        -----
        This operation changes the coordinates of all blocks that belong to the same
        pointcloud as the object the function is called on.
        """
        subject = self if inplace else self.deepcopy()
        source = subject.source()
        target = source.frame if target is None else target
        center = self.center(target)
        if axes is not None:
            all_axes = set([0, 1, 2])
            input_axes = set(axes)
            missing_axes = list(all_axes - input_axes)
            center[missing_axes] = 0.0
        for block in source.pointblocks(inclusive=True):
            block_points = block.pd.x
            block.pd.x = block_points - center
        return subject

    def k_nearest_cell_neighbours(
        self, k, *args, knn_options: dict | NoneType = None, **kwargs
    ):
        """
        Returns the k closest neighbours of the cells of the mesh, based
        on the centers of each cell.

        The argument `knn_options` is passed to the KNN search algorithm,
        the rest to the `centers` function of the mesh.

        Examples
        --------
        >>> from sigmaepsilon.mesh.grid import grid
        >>> from sigmaepsilon.mesh import KNN
        >>> size = 80, 60, 20
        >>> shape = 10, 8, 4
        >>> X, _ = grid(size=size, shape=shape, eshape='H8')
        >>> i = KNN(X, X, k=3, max_distance=10.0)

        See Also
        --------
        :func:`KNN`
        """
        c = self.centers(*args, **kwargs)
        knn_options = {} if knn_options is None else knn_options
        return KNN(c, c, k=k, **knn_options)

    def areas(self, *args, **kwargs) -> ndarray:
        """Returns the areas."""
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        blocks2d = filter(
            lambda b: b.celltype.Geometry.number_of_spatial_dimensions < 3, blocks
        )
        amap = map(lambda b: b.celldata.areas(), blocks2d)
        return np.concatenate(list(amap))

    def area(self, *args, **kwargs) -> float:
        """Returns the sum of areas in the model."""
        return np.sum(self.areas(*args, **kwargs))

    def volumes(self, *args, **kwargs) -> ndarray:
        """Returns the volumes of the cells."""
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        vmap = map(lambda b: b.celldata.volumes(), blocks)
        return np.concatenate(list(vmap))

    def volume(self, *args, **kwargs) -> float:
        """Returns the net volume of the mesh."""
        return np.sum(self.volumes(*args, **kwargs))

    def index_of_closest_point(self, target: Iterable) -> int:
        """Returns the index of the closest point to a target."""
        return index_of_closest_point(self.coords(), np.array(target, dtype=float))

    def index_of_furthest_point(self, target: Iterable) -> int:
        """
        Returns the index of the furthest point to a target.
        """
        return index_of_furthest_point(self.coords(), np.array(target, dtype=float))

    def index_of_closest_cell(self, target: Iterable) -> int:
        """Returns the index of the closest cell to a target."""
        return index_of_closest_point(self.centers(), np.array(target, dtype=float))

    def index_of_furthest_cell(self, target: Iterable) -> int:
        """
        Returns the index of the furthest cell to a target.
        """
        return index_of_furthest_point(self.centers(), np.array(target, dtype=float))

    def nodal_distribution_factors(
        self, weights: str | ndarray = "volume"
    ) -> ndarray | csr_matrix:
        """
        Retruns nodal distribution factors for all nodes of all cells
        as a 2d array. The returned array has the same shape as the
        topology array, where the j-th factor of the i-th row is the
        contribution of element i to the j-th node of the cell.

        Parameters
        ----------
        weights: Union[str, numpy.ndarray], Optional
            The metric which is used to calculate the factors. Valid
            strings are 'volume' and 'uniform'. If it is an array, it
            must be an 1d array with a length matching the number of
            cells. Default is 'volume'.

        Returns
        -------
        numpy.ndarray or sigmaepsilon.math.linalg.sparse.csr.csr_matrix
            An array with the same shape as the topology.

        Note
        ----
        For a given node, the sum of all contribution factors from all
        the cells that meet at that node is one.

        See also
        --------
        :func:`~sigmaepsilon.mesh.utils.utils.nodal_distribution_factors`
        """
        assert self.is_source(), "This can only be called on objects with PointData."
        topo = self.topology()

        if isinstance(topo, TopologyArray):
            if topo.is_jagged():
                topo = topo.to_csr()
            else:
                topo = topo.to_numpy()

        if isinstance(weights, str):
            if weights == "volume":
                weights = self.volumes()
            elif weights == "uniform":
                weights = np.ones(topo.shape[0], dtype=float)

        assert isinstance(weights, ndarray), "'weights' must be a NumPy array!"
        assert len(weights) == topo.shape[0], (
            "Mismatch in shape. The weights must have the same number of "
            + "values as cells in the block."
        )

        return nodal_distribution_factors(topo, weights)

    def _rotate_attached_cells_(self, *args, **kwargs):
        for block in self.cellblocks(inclusive=True):
            block.cd._rotate_(*args, **kwargs)

    def _in_all_pointdata_(self, key: str) -> bool:
        blocks = self.pointblocks(inclusive=True)
        return all(list(map(lambda b: key in b.pointdata.db.fields, blocks)))

    def _in_all_celldata_(self, key: str) -> bool:
        blocks = self.cellblocks(inclusive=True)
        return all(list(map(lambda b: key in b.celldata.db.fields, blocks)))

    def _detach_block_data_(self, data: str | ndarray | NoneType = None) -> Iterator:
        blocks = self.cellblocks(inclusive=True, deep=True)
        for block in blocks:
            source = block.source()
            coords = source.coords()
            topo = block.topology().to_numpy()

            point_data = None
            if isinstance(data, ndarray):
                if not data.shape[0] == len(source.pd):
                    raise ValueError(
                        "The length of scalars must match the number of points."
                    )
                point_data = data
            elif isinstance(data, str):
                if data in source.pd.fields:
                    point_data = source.pd.db[data].to_numpy()
            else:
                if data is not None:
                    if not isinstance(data, str):
                        raise TypeError("Data must be a NumPy array or a string.")

            if point_data is not None:
                c, d, t = detach_mesh_data_bulk(coords, topo, point_data)
                yield block, c, t, d
            else:
                c, t = detach_mesh_bulk(coords, topo)
                if data is not None:
                    if data in block.cd.fields:
                        d = block.cd.db[data].to_numpy()
                        if len(d.shape) == 2:
                            c, t, d = explode_mesh_data_bulk(c, t, d)
                        else:
                            assert len(d.shape) == 1, "Cell data must be 1d or 2d."
                        yield block, c, t, d
                    else:
                        yield block, c, t, None
                else:
                    yield block, c, t, None

    def _has_plot_scalars_(self, scalars: str | ndarray | NoneType) -> list:
        """
        Returns a boolean value for every cell block in the mesh. A value
        for a block is True, if data is provided for plotting. If 'scalars'
        is a NumPy array, it is assumed that the mesh is centralized an therefore
        all values are True. Otherwise the data key must be a sting and if data is found
        in a blocks database or in the database of the related source, the value is True.
        """
        res = []
        for block in self.cellblocks(inclusive=True, deep=True):
            if isinstance(scalars, ndarray):
                res.append(True)
            elif isinstance(scalars, str):
                if block.source(scalars) is not None:
                    res.append(True)
                elif scalars in block.cd.fields:
                    res.append(True)
                else:
                    res.append(False)
            elif scalars is None:
                res.append(False)
            else:
                raise ValueError("'scalars' must be a string or a NumPy array")
        return res

    def _get_config_(self, key: str) -> dict:
        if key in self.config:
            return self.config[key]
        else:
            if self.parent is not None:
                return self.parent._get_config_(key)
            else:
                return {}

    if __hasvtk__:
        import vtk

        def to_vtk(
            self, deepcopy: bool = False, multiblock: bool = False
        ) -> vtk.vtkUnstructuredGrid | vtk.vtkMultiBlockDataSet:
            """
            Returns the mesh as a `VTK` object.

            Parameters
            ----------
            deepcopy: bool, Optional
                Default is False.
            multiblock: bool, Optional
                Wether to return the blocks as a `vtkMultiBlockDataSet` or a list
                of `vtkUnstructuredGrid` instances. Default is False.

            Returns
            -------
            vtk.vtkUnstructuredGrid or vtk.vtkMultiBlockDataSet
            """
            exporter: Callable = exporters["vtk"]
            return exporter(self, deepcopy=deepcopy, multiblock=multiblock)

    if __hasvtk__ and __haspyvista__:
        import vtk
        import pyvista as pv

        def to_pv(
            self,
            deepcopy: bool = False,
            multiblock: bool = False,
            scalars: str | ndarray | NoneType = None,
        ) -> pv.UnstructuredGrid | pv.MultiBlock:
            """
            Returns the mesh as a `PyVista` object, optionally set up with data.

            Parameters
            ----------
            deepcopy: bool, Optional
                Default is False.
            multiblock: bool, Optional
                Wether to return the blocks as a `vtkMultiBlockDataSet` or a list
                of `vtkUnstructuredGrid` instances. Default is False.
            scalars: str or numpy.ndarray, Optional
                A string or an array describing scalar data. Default is None.

            Returns
            -------
            pyvista.UnstructuredGrid or pyvista.MultiBlock
            """
            exporter: Callable = exporters["PyVista"]
            return exporter(
                self, deepcopy=deepcopy, multiblock=multiblock, scalars=scalars
            )

    if __hask3d__:
        import k3d

        def to_k3d(self, *args, **kwargs) -> k3d.Plot:
            """
            Returns the mesh as a k3d mesh object. All arguments are forwarded to
            :func:~`sigmaepsilon.mesh.io.to_k3d.to_k3d`, refer to its documentation
            for the details.

            :: warning:
                Calling this method raises a UserWarning inside the `traittypes`
                package saying "Given trait value dtype 'float32' does not match
                required type 'float32'." However, plotting seems to be fine.

            Returns
            -------
            k3d.Plot
                A K3D Plot Widget, which is a result of a call to `k3d.plot`.
            """
            exporter: Callable = exporters["k3d"]
            return exporter(self, *args, **kwargs)

        def k3dplot(self, *args, **kwargs) -> k3d.Plot:
            """
            Convenience function for plotting the mesh using K3D. All arguments are
            forwarded to :func:~`sigmaepsilon.mesh.plotting.k3dplot.k3dplot`, refer the
            documentation of this function for the details.

            .. warning::
                During this call there is a UserWarning saying 'Given trait value dtype
                "float32" does not match required type "float32"'. Although this is weird,
                plotting seems to be just fine.

            Returns
            -------
            k3d.Plot
                A K3D Plot Widget, which is a result of a call to `k3d.plot`.

            See Also
            --------
            :func:`to_k3d`
            :func:`k3d.plot`
            """
            plotter: Callable = plotters["k3d"]
            return plotter(self, *args, **kwargs)

    if __haspyvista__:
        import pyvista as pv

        def pvplot(self, *args, **kwargs) -> NoneType | pv.Plotter | ndarray:
            """
            Convenience function for plotting the mesh using PyVista. All arguments are
            forwarded to :func:~`sigmaepsilon.mesh.plotting.pvplot.pvplot`, refer the
            documentation of this function for the details.

            .. note::
                See https://github.com/pyvista/pyvista for more examples with
                ``pyvista``. Be sure to install ``pyvista`` with ``pip install
                pyvista`` if you wish to use it.

            Returns
            -------
            Union[None, pv.Plotter, numpy.ndarray]
                A PyVista plotter if `return_plotter` is `True`, a NumPy array if
                `return_img` is `True`, or nothing.

            See Also
            --------
            :func:`to_pv`
            :func:`to_vtk`
            """
            plotter: Callable = plotters["PyVista"]
            return plotter(self, *args, **kwargs)

    def plot(
        self,
        *,
        notebook: bool = False,
        backend: str = "pyvista",
        config_key: str | NoneType = None,
        **kwargs,
    ) -> Any:
        """
        Plots the mesh using supported backends. The default backend is PyVista.

        Parameters
        ----------
        notebook: bool, Optional
            Whether to plot in an IPython notebook enviroment. This is only
            available for PyVista at the moment. Default is False.
        backend: str, Optional
            The backend to use. Valid options are 'k3d' and 'pyvista'.
            Default is 'pyvista'.
        config_key: str, Optional
            A configuration key if the block were configured previously.
            Default is None.
        **kwargs: dict, Optional
            Extra keyword arguments forwarded to the plotter function according
            to the selected backend.

        See Also
        --------
        :func:`pvplot`
        :func:`k3dplot`
        """
        backend = backend.lower()
        if notebook and backend == "k3d":
            return self.k3dplot(config_key=config_key, **kwargs)
        elif backend == "pyvista":
            return self.pvplot(notebook=notebook, config_key=config_key, **kwargs)

    def __join_parent__(
        self, parent: DeepDict, key: Hashable | NoneType = None
    ) -> NoneType:
        super().__join_parent__(parent, key)
        if self.celldata is not None:
            GIDs = self.root.cim.generate_np(len(self.celldata.db))
            self.celldata.db.id = atleast1d(GIDs)
            if self.celldata.pd is None:
                self.celldata.pd = self.source().pd
            self.celldata.container = self

    def __leave_parent__(self) -> NoneType:
        if self.celldata is not None:
            self.root.cim.recycle(self.celldata.db.id)
            dbkey = self.celldata.db._dbkey_id_
            del self.celldata.db._wrapped[dbkey]
        super().__leave_parent__()

    def __repr__(self):
        return "PolyData(%s)" % (dict.__repr__(self))
