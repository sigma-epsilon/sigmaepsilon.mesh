from copy import copy, deepcopy
from typing import Union, Hashable, Collection, Iterable, Tuple, Any
from collections import defaultdict
import functools
from functools import partial
import warnings

from numpy import ndarray
import numpy as np
from scipy.sparse import spmatrix
import awkward as ak
from meshio import Mesh as MeshioMesh

from sigmaepsilon.core.warning import SigmaEpsilonPerformanceWarning
from linkeddeepdict import DeepDict
from sigmaepsilon.math.linalg.sparse import csr_matrix
from sigmaepsilon.math.linalg import Vector, ReferenceFrame as FrameLike
from sigmaepsilon.math import atleast1d, minmax, repeat

from .akwrap import AkWrapper
from .utils.topology.topo import inds_to_invmap_as_dict, remap_topo_1d
from .space import CartesianFrame, PointCloud
from .utils.utils import (
    cells_coords,
    cells_around,
    cell_centers_bulk,
    explode_mesh_data_bulk,
    nodal_distribution_factors,
)
from .utils.knn import k_nearest_neighbours as KNN
from .vtkutils import mesh_to_UnstructuredGrid as mesh_to_vtk
from .cells import (
    L2 as Line,
    T3 as Triangle,
    Q4 as Quadrilateral,
    H8 as Hexahedron,
    H27 as TriquadraticHexaHedron,
    Q9,
    TET10,
    W6,
    W18,
)
from .utils.space import (
    index_of_closest_point,
    index_of_furthest_point,
    frames_of_surfaces,
    frames_of_lines,
)
from .utils.topology import (
    nodal_adjacency,
    detach_mesh_data_bulk,
    detach_mesh_bulk,
    cells_at_nodes,
)
from .topoarray import TopologyArray
from .pointdata import PointData
from .celldata import CellData
from .base import PolyDataBase
from .cells.base.cell import PolyCell
from .helpers import meshio_to_celltype, vtk_to_celltype
from .vtkutils import PolyData_to_mesh
from .config import __hasvtk__, __haspyvista__, __hask3d__, __hasmatplotlib__

if __hasvtk__:
    import vtk

if __hask3d__:
    import k3d

if __hasmatplotlib__:
    import matplotlib as mpl

NoneType = type(None)

if __haspyvista__:
    import pyvista as pv
    from pyvista import themes

    pyVistaLike = Union[pv.PolyData, pv.PointGrid, pv.UnstructuredGrid]
else:
    pyVistaLike = NoneType


VectorLike = Union[Vector, ndarray]

__all__ = ["PolyData"]


class PolyData(PolyDataBase):
    """
    A class to handle complex polygonal meshes.

    The `PolyData` class is the most important class in the library
    and a backbone of all mesh classes.

    The implementation is based on the `awkward` library, which provides
    memory-efficient, numba-jittable data classes to deal with dense, sparse,
    complete or incomplete data. These data structures are managed in pure
    Python by the `DeepDict` class.

    The class accepts several kinds of inputs, allowing for a wide range of
    possible use cases. The fastes way to create a PolyData is from predefined
    `PointData` and `CellData` instances, defined separately.

    Parameters
    ----------
    pd: PointData or CellData, Optional
        A PolyData or a CellData instance. Dafault is None.
    cd: CellData, Optional
        A CellData instance, if the first argument is provided. Dafault is None.
    celltype: int, Optional
        An integer spcifying a valid celltype.

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
    >>> pd = PointData(coords=coords, frame=frame)
    >>> cd = H27(topo=topo, frames=frame)
    >>> mesh = PolyData(pd, frame=frame)
    >>> mesh['A']['Part1'] = PolyData(cd=H27(topo=topo[:10], frames=frame))
    >>> mesh['A']['Part2'] = PolyData(cd=H27(topo=topo[10:-10], frames=frame))
    >>> mesh['A']['Part3'] = PolyData(cd=H27(topo=topo[-10:], frames=frame))
    >>> mesh.plot()

    Load a mesh from a PyVista object:

    >>> from pyvista import examples
    >>> from sigmaepsilon.mesh import PolyData
    >>> bunny = examples.download_bunny_coarse()
    >>> mesh = PolyData.from_pv(bunny)

    Read from a .vtk file:

    >>> from sigmaepsilon.mesh import PolyData
    >>> from sigmaepsilon.core.downloads import download_stand
    >>> vtkpath = download_stand()
    >>> mesh = PolyData.read(vtkpath)

    See also
    --------
    :class:`~sigmaepsilon.mesh.trimesh.TriMesh`
    :class:`~sigmaepsilon.mesh.pointdata.PointData`
    :class:`~sigmaepsilon.mesh.celldata.CellData`
    """

    _point_array_class_ = PointCloud
    _point_class_ = PointData
    _frame_class_ = CartesianFrame
    _cell_classes_ = {
        2: Line,
        3: Triangle,
        4: Quadrilateral,
        6: W6,
        8: Hexahedron,
        9: Q9,
        10: TET10,
        18: W18,
        27: TriquadraticHexaHedron,
    }
    _pv_config_key_ = ("pv", "default")
    _k3d_config_key_ = ("k3d", "default")

    def __init__(
        self,
        pd: Union[PointData, CellData] = None,
        cd: CellData = None,
        *args,
        coords: ndarray = None,
        topo: ndarray = None,
        celltype=None,
        frame: FrameLike = None,
        newaxis: int = 2,
        cell_fields: dict = None,
        point_fields: dict = None,
        parent: "PolyData" = None,
        frames: Union[FrameLike, ndarray] = None,
        **kwargs,
    ):
        self._reset_point_data()
        self._reset_cell_data()
        self._frame = frame
        self._newaxis = newaxis
        self._parent = parent
        self._config = DeepDict()
        self._cid2bid = None  # maps cell indices to block indices
        self._bid2b = None  # maps block indices to block addresses
        self._init_config_()

        self._pointdata = None
        self._celldata = None

        self.point_index_manager = IndexManager()
        self.cell_index_manager = IndexManager()

        if isinstance(pd, PointData):
            self.pointdata = pd
            if isinstance(cd, CellData):
                self.celldata = cd
        elif isinstance(pd, CellData):
            self.celldata = pd
            if isinstance(cd, PointData):
                self.pointdata = cd
        elif isinstance(cd, CellData):
            self.celldata = cd

        pidkey = self.__class__._point_class_._dbkey_id_
        cidkey = CellData._dbkey_id_

        if self.pointdata is not None:
            if self.pd.has_id:
                if self.celldata is not None:
                    imap = self.pd.id
                    self.cd.rewire(imap=imap, invert=True)
            N = len(self.pointdata)
            GIDs = self.root().pim.generate_np(N)
            self.pd[pidkey] = GIDs
            self.pd.container = self

        if self.celldata is not None:
            N = len(self.celldata)
            GIDs = self.root().cim.generate_np(N)
            self.cd[cidkey] = GIDs
            try:
                pd = self.source().pd
            except Exception:
                pd = None
            self.cd.pd = pd
            self.cd.container = self

        super().__init__(*args, **kwargs)

        if self.pointdata is None and coords is not None:
            point_fields = {} if point_fields is None else point_fields
            pointtype = self.__class__._point_class_
            GIDs = self.root().pim.generate_np(coords.shape[0])
            point_fields[pidkey] = GIDs
            self.pointdata = pointtype(
                coords=coords,
                frame=frame,
                newaxis=newaxis,
                stateful=True,
                fields=point_fields,
            )

        if self.celldata is None and topo is not None:
            cell_fields = {} if cell_fields is None else cell_fields
            if celltype is None:
                celltype = self.__class__._cell_classes_.get(topo.shape[1], None)
            elif isinstance(celltype, int):
                raise NotImplementedError
            if not issubclass(celltype, CellData):
                raise TypeError("Invalid cell type <{}>".format(celltype))

            if isinstance(topo, np.ndarray):
                topo = topo.astype(int)
            else:
                raise TypeError("Topo must be an 1d array of integers.")

            if frames is not None:
                if isinstance(frames, FrameLike):
                    cell_fields["frames"] = repeat(frames.show(), topo.shape[0])
                elif isinstance(frames, ndarray):
                    if len(frames.shape) == 2:
                        cell_fields["frames"] = repeat(frames, topo.shape[0])
                    else:
                        assert (
                            len(frames.shape) == 3
                        ), "'frames' must be a 2d or 3d array."
                        cell_fields["frames"] = frames
            elif isinstance(frame, FrameLike):
                cell_fields["frames"] = repeat(frame.show(), topo.shape[0])

            GIDs = self.root().cim.generate_np(topo.shape[0])
            cell_fields[cidkey] = GIDs
            try:
                pd = self.source().pointdata
            except Exception:
                pd = None
            self.celldata = celltype(topo, fields=cell_fields, pointdata=pd)

        if self.celldata is not None:
            self.celltype = self.celldata.__class__
            self.celldata.container = self

    def __deepcopy__(self, memo):
        return self.__copy__(memo)

    def __copy__(self, memo=None):
        cls = type(self)
        copy_function = copy if (memo is None) else partial(deepcopy, memo=memo)
        is_deep = memo is not None
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

    def copy(self) -> "PolyData":
        return copy(self)

    def deepcopy(self) -> "PolyData":
        return deepcopy(self)

    def __getitem__(self, key) -> "PolyData":
        return super().__getitem__(key)

    @property
    def pointdata(self) -> PointData:
        """
        Returns the attached pointdata.
        """
        return self._pointdata

    @pointdata.setter
    def pointdata(self, pd: Union[PointData, None]) -> None:
        """
        Returns the attached pointdata.
        """
        if pd is not None and not isinstance(pd, PointData):
            raise TypeError("Value must be a PointData instance.")
        self._pointdata = pd
        if isinstance(pd, PointData):
            self._pointdata.container = self

    @property
    def pd(self) -> PointData:
        """
        Returns the attached pointdata.
        """
        return self.pointdata

    @property
    def celldata(self) -> PolyCell:
        """
        Returns the attached celldata.
        """
        return self._celldata

    @celldata.setter
    def celldata(self, cd: Union[PolyCell, None]) -> None:
        """
        Returns the attached celldata.
        """
        if cd is not None and not isinstance(cd, PolyCell):
            raise TypeError("Value must be a PolyCell instance.")
        self._celldata = cd
        if isinstance(cd, PolyCell):
            self._celldata.container = self

    @property
    def cd(self) -> PolyCell:
        """
        Returns the attached celldata.
        """
        return self.celldata

    def lock(self, create_mappers: bool = False) -> "PolyData":
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

    def unlock(self) -> "PolyData":
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

    def blocks_of_cells(self, i: Union[int, Iterable] = None) -> dict:
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
    def read(cls, *args, **kwargs) -> "PolyData":
        """
        Reads from a file using PyVista.

        Example
        -------
        Download a .vtk file and read it:

        >>> from sigmaepsilon.mesh import PolyData
        >>> from sigmaepsilon.mesh.examples import stand_vtk
        >>> vtkpath = stand_vtk(read=False)
        >>> mesh = PolyData.read(vtkpath)
        """
        return cls.from_pv(pv.read(*args, **kwargs))

    @classmethod
    def from_meshio(cls, mesh: MeshioMesh) -> "PolyData":
        """
        Returns a :class:`~sigmaepsilon.mesh.polydata.PolyData` instance from a :class:`meshio.Mesh` instance.
        """
        GlobalFrame = CartesianFrame(dim=3)

        coords = mesh.points
        polydata = PolyData(coords=coords, frame=GlobalFrame)

        for cb in mesh.cells:
            cd = None
            cbtype = cb.type
            celltype: PolyCell = meshio_to_celltype.get(cbtype, None)
            if celltype:
                topo = np.array(cb.data, dtype=int)

                NDIM = celltype.NDIM
                if NDIM == 1:
                    frames = frames_of_lines(coords, topo)
                elif NDIM == 2:
                    frames = frames_of_surfaces(coords, topo)
                elif NDIM == 3:
                    frames = GlobalFrame

                cd = celltype(topo=topo, frames=frames)
                polydata[cbtype] = PolyData(cd, frame=GlobalFrame)
            else:
                pass
                # FIXME needs warning and logging
                # msg = f"Cells of type '{cbtype}' are not supported here."
                # raise NotImplementedError(msg)

        return polydata

    @classmethod
    def from_pv(cls, pvobj: pyVistaLike) -> "PolyData":
        """
        Returns a :class:`~sigmaepsilon.mesh.polydata.PolyData` instance from
        a :class:`pyvista.PolyData` or a :class:`pyvista.UnstructuredGrid`
        instance.

        Example
        -------
        >>> from pyvista import examples
        >>> from sigmaepsilon.mesh import PolyData
        >>> bunny = examples.download_bunny_coarse()
        >>> mesh = PolyData.from_pv(bunny)
        """
        if isinstance(pvobj, pv.PolyData):
            coords, topo = PolyData_to_mesh(pvobj)
            if isinstance(topo, dict):
                cells_dict = topo
            elif isinstance(topo, np.ndarray):
                assert isinstance(cls._cell_classes_, dict)
                ct = cls._cell_classes_[topo.shape[-1]]
                cells_dict = {ct.vtkCellType: topo}
        elif isinstance(pvobj, pv.UnstructuredGrid):
            coords = pvobj.points.astype(float)
            cells_dict = pvobj.cells_dict
        elif isinstance(pvobj, pv.PointGrid):
            ugrid = pvobj.cast_to_unstructured_grid()
            coords = pvobj.points.astype(float)
            cells_dict = ugrid.cells_dict
        else:
            try:
                ugrid = pvobj.cast_to_unstructured_grid()
                return PolyData.from_pv(ugrid)
            except Exception:
                raise TypeError("Invalid inut type!")

        GlobalFrame = CartesianFrame(dim=3)
        pd = PolyData(coords=coords, frame=GlobalFrame)  # this fails without a frame

        for vtkid, vtktopo in cells_dict.items():
            if vtkid in vtk_to_celltype:
                celltype = vtk_to_celltype[vtkid]

                NDIM = celltype.NDIM
                if NDIM == 1:
                    frames = frames_of_lines(coords, vtktopo)
                elif NDIM == 2:
                    frames = frames_of_surfaces(coords, vtktopo)
                elif NDIM == 3:
                    frames = GlobalFrame

                cd = celltype(topo=vtktopo, frames=frames)
                pd[vtkid] = PolyData(cd, frame=GlobalFrame)
            else:
                msg = "The element type with vtkId <{}> is not jet " + "supported here."
                raise NotImplementedError(msg.format(vtkid))

        return pd

    def to_dataframe(
        self,
        *args,
        point_fields: Iterable[str] = None,
        cell_fields: Iterable[str] = None,
        **kwargs,
    ):
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
        >>> from sigmaepsilon.mesh.examples import stand_vtk
        >>> mesh = stand_vtk(read=True)
        >>> mesh.to_dataframe(point_fields=['x'])
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
        point_fields: Iterable[str] = None,
        cell_fields: Iterable[str] = None,
        **kwargs,
    ):
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
        >>> from sigmaepsilon.mesh.examples import stand_vtk
        >>> mesh = stand_vtk(read=True)
        >>> mesh.to_parquet('pd.parquet', 'cd.parquet', point_fields=['x'])
        """
        ak_pd, ak_cd = self.to_ak(
            *args, point_fields=point_fields, cell_fields=cell_fields
        )
        ak.to_parquet(ak_pd, path_pd, **kwargs)
        ak.to_parquet(ak_cd, path_cd, **kwargs)

    def to_ak(
        self,
        *args,
        point_fields: Iterable[str] = None,
        cell_fields: Iterable[str] = None,
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
        >>> from sigmaepsilon.mesh.examples import stand_vtk
        >>> mesh = stand_vtk(read=True)
        >>> mesh.to_ak(point_fields=['x'])
        """
        lp, lc = self.to_lists(
            *args, point_fields=point_fields, cell_fields=cell_fields
        )
        return ak.from_iter(lp), ak.from_iter(lc)

    def to_lists(
        self, *, point_fields: Iterable[str] = None, cell_fields: Iterable[str] = None
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
        >>> from sigmaepsilon.mesh.examples import stand_vtk
        >>> mesh = stand_vtk(read=True)
        >>> mesh.to_lists(point_fields=['x'])
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
        >>> from sigmaepsilon.mesh.examples import download_stand
        >>> mesh = download_stand(read=True)

        To set configuration values related to plotting with `pyVista`,
        do the following:

        >>> mesh.config['pyvista', 'plot', 'color'] = 'red'
        >>> mesh.config['pyvista', 'plot', 'style'] = 'wireframe'

        Then, when it comes to plotting, you can specify your configuration
        with the `config_key` keyword argument:

        >>> mesh.pvplot(config_key=('pyvista', 'plot'))

        This way, you can store several different configurations for
        different scenarios.
        """
        return self._config

    def _init_config_(self):
        key = self.__class__._pv_config_key_
        self.config[key]["show_edges"] = True

    @property
    def pim(self) -> "IndexManager":
        return self.point_index_manager

    @property
    def cim(self) -> "IndexManager":
        return self.cell_index_manager

    @property
    def parent(self) -> "PolyData":
        """Returns the parent of the object."""
        return self._parent

    @parent.setter
    def parent(self, value: "PolyData"):
        """Sets the parent."""
        self._parent = value

    def is_source(self, key: str = None) -> bool:
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

    def source(self, key: str = None) -> Union["PolyData", None]:
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
        self, *, inclusive: bool = False, blocktype: Any = None, deep: bool = True, **kw
    ) -> Collection["PolyData"]:
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
        Any
            A PolyData instance. The actual type depends on the 'blocktype'
            parameter.
        """
        dtype = PolyData if blocktype is None else blocktype
        return self.containers(self, inclusive=inclusive, dtype=dtype, deep=deep)

    def pointblocks(self, *args, **kwargs) -> Iterable["PolyData"]:
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
        :class:`~sigmaepsilon.mesh.pointdata.PointData`
        """
        return filter(lambda i: i.pd is not None, self.blocks(*args, **kwargs))

    def cellblocks(self, *args, **kwargs) -> Iterable["PolyData"]:
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
        :class:`~sigmaepsilon.mesh.celldata.CellData`
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
        # there is a reference in PointData to the variable `self._frame`
        result = None
        if self._frame is not None:
            result = self._frame
        elif self.pd is not None:
            if self.pd.has_x:
                result = self.pd.frame
        if result is None:
            if self.parent is not None:
                result = self.parent.frame
        if result is None:
            raise AttributeError("This instance have no attached pointcloud.")
        else:
            return result

    @property
    def frames(self) -> ndarray:
        """Returnes the frames of the cells."""
        if self.celldata is not None:
            return self.celldata.frames

    @frames.setter
    def frames(self, value: ndarray):
        """Sets the frames of the cells."""
        assert self.celldata is not None
        if isinstance(value, ndarray):
            self.celldata.frames = value
        else:
            raise TypeError(
                ("Type {} is not a supported" + " type to specify frames.").format(
                    type(value)
                )
            )

    def _reset_point_data(self):
        self.pointdata = None
        self.cell_index_manager = None

    def _reset_cell_data(self):
        self.celldata = None
        self.celltype = None

    def rewire(
        self, deep: bool = True, imap: ndarray = None, invert: bool = False
    ) -> "PolyData":
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
        self,
        inplace: bool = True,
        default_point_fields: dict = None,
        default_cell_fields: dict = None,
    ) -> "PolyData":
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
        frame, axis = self._frame, self._newaxis
        point_fields = {}
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
        for f in data.keys():
            point_fields[f] = np.concatenate(data[f], axis=0)
        self.pointdata = pointtype(
            coords=X, frame=frame, newaxis=axis, fields=point_fields, container=self
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
        :class:`~sigmaepsilon.mesh.space.pointcloud.PointCloud`
        :func:`coords`
        """
        target = self.frame
        if from_cells:
            inds_ = np.unique(self.topology())
            x, inds = self.root().points(from_cells=False, return_inds=True)
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
                v = Vector(x, frame=fr)
                coords.append(v.show(target))
                inds.append(i)
            coords = np.vstack(list(coords))
            inds = np.concatenate(inds).astype(int)
        __cls__ = self.__class__._point_array_class_
        points = __cls__(coords, frame=target, inds=inds)
        if return_inds:
            return points, inds
        return points

    def coords(
        self, *args, return_inds: bool = False, from_cells: bool = False, **kwargs
    ) -> ndarray:
        """
        Returns the coordinates as an array.

        Parameters
        ----------
        return_inds: bool, Optional
            Returns the indices of the points. Default is False.
        from_cells: bool, Optional
            If there is no pointdata attaached to the current block, the
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
        >>> from sigmaepsilon.mesh.examples import download_stand
        >>> pd = download_stand(read=True)
        >>> pd.bounds()
        """
        c = self.coords(*args, **kwargs)
        return [minmax(c[:, 0]), minmax(c[:, 1]), minmax(c[:, 2])]

    def surface(self) -> "PolyData":
        """
        Returns the surface of the mesh as another `PolyData` instance.
        """
        blocks = list(self.cellblocks(inclusive=True))
        source = self.source()
        coords = source.coords()
        frame = source.frame
        triangles = []
        for block in blocks:
            assert block.celldata.NDIM == 3, "This is only for 3d cells."
            triangles.append(block.cd.extract_surface(detach=False)[-1])
        triangles = np.vstack(triangles)
        if len(blocks) > 1:
            _, indices = np.unique(triangles, axis=0, return_index=True)
            triangles = triangles[indices]
        pointtype = self.__class__._point_class_
        pd = pointtype(coords=coords, frame=frame)
        cd = Triangle(topo=triangles, pointdata=pd)
        return self.__class__(pd, cd, frame=frame)

    def topology(
        self, *args, return_inds: bool = False, jagged: bool = None, **kwargs
    ) -> Union[ndarray, TopologyArray]:
        """
        Returns the topology.

        Parameters
        ----------
        return_inds: bool, Optional
            Returns the indices of the points. Default is False.
        jagged: bool, Optional
            If True, returns the topology as a :class:`~sigmaepsilon.mesh.topoarray.TopologyArray`
            instance, even if the mesh is regular. Default is False.

        Returns
        -------
        Union[numpy.ndarray, :class:`~sigmaepsilon.mesh.topoarray.TopologyArray`]
            The topology as a 2d integer array.
        """
        blocks = list(self.cellblocks(*args, inclusive=True, **kwargs))
        topo = list(map(lambda i: i.celldata.topology(), blocks))
        widths = np.concatenate(list(map(lambda t: t.widths(), topo)))
        jagged = False if not isinstance(jagged, bool) else jagged
        needs_jagged = not np.all(widths == widths[0])
        if jagged or needs_jagged:
            topo = np.vstack(topo)
            if return_inds:
                inds = list(map(lambda i: i.celldata.id, blocks))
                return topo, np.concatenate(inds)
            else:
                return topo
        else:
            topo = np.vstack([t.to_numpy() for t in topo])
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

    def detach(self, nummrg: bool = False) -> "PolyData":
        """
        Returns a detached version of the mesh.

        Parameters
        ----------
        nummrg: bool, Optional
            If True, merges node numbering. Default is False.
        """
        s = self.source()
        pd = PolyData(s.pd, frame=s.frame)
        l0 = len(self.address)
        if self.celldata is not None:
            db = deepcopy(self.cd.db)
            cd = self.celltype(pointdata=pd, db=db)
            pd.celldata = cd
            pd.celltype = self.celltype
        for cb in self.cellblocks(inclusive=False):
            addr = cb.address
            if len(addr) > l0:
                db = deepcopy(cb.cd.db)
                cd = cb.celltype(pointdata=pd, db=db)
                assert cd is not None
                pd[addr[l0:]] = PolyData(None, cd)
                assert pd[addr[l0:]].celldata is not None
        if nummrg:
            pd.nummrg()
        return pd

    def nummrg(self):
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
        self, v: VectorLike, frame: FrameLike = None, inplace: bool = True
    ) -> "PolyData":
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

        >>> from sigmaepsilon.mesh.examples import download_bunny
        >>> import numpy as np
        >>> bunny = download_bunny(tetra=False, read=True)
        >>> bunny.move([0.2, 0, 0])
        """
        subject = self if inplace else self.deepcopy()
        if subject.is_source():
            pc = subject.points()
            pc.move(v, frame)
            subject.pointdata.x = pc.array
        else:
            source = subject.source()
            inds = np.unique(subject.topology())
            pc = source.points()[inds]
            pc.move(v, frame)
            source.pointdata.x = pc.array
        return subject

    def rotate(self, *args, inplace: bool = True, **kwargs) -> "PolyData":
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

        >>> from sigmaepsilon.mesh.examples import download_bunny
        >>> import numpy as np
        >>> bunny = download_bunny(tetra=False, read=True)
        >>> bunny.rotate("Space", [0, 0, np.pi/2], "xyz")
        """
        subject = self if inplace else self.deepcopy()
        if subject.is_source():
            pc = subject.points()
            source = subject
        else:
            source = subject.source()
            inds = np.unique(subject.topology())
            pc = source.points()[inds]
        pc.rotate(*args, **kwargs)
        subject._rotate_attached_cells_(*args, **kwargs)
        source.pointdata.x = pc.show(subject.frame)
        return subject

    def spin(self, *args, inplace: bool = True, **kwargs) -> "PolyData":
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

        >>> from sigmaepsilon.mesh.examples import download_bunny
        >>> import numpy as np
        >>> bunny = download_bunny(tetra=False, read=True)
        >>> bunny.spin("Space", [0, 0, np.pi/2], "xyz")
        """
        subject = self if inplace else self.deepcopy()
        if subject.is_source():
            pc = subject.points()
            source = subject
        else:
            source = subject.source()
            inds = np.unique(subject.topology())
            pc = source.points()[inds]
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

    def cells_around_cells(self, radius: float, frmt: str = "dict"):
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
        Returns the nodal adjecency matrix. The arguments are
        forwarded to the corresponding utility function (see below)
        alongside the topology of the mesh as the first argument.

        See also
        --------
        :func:`~sigmaepsilon.mesh.utils.topology.topo.nodal_adjacency`
        """
        topo = self.topology(jagged=True).to_array()
        if isinstance(topo, ak.Array):
            topo = ak.values_astype(topo, "int64")
        else:
            assert isinstance(topo, ndarray)
            topo = topo.astype(np.int64)
        return nodal_adjacency(topo, *args, **kwargs)

    def nodal_adjacency_matrix(self) -> spmatrix:
        """
        Returns the nodal adjecency matrix. The arguments are
        forwarded to the corresponding utility function (see below)
        alongside the topology of the mesh as the first argument.

        See also
        --------
        :func:`~sigmaepsilon.mesh.utils.topology.topo.nodal_adjacency`

        Returns
        -------
        scipy.sparse.spmatrix
        """
        return self.nodal_adjacency(frmt="scipy-csr")

    def number_of_cells(self) -> int:
        """Returns the number of cells."""
        blocks = self.cellblocks(inclusive=True)
        return np.sum(list(map(lambda i: len(i.celldata), blocks)))

    def number_of_points(self) -> int:
        """Returns the number of points."""
        return len(self.source().pointdata)

    def cells_coords(self, *, _topo=None, **kwargs) -> ndarray:
        """Returns the coordiantes of the cells in explicit format."""
        _topo = self.topology() if _topo is None else _topo
        return cells_coords(self.source().coords(), _topo)

    def center(self, target: FrameLike = None) -> ndarray:
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
        if self.is_source():
            return self.points().center(target)
        else:
            source = self.source()
            inds = np.unique(self.topology())
            pc = source.points()[inds]
            return pc.center(target)

    def centers(self, target: FrameLike = None) -> ndarray:
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

        def foo(b: PolyData):
            t = b.cd.topology().to_numpy()
            return cell_centers_bulk(coords, t)

        centers = np.vstack(list(map(foo, blocks)))

        if target:
            pc = PointCloud(centers, frame=source.frame)
            centers = pc.show(target)

        return centers

    def centralize(
        self, target: FrameLike = None, inplace: bool = True, axes: Iterable = None
    ) -> "PolyData":
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
        axes: Iterable, Optional
            The axes on which centralization is to be performed. A `None` value
            means all axes. Default is None.

        Notes
        -----
        This operation changes the coordinates of all blocks that belong to the same
        pointcloud as the object the function is called on.
        """
        subject = self if inplace else self.deepcopy()
        source = subject.source()
        target = source.frame if target is None else target
        center = self.center(target)
        for block in source.pointblocks(inclusive=True):
            block_points = block.pd.x
            block.pd.x = block_points - center
        return subject

    def k_nearest_cell_neighbours(self, k, *args, knn_options: dict = None, **kwargs):
        """
        Returns the k closest neighbours of the cells of the mesh, based
        on the centers of each cell.

        The argument `knn_options` is passed to the KNN search algorithm,
        the rest to the `centers` function of the mesh.

        Examples
        --------
        >>> from sigmaepsilon.mesh.grid import Grid
        >>> from sigmaepsilon.mesh import KNN
        >>> size = 80, 60, 20
        >>> shape = 10, 8, 4
        >>> grid = Grid(size=size, shape=shape, eshape='H8')
        >>> X = grid.centers()
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
        blocks2d = filter(lambda b: b.celltype.NDIM < 3, blocks)
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
        return index_of_closest_point(self.coords(), target)

    def index_of_furthest_point(self, target: Iterable) -> int:
        """
        Returns the index of the furthest point to a target.
        """
        return index_of_furthest_point(self.coords(), target)

    def index_of_closest_cell(self, target: Iterable) -> int:
        """Returns the index of the closest cell to a target."""
        return index_of_closest_point(self.centers(), target)

    def index_of_furthest_cell(self, target: Iterable) -> int:
        """
        Returns the index of the furthest cell to a target.
        """
        return index_of_furthest_point(self.centers(), target)

    def nodal_distribution_factors(
        self, weights: Union[str, ndarray] = "volume"
    ) -> Union[ndarray, csr_matrix]:
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
        numpy.ndarray or csr_matrix
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
                weights = np.ones(len(topo), dtype=float)
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
        return all(list(map(lambda b: key in b.db.fields, blocks)))

    def _in_all_celldata_(self, key: str) -> bool:
        blocks = self.cellblocks(inclusive=True)
        return all(list(map(lambda b: key in b.db.fields, blocks)))

    def _detach_block_data_(self, data: Union[str, ndarray] = None) -> Tuple:
        blocks = self.cellblocks(inclusive=True, deep=True)
        for block in blocks:
            source = block.source()
            coords = source.coords()
            topo = block.topology()

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

    def _has_plot_scalars_(self, scalars: Union[str, ndarray]):
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

        def to_vtk(
            self, deepcopy: bool = False, multiblock: bool = False
        ) -> Union[vtk.vtkUnstructuredGrid, vtk.vtkMultiBlockDataSet]:
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
            if not __hasvtk__:
                raise ImportError("VTK must be installed for this!")
            ugrids = []
            for block, c, t, _ in self._detach_block_data_():
                vtkct = block.celltype.vtkCellType
                ugrid = mesh_to_vtk(c, t, vtkct, deepcopy)
                ugrids.append(ugrid)
            if multiblock:
                mb = vtk.vtkMultiBlockDataSet()
                mb.SetNumberOfBlocks(len(ugrids))
                for i, ugrid in enumerate(ugrids):
                    mb.SetBlock(i, ugrid)
                return mb
            else:
                if len(ugrids) > 1:
                    return ugrids
                else:
                    return ugrids[0]

    if __hasvtk__ and __haspyvista__:

        def to_pv(
            self,
            deepcopy: bool = False,
            multiblock: bool = False,
            scalars: Union[str, ndarray] = None,
        ) -> Union[pv.UnstructuredGrid, pv.MultiBlock]:
            """
            Returns the mesh as a `PyVista` oject, optionally set up with data.

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
            if not __hasvtk__:
                raise ImportError("VTK must be installed for this!")
            if not __haspyvista__:
                raise ImportError("PyVista must be installed for this!")
            ugrids = []
            data = []
            for block, c, t, d in self._detach_block_data_(scalars):
                vtkct = block.celltype.vtkCellType
                ugrid = mesh_to_vtk(c, t, vtkct, deepcopy)
                ugrids.append(ugrid)
                data.append(d)
            if multiblock:
                mb = vtk.vtkMultiBlockDataSet()
                mb.SetNumberOfBlocks(len(ugrids))
                for i, ugrid in enumerate(ugrids):
                    mb.SetBlock(i, ugrid)
                mb = pv.wrap(mb)
                try:
                    mb.wrap_nested()
                except AttributeError:
                    pass
                return mb
            else:
                if scalars is None:
                    return [pv.wrap(ugrid) for ugrid in ugrids]
                else:
                    res = []
                    for ugrid, d in zip(ugrids, data):
                        pvobj = pv.wrap(ugrid)
                        if isinstance(d, ndarray):
                            if isinstance(scalars, str):
                                pvobj[scalars] = d
                            else:
                                pvobj["scalars"] = d
                        res.append(pvobj)
                    return res

    if __hask3d__:

        def to_k3d(
            self,
            *,
            scene: object = None,
            deep: bool = True,
            config_key: str = None,
            menu_visibility: bool = True,
            cmap: list = None,
            show_edges: bool = True,
            scalars: ndarray = None,
        ):
            """
            Returns the mesh as a k3d mesh object.

            :: warning:
                Calling this method raises a UserWarning inside the `traittypes`
                package saying "Given trait value dtype 'float32' does not match
                required type 'float32'." However, plotting seems to be fine.

            Returns
            -------
            object
                A K3D Plot Widget, which is a result of a call to `k3d.plot`.

            See also
            --------
            :func:`k3d.lines`
            :func:`k3d.mesh`
            """
            if not __hask3d__:
                raise ImportError(
                    "The python package 'k3d' must be installed for this."
                )
            if scene is None:
                scene = k3d.plot(menu_visibility=menu_visibility)
            source = self.source()
            coords = source.coords()

            if isinstance(scalars, ndarray):
                color_range = minmax(scalars)
                color_range = [scalars.min() - 1, scalars.max() + 1]

            k3dparams = dict(wireframe=False)
            if config_key is None:
                config_key = self.__class__._k3d_config_key_

            for b in self.cellblocks(inclusive=True, deep=deep):
                params = copy(k3dparams)
                config = b._get_config_(config_key)
                params.update(config)
                if "color" in params:
                    if isinstance(params["color"], str):
                        hexstr = mpl.colors.to_hex(params["color"])
                        params["color"] = int("0x" + hexstr[1:], 16)
                if cmap is not None:
                    params["color_map"] = cmap
                if b.celltype.NDIM == 1:
                    topo = b.cd.topology().to_numpy()
                    if isinstance(scalars, ndarray):
                        c, d, t = detach_mesh_data_bulk(coords, topo, scalars)
                        params["attribute"] = d
                        params["color_range"] = color_range
                        params["indices_type"] = "segment"
                    else:
                        c, t = detach_mesh_bulk(coords, topo)
                        params["indices_type"] = "segment"
                    c = c.astype(np.float32)
                    t = t.astype(np.uint32)
                    scene += k3d.lines(c, t, **params)
                elif b.celltype.NDIM == 2:
                    topo = b.cd.to_triangles()
                    if isinstance(scalars, ndarray):
                        c, d, t = detach_mesh_data_bulk(coords, topo, scalars)
                        params["attribute"] = d
                        params["color_range"] = color_range
                    else:
                        c, t = detach_mesh_bulk(coords, topo)
                    c = c.astype(np.float32)
                    t = t.astype(np.uint32)
                    if "side" in params:
                        if params["side"].lower() == "both":
                            params["side"] = "front"
                            scene += k3d.mesh(c, t, **params)
                            params["side"] = "back"
                            scene += k3d.mesh(c, t, **params)
                        else:
                            scene += k3d.mesh(c, t, **params)
                    else:
                        scene += k3d.mesh(c, t, **params)
                    if show_edges:
                        scene += k3d.mesh(c, t, wireframe=True, color=0)
                elif b.celltype.NDIM == 3:
                    topo = b.surface().topology()
                    if isinstance(scalars, ndarray):
                        c, d, t = detach_mesh_data_bulk(coords, topo, scalars)
                        params["attribute"] = d
                        params["color_range"] = color_range
                    else:
                        c, t = detach_mesh_bulk(coords, topo)
                    c = c.astype(np.float32)
                    t = t.astype(np.uint32)
                    scene += k3d.mesh(c, t, **params)
                    if show_edges:
                        scene += k3d.mesh(c, t, wireframe=True, color=0)
            return scene

        def k3dplot(self, scene=None, *, menu_visibility: bool = True, **kwargs):
            """
            Plots the mesh using 'k3d' as the backend.

            .. warning::
                During this call there is a UserWarning saying 'Given trait value dtype
                "float32" does not match required type "float32"'. Although this is weird,
                plotting seems to be just fine.

            Parameters
            ----------
            scene: object, Optional
                A K3D plot widget to append to. This can also be given as the
                first positional argument. Default is None, in which case it is
                created using a call to :func:`k3d.plot`.
            menu_visibility: bool, Optional
                Whether to show the menu or not. Default is True.
            **kwargs: dict, Optional
                Extra keyword arguments forwarded to :func:`to_k3d`.

            See Also
            --------
            :func:`to_k3d`
            :func:`k3d.plot`
            """
            if scene is None:
                scene = k3d.plot(menu_visibility=menu_visibility)
            return self.to_k3d(scene=scene, **kwargs)

    if __haspyvista__:

        def pvplot(
            self,
            *,
            deepcopy: bool = False,
            jupyter_backend: str = "pythreejs",
            show_edges: bool = True,
            notebook: bool = False,
            theme: str = None,
            scalars: Union[str, ndarray] = None,
            window_size: Tuple = None,
            return_plotter: bool = False,
            config_key: Tuple = None,
            plotter: pv.Plotter = None,
            cmap: Union[str, Iterable] = None,
            camera_position: Tuple = None,
            lighting: bool = False,
            edge_color: str = None,
            return_img: bool = False,
            show_scalar_bar: Union[bool, None] = None,
            **kwargs,
        ) -> Union[None, pv.Plotter, np.ndarray]:
            """
            Plots the mesh using PyVista. The parameters listed here only grasp
            a fraction of what PyVista provides. The idea is to have a function
            that narrows down the parameters as much as possible to the ones that
            are most commonly used. If you want more control, create a plotter
            prior to calling this function and provide it using the parameter
            `plotter`. Then by setting `return_plotter` to `True`, the function
            adds the cells to the plotter and returns it for further customization.

            Parameters
            ----------
            deepcopy: bool, Optional
                If True, a deep copy is created. Default is False.
            jupyter_backend: str, Optional
                The backend to use when plotting in a Jupyter enviroment.
                Default is 'pythreejs'.
            show_edges: bool, Optional
                If True, the edges of the mesh are shown as a wireframe.
                Default is True.
            notebook: bool, Optional
                If True and in a Jupyter enviroment, the plot is embedded
                into the Notebook. Default is False.
            theme: str, Optional
                The theme to use with PyVista. Default is None.
            scalars: Union[str, ndarray]
                A string that refers to a field in the celldata objects
                of the block of the mesh, or a NumPy array with values for
                each point in the mesh.
            window_size: tuple, Optional
                The size of the window, only is `notebook` is `False`.
                Default is None.
            return_plotter: bool, Optional
                If True, an instance of :class:`pyvista.Plotter` is returned
                without being shown. Default is False.
            config_key: tuple, Optional
                A tuple of strings that refer to a configuration for PyVista.
            plotter: pyvista.Plotter, Optional
                A plotter to use. If not provided, a plotter is created in the
                background. Default is None.
            cmap: Union[str, Iterable], Optional
                A color map for plotting. See PyVista's docs for the details.
                Default is None.
            camera_position: tuple, Optional
                Camera position. See PyVista's docs for the details. Default is None.
            lighting: bool, Optional
                Whether to use lighting or not. Default is None.
            edge_color: str, Optional
                The color of the edges if `show_edges` is `True`. Default is None,
                which equals to the default PyVista setting.
            return_img: bool, Optional
                If True, a screenshot is returned as an image. Default is False.
            show_scalar_bar: Union[bool, None], Optional
                Whether to show the scalar bar or not. A `None` value means that the option
                is governed by the configurations of the blocks. If a boolean is provided here,
                it overrides the configurations of the blocks. Default is None.
            **kwargs
                Extra keyword arguments passed to `pyvista.Plotter`, it the plotter
                has to be created.

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
            if not __haspyvista__:
                raise ImportError("You need to install `pyVista` for this.")
            polys = self.to_pv(deepcopy=deepcopy, multiblock=False, scalars=scalars)
            if isinstance(theme, str):
                try:
                    new_theme_type = pv.themes._ALLOWED_THEMES[theme].value
                    theme = new_theme_type()
                except Exception:
                    if theme == "dark":
                        theme = themes.DarkTheme()
                        theme.lighting = False
                    elif theme == "bw":
                        theme = themes.DefaultTheme()
                        theme.color = "black"
                        theme.lighting = True
                        theme.edge_color = "white"
                        theme.background = "white"
                    elif theme == "document":
                        theme = themes.DocumentTheme()

            if theme is None:
                theme = pv.global_theme

            theme.show_edges = show_edges

            if lighting is not None:
                theme.lighting = lighting
            if edge_color is not None:
                theme.edge_color = edge_color

            if plotter is None:
                pvparams = dict()
                if window_size is not None:
                    pvparams.update(window_size=window_size)
                pvparams.update(kwargs)
                pvparams.update(notebook=notebook)
                pvparams.update(theme=theme)
                if "title" not in pvparams:
                    pvparams["title"] = "sigmaepsilon.mesh"
                plotter = pv.Plotter(**pvparams)

            if camera_position is not None:
                plotter.camera_position = camera_position

            pvparams = dict()
            blocks = self.cellblocks(inclusive=True, deep=True)

            blocks_have_data = self._has_plot_scalars_(scalars)

            if config_key is None:
                config_key = self.__class__._pv_config_key_

            for block, poly, has_data in zip(blocks, polys, blocks_have_data):
                params = copy(pvparams)
                config = block._get_config_(config_key)
                if has_data:
                    config.pop("color", None)
                params.update(config)
                if cmap is not None:
                    params["cmap"] = cmap
                if block.cd.NDIM > 1:
                    params["show_edges"] = show_edges
                if isinstance(show_scalar_bar, bool):
                    params["show_scalar_bar"] = show_scalar_bar
                plotter.add_mesh(poly, **params)

            if return_plotter:
                return plotter

            show_params = dict()
            if notebook:
                show_params.update(jupyter_backend=jupyter_backend)
            else:
                if return_img:
                    plotter.show(auto_close=False)
                    plotter.show(screenshot=True)
                    return plotter.last_image

            return plotter.show(**show_params)

    def plot(
        self,
        *,
        notebook: bool = False,
        backend: str = "pyvista",
        config_key: str = None,
        **kwargs,
    ):
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

    def __join_parent__(self, parent: DeepDict, key: Hashable = None):
        super().__join_parent__(parent, key)
        if self.celldata is not None:
            GIDs = self.root().cim.generate_np(len(self.celldata))
            self.celldata.id = atleast1d(GIDs)
            if self.celldata.pd is None:
                self.celldata.pd = self.source().pd
            self.celldata.container = self

    def __leave_parent__(self):
        if self.celldata is not None:
            self.root().cim.recycle(self.celldata.id)
            dbkey = self.celldata._dbkey_id_
            del self.celldata._wrapped[dbkey]
        super().__leave_parent__()

    def __repr__(self):
        return "PolyData(%s)" % (dict.__repr__(self))


class IndexManager(object):
    """
    Manages and index set by generating and recycling indices
    of a set of points or cells.
    """

    def __init__(self, start=0):
        self.queue = []
        self.next = start

    def generate_np(self, n: int = 1) -> Union[int, ndarray]:
        if n == 1:
            return self.generate(1)
        else:
            return np.array(self.generate(n))

    def generate(self, n: int = 1) -> Union[int, ndarray]:
        nQ = len(self.queue)
        if nQ > 0:
            if n == 1:
                res = self.queue.pop()
            else:
                if nQ >= n:
                    res = self.queue[:n]
                    del self.queue[:n]
                else:
                    res = copy(self.queue)
                    res.extend(range(self.next, self.next + n - nQ))
                    self.queue = []
                self.next += n - nQ
        else:
            if n == 1:
                res = self.next
            else:
                res = list(range(self.next, self.next + n))
            self.next += n
        return res

    def recycle(self, *args):
        for a in args:
            if isinstance(a, Iterable):
                self.queue.extend(a)
            else:
                self.queue.append(a)
