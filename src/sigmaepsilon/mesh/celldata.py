from typing import Union
from copy import copy, deepcopy
from functools import partial

import numpy as np
from numpy import ndarray

from sigmaepsilon.core import classproperty
from sigmaepsilon.math import atleast2d, atleast3d, repeat
from sigmaepsilon.math.linalg.sparse import csr_matrix
from sigmaepsilon.math.linalg import ReferenceFrame

from .base import PointDataBase, CellDataBase, PolyDataBase as PolyData
from .akwrap import AwkwardLike
from .utils import (
    avg_cell_data,
    distribute_nodal_data_bulk,
    distribute_nodal_data_sparse,
)


class CellData(CellDataBase):
    """
    A class to handle data related to the cells of a polygonal mesh.

    Technically this is a wrapper around an Awkward data object instance.

    If you are not a developer, you probably don't have to ever create any
    instance of this class, but since it operates in the background of every
    polygonal data structure, it is useful to understand how it works.

    Parameters
    ----------
    activity: numpy.ndarray, Optional
        1d boolean array describing the activity of the elements.
    t or thickness: numpy.ndarray, Optional
        1d float array of thicknesses. Only for 2d cells.
        Default is None.
    areas: numpy.ndarray, Optional
        1d float array of cross sectional areas. Only for 1d cells.
        Default is None.
    fields: dict, Optional
        Every value of this dictionary is added to the dataset.
        Default is `None`.
    frames: numpy.ndarray, Optional
        Coordinate axes representing cartesian coordinate frames.
        Default is None.
    topo: numpy.ndarray, Optional
        2d integer array representing node indices. Default is None.
    **kwargs: dict, Optional
        For every key and value pair where the value is a numpy array
        with a matching shape (has entries for all cells), the key
        is considered as a field and the value is added to the database.

    See Also
    --------
    :class:`awkward.Array`
    :class:`awkward.Record`
    """

    _attr_map_ = {
        "nodes": "_nodes",  # node indices
        "frames": "_frames",  # coordinate frames
        "ndf": "_ndf",  # nodal distribution factors
        "id": "_id",  # global indices of the cells
        "areas": "_areas",  # areas of 1d cells
        "t": "_t",  # thicknesses for 2d cells
        "activity": "_activity",  # activity of the cells
    }

    def __init__(
        self,
        *args,
        pointdata: PointDataBase = None,
        wrap: AwkwardLike = None,
        topo: ndarray = None,
        fields: dict = None,
        activity: ndarray = None,
        frames: Union[ndarray, ReferenceFrame] = None,
        areas: Union[ndarray, float] = None,
        t: Union[ndarray, float] = None,
        db: AwkwardLike = None,
        container: PolyData = None,
        **kwargs,
    ):
        fields = {} if fields is None else fields
        assert isinstance(fields, dict)
        if len(fields) > 0:
            attr_map = self._attr_map_
            fields = {attr_map.get(k, k): v for k, v in fields.items()}

        if db is not None:
            wrap = db
        elif wrap is None:
            nodes = None
            if len(args) > 0:
                if isinstance(args[0], ndarray):
                    nodes = args[0]
            else:
                nodes = topo

            if isinstance(activity, ndarray):
                fields[self._dbkey_activity_] = activity

            if isinstance(nodes, ndarray):
                fields[self._dbkey_nodes_] = nodes
                N = nodes.shape[0]
                for k, v in kwargs.items():
                    if isinstance(v, ndarray):
                        if v.shape[0] == N:
                            fields[k] = v

            if isinstance(areas, np.ndarray):
                fields[self._dbkey_areas_] = areas

        super().__init__(*args, wrap=wrap, fields=fields, **kwargs)

        self._pointdata = pointdata
        self._container = container

        if self.db is not None:
            if frames is not None:
                if isinstance(frames, (ReferenceFrame, ndarray)):
                    self.frames = frames
                else:
                    msg = (
                        "'frames' must be a NumPy array, or a ",
                        "sigmaepsilon.math.linalg.ReferenceFrame instance.",
                    )
                    raise TypeError(msg)

            if t is not None:
                self.t = t

            if areas is not None:
                self.A = areas

    def __deepcopy__(self, memo):
        return self.__copy__(memo)

    def __copy__(self, memo=None):
        cls = type(self)
        copy_function = copy if (memo is None) else partial(deepcopy, memo=memo)
        is_deep = memo is not None

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

    @classproperty
    def _dbkey_nodes_(cls) -> str:
        return cls._attr_map_["nodes"]

    @classproperty
    def _dbkey_frames_(cls) -> str:
        return cls._attr_map_["frames"]

    @classproperty
    def _dbkey_areas_(cls) -> str:
        return cls._attr_map_["areas"]

    @classproperty
    def _dbkey_thickness_(cls) -> str:
        return cls._attr_map_["t"]

    @classproperty
    def _dbkey_activity_(cls) -> str:
        return cls._attr_map_["activity"]

    @classproperty
    def _dbkey_ndf_(cls) -> str:
        return cls._attr_map_["ndf"]

    @classproperty
    def _dbkey_id_(cls) -> str:
        return cls._attr_map_["id"]

    @property
    def has_id(self) -> ndarray:
        return self._dbkey_id_ in self._wrapped.fields

    @property
    def has_frames(self):
        return self._dbkey_frames_ in self._wrapped.fields

    @property
    def has_thickness(self):
        return self._dbkey_thickness_ in self._wrapped.fields

    @property
    def has_areas(self):
        return self._dbkey_areas_ in self._wrapped.fields

    @property
    def pointdata(self) -> PointDataBase:
        """
        Returns the attached point database. This is what
        the topology of the cells are referring to.
        """
        return self._pointdata

    @pointdata.setter
    def pointdata(self, value: PointDataBase):
        """
        Sets the attached point database. This is what
        the topology of the cells are referring to.
        """
        if value is not None:
            assert isinstance(value, PointDataBase)
        self._pointdata = value

    @property
    def pd(self) -> PointDataBase:
        """
        Returns the attached point database. This is what
        the topology of the cells are referring to.
        """
        return self.pointdata

    @pd.setter
    def pd(self, value: PointDataBase):
        """Sets the attached pointdata."""
        self.pointdata = value

    @property
    def container(self) -> PolyData:
        """Returns the container object of the block."""
        return self._container

    @container.setter
    def container(self, value: PolyData):
        """Sets the container of the block."""
        assert isinstance(value, PolyData)
        self._container = value

    def root(self) -> PolyData:
        """
        Returns the top level container of the model the block is
        the part of.
        """
        c = self.container
        return None if c is None else c.root()

    def source(self) -> PolyData:
        """
        Retruns the source of the cells. This is the PolyData block
        that stores the PointData object the topology of the cells
        are referring to.
        """
        c = self.container
        return None if c is None else c.source()

    def __getattr__(self, attr):
        """
        Modified for being able to fetch data from pointcloud.
        """
        if attr in self.__dict__:
            return getattr(self, attr)
        try:
            return getattr(self._wrapped, attr)
        except AttributeError:
            try:
                if self.pointdata is not None:
                    if attr in self.pointdata.fields:
                        data = self.pointdata[attr].to_numpy()
                        topo = self.nodes
                        return avg_cell_data(data, topo)
            except:
                pass
            name = self.__class__.__name__
            raise AttributeError(f"'{name}' object has no attribute called {attr}")
        except Exception:
            name = self.__class__.__name__
            raise AttributeError(f"'{name}' object has no attribute called {attr}")

    def set_nodal_distribution_factors(self, factors: ndarray, key: str = None):
        """
        Sets nodal distribution factors.

        Parameters
        ----------
        factors: numpy.ndarray
            A 3d float array. The length of the array must equal the number
            pf cells in the block.
        key: str, Optional
            A key used to store the values in the database. This makes you able
            to use more nodal distribution strategies in one model.
            If not specified, a default key is used.
        """
        if key is None:
            key = self.__class__._attr_map_[self._dbkey_ndf_]
        if len(factors) != len(self._wrapped):
            self._wrapped[key] = factors[self._wrapped.id]
        else:
            self._wrapped[key] = factors

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

    @property
    def fields(self):
        """Returns the fields in the database object."""
        return self._wrapped.fields

    @property
    def nodes(self) -> ndarray:
        """Returns the topology of the cells."""
        return self._wrapped[self._dbkey_nodes_].to_numpy()

    @nodes.setter
    def nodes(self, value: ndarray):
        """
        Sets the topology of the cells.

        Parameters
        ----------
        value: numpy.ndarray
            A 2d integer array.
        """
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_nodes_] = value

    @property
    def frames(self) -> ndarray:
        """Returns local coordinate frames of the cells."""
        return self._wrapped[self._dbkey_frames_].to_numpy()

    @frames.setter
    def frames(self, value: ndarray):
        """
        Sets local coordinate frames of the cells.

        Parameters
        ----------
        value: numpy.ndarray
            A 3d float array.
        """
        if isinstance(value, ReferenceFrame):
            frames = value.show()
        else:
            assert isinstance(value, ndarray)
            frames = value
        frames = atleast3d(frames)
        if len(frames) == 1:
            frames = repeat(frames[0], len(self._wrapped))
        else:
            assert len(frames) == len(self._wrapped)
        self._wrapped[self._dbkey_frames_] = frames

    @property
    def t(self):
        """Returns the thicknesses of the cells."""
        return self._wrapped[self._dbkey_thickness_].to_numpy()

    @t.setter
    def t(self, value: Union[float, int, ndarray]):
        """Returns the thicknesses of the cells."""
        if isinstance(value, (int, float)):
            value = np.full(len(self), value)
        self._wrapped[self._dbkey_thickness_] = value

    @property
    def A(self):
        """Returns the thicknesses of the cells."""
        return self._wrapped[self._dbkey_areas_].to_numpy()

    @A.setter
    def A(self, value: Union[float, int, ndarray]):
        """Returns the thicknesses of the cells."""
        if isinstance(value, (int, float)):
            value = np.full(len(self), value)
        self._wrapped[self._dbkey_areas_] = value

    @property
    def id(self) -> ndarray:
        """Returns global indices of the cells."""
        return self._wrapped[self._dbkey_id_].to_numpy()

    @id.setter
    def id(self, value: ndarray):
        """
        Sets global indices of the cells.

        Parameters
        ----------
        value: numpy.ndarray
            An 1d integer array.
        """
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_id_] = value

    @property
    def activity(self) -> ndarray:
        """Returns a 1d boolean array of cell activity."""
        return self._wrapped[self._dbkey_activity_].to_numpy()

    @activity.setter
    def activity(self, value: ndarray):
        """
        Sets cell activity with a 1d boolean array.

        Parameters
        ----------
        value: numpy.ndarray
            An 1d bool array.
        """
        if isinstance(value, bool):
            value = np.full(len(self), value, dtype=bool)
        self._wrapped[self._dbkey_activity_] = value
