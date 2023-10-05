from typing import Union, Iterable, Optional
from copy import deepcopy

import numpy as np
from numpy import ndarray
from awkward import Record as akRecord

from sigmaepsilon.core import classproperty
from sigmaepsilon.math.linalg import ReferenceFrame as FrameLike
from sigmaepsilon.math.logical import isboolarray
from sigmaepsilon.math.linalg.sparse import csr_matrix

from ..typing.abcakwrapper import ABC_AkWrapper
from .akwrapper import AkWrapper
from ..space import CartesianFrame, PointCloud
from ..typing import PolyDataProtocol
from ..utils import collect_nodal_data


__all__ = ["PointData"]


def gen_frame(coords: ndarray) -> CartesianFrame:
    return CartesianFrame(dim=coords.shape[1])


class PointData(AkWrapper, ABC_AkWrapper):
    """
    A class to handle data related to the pointcloud of a polygonal mesh.

    Technicall this is a wrapper around an `awkward.Record` instance.

    If you are not a developer, you probably don't have to ever create any
    instance of this class, but since it operates in the background of every
    polygonal data structure, it is important to understand how it works.
    """

    _point_cls_ = PointCloud
    _frame_class_ = CartesianFrame
    _attr_map_ = {
        "x": "_x",  # coordinates
        "activity": "_activity",  # activity of the points
        "id": "_id",  # global indices of the points
    }

    def __init__(
        self,
        *args,
        points: Optional[Union[ndarray, None]] = None,
        coords: Optional[Union[ndarray, None]] = None,
        wrap: Optional[Union[akRecord, None]] = None,
        fields: Optional[Union[Iterable, None]] = None,
        frame: Optional[Union[FrameLike, None]] = None,
        db: Optional[Union[akRecord, None]] = None,
        container: Optional[Union[PolyDataProtocol, None]] = None,
        **kwargs,
    ):
        if db is not None:
            wrap = db
        elif wrap is None:
            fields = {} if fields is None else fields
            assert isinstance(fields, dict)

            X = None

            if len(args) > 0:
                if isinstance(args[0], np.ndarray):
                    X = args[0]
            else:
                X = points if coords is None else coords

            if X is None:
                raise ValueError("Coordinates must be specified.")

            if not isinstance(X, ndarray):
                raise TypeError("Coordinates must be specified as a NumPy array!")

            fields[self._dbkey_x_] = X
            nP = len(X)

            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    if v.shape[0] == nP:
                        fields[k] = v

        # coordinate frame
        if not isinstance(frame, FrameLike):
            if coords is not None:
                frame = gen_frame(coords)

        self._frame = frame

        super().__init__(*args, wrap=wrap, fields=fields, **kwargs)
        self._container = container

    def __deepcopy__(self, memo):
        return self.__copy__(memo)

    def __copy__(self, memo=None):
        cls = type(self)
        is_deep = memo is not None

        if is_deep:
            copy_function = lambda x: deepcopy(x, memo)
        else:
            copy_function = lambda x: x

        db = copy_function(self.db)
        f = self.frame

        if f is not None:
            axes = copy_function(f.axes)
            if is_deep:
                memo[id(f.axes)] = axes
            frame_cls = type(f)
            frame = frame_cls(axes)
        else:
            frame = None

        result = cls(db=db, frame=frame)

        if is_deep:
            memo[id(self)] = result

        result_dict = result.__dict__

        for k, v in self.__dict__.items():
            if not k in result_dict:
                setattr(result, k, copy_function(v))

        return result

    @classproperty
    def _dbkey_id_(cls) -> str:
        return cls._attr_map_["id"]

    @classproperty
    def _dbkey_x_(cls) -> str:
        return cls._attr_map_["x"]

    @classproperty
    def _dbkey_activity_(cls) -> str:
        return cls._attr_map_["activity"]

    @property
    def has_id(self) -> ndarray:
        return self._dbkey_id_ in self._wrapped.fields

    @property
    def has_x(self) -> ndarray:
        return self._dbkey_x_ in self._wrapped.fields

    @property
    def container(self) -> PolyDataProtocol:
        """
        Returns the container object of the block.
        """
        return self._container

    @container.setter
    def container(self, value: PolyDataProtocol):
        """
        Sets the container of the block.
        """
        assert isinstance(value, PolyDataProtocol)
        self._container = value

    @property
    def frame(self) -> FrameLike:
        """
        Returns the frame of the underlying pointcloud.
        """
        result = None

        if isinstance(self._frame, FrameLike):
            result = self._frame

        if result is None:
            dim = self.x.shape[-1]
            result = self._frame_class_(dim=dim)

        return result

    @property
    def activity(self) -> ndarray:
        return self._wrapped[self._dbkey_activity_].to_numpy()

    @activity.setter
    def activity(self, value: ndarray):
        if not isinstance(value, ndarray):
            raise TypeError(f"Expected a NumPy array, got {type(value)}.")
        
        if not isboolarray(value):
            raise ValueError(f"Expected a boolean array, got {value.dtype}.")
        
        self._wrapped[self._dbkey_activity_] = value

    @property
    def x(self) -> ndarray:
        return self._wrapped[self._dbkey_x_].to_numpy()

    @x.setter
    def x(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_x_] = value

    @property
    def id(self) -> ndarray:
        return self._wrapped[self._dbkey_id_].to_numpy()

    @id.setter
    def id(self, value: ndarray):
        assert isinstance(value, ndarray)
        self._wrapped[self._dbkey_id_] = value

    def pull(self, key: str, ndf: Union[ndarray, csr_matrix] = None) -> ndarray:
        """
        Pulls data from the cells in the model. The pulled data is either copied or
        distributed according to a measure.

        Parameters
        ----------
        key: str
            A field key to identify data in the databases of the attached
            CellData instances of the blocks.
        ndf: Union[numpy.ndarray, csr_matrix], Optional
            The nodal distribution factors to use. If not provided, the
            default factors are used. Default is None.

        See Also
        --------
        :func:`~sigmaepsilon.mesh.data.polydata.PolyData.nodal_distribution_factors`
        :func:`~sigmaepsilon.mesh.utils.utils.collect_nodal_data`
        """
        source: PolyDataProtocol = self.container.source()

        if ndf is None:
            ndf = source.nodal_distribution_factors()

        if isinstance(ndf, ndarray):
            ndf = csr_matrix(ndf)

        blocks = list(source.cellblocks(inclusive=True))
        b = blocks.pop(0)
        cids = b.cd.id
        topo = b.cd.nodes
        celldata = b.cd.db[key].to_numpy()

        if len(celldata.shape) == 1:
            nE, nNE = topo.shape
            celldata = np.repeat(celldata, nNE).reshape(nE, nNE)

        shp = [len(self)] + list(celldata.shape[2:])
        res = np.zeros(shp, dtype=float)
        collect_nodal_data(celldata, topo, cids, ndf, res)

        for b in blocks:
            cids = b.cd.id
            topo = b.cd.nodes
            celldata = b.cd.db[key].to_numpy()
            if len(celldata.shape) == 1:
                nE, nNE = topo.shape
                celldata = np.repeat(celldata, nNE).reshape(nE, nNE)
            collect_nodal_data(celldata, topo, cids, ndf, res)

        return res
