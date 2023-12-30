from typing import Union, Iterable, Generic, TypeVar, Optional
from copy import deepcopy

import numpy as np
from numpy import ndarray

from sigmaepsilon.core import classproperty
from sigmaepsilon.math import atleast3d, repeat
from sigmaepsilon.math.linalg import ReferenceFrame

from .akwrapper import AkWrapper
from ..typing import PolyDataProtocol, PointDataProtocol
from ..typing.abcakwrapper import ABC_AkWrapper
from .akwrapper import AwkwardLike

PointDataLike = TypeVar("PointDataLike", bound=PointDataProtocol)
PolyDataLike = TypeVar("PolyDataLike", bound=PolyDataProtocol)


__all__ = ["CellData"]


class CellData(Generic[PolyDataLike, PointDataLike], AkWrapper, ABC_AkWrapper):
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
    i: numpy.ndarray, Optional
        The (global) indices of the cells. Default is None.
    **kwargs: dict, Optional
        For every key and value pair where the value is a numpy array
        with a matching shape (has entries for all cells), the key
        is considered as a field and the value is added to the database.
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
        wrap: Optional[Union[AwkwardLike, None]] = None,
        topo: Optional[Union[ndarray, None]] = None,
        fields: Optional[Union[dict, None]] = None,
        activity: Optional[Union[ndarray, None]] = None,
        frames: Optional[Union[ndarray, ReferenceFrame, None]] = None,
        areas: Optional[Union[ndarray, float, None]] = None,
        t: Optional[Union[ndarray, float, None]] = None,
        db: Optional[Union[AwkwardLike, None]] = None,
        i: Optional[Union[ndarray, None]] = None,
        **kwargs,
    ):
        fields = {} if fields is None else fields
        assert isinstance(fields, dict)
        if len(fields) > 0:
            attr_map = self._attr_map_
            fields = {attr_map.get(k, k): v for k, v in fields.items()}

        # cell indices
        if isinstance(i, ndarray):
            kwargs[self._dbkey_id_] = i

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

    def __deepcopy__(self, memo: dict) -> "CellData":
        return self.__copy__(memo)

    def __copy__(self, memo: dict = None) -> "CellData":
        cls = type(self)
        is_deep = memo is not None

        if is_deep:
            copy_function = lambda x: deepcopy(x, memo)
        else:
            copy_function = lambda x: x

        db = copy_function(self.db)

        result = cls(db=db)
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
    def has_nodes(self) -> bool:
        return self._dbkey_nodes_ in self._wrapped.fields

    @property
    def has_id(self) -> bool:
        return self._dbkey_id_ in self._wrapped.fields

    @property
    def has_frames(self) -> bool:
        return self._dbkey_frames_ in self._wrapped.fields

    @property
    def has_thickness(self) -> bool:
        return self._dbkey_thickness_ in self._wrapped.fields

    @property
    def has_areas(self) -> bool:
        return self._dbkey_areas_ in self._wrapped.fields

    @property
    def db(self) -> AwkwardLike:
        return self._wrapped

    @property
    def fields(self) -> Iterable[str]:
        """Returns the fields in the database object."""
        return self._wrapped.fields

    @property
    def nodes(self) -> ndarray:
        """Returns the topology of the cells."""
        return self._wrapped[self._dbkey_nodes_].to_numpy()

    @nodes.setter
    def nodes(self, value: ndarray) -> None:
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
    def frames(self, value: Union[ReferenceFrame, ndarray]) -> None:
        """
        Sets local coordinate frames of the cells.

        Parameters
        ----------
        value: numpy.ndarray
            A 3d float array.
        """
        if isinstance(value, ReferenceFrame):
            frames = value.show()
        elif isinstance(value, ndarray):
            frames = value
        else:
            raise TypeError(f"Expected ndarray or FrameLike, got {type(value)}.")

        frames = atleast3d(frames)

        if len(frames) == 1:
            frames = repeat(frames[0], len(self._wrapped))
        else:
            assert len(frames) == len(self._wrapped)

        self._wrapped[self._dbkey_frames_] = frames

    @property
    def t(self) -> ndarray:
        """Returns the thicknesses of the cells."""
        return self._wrapped[self._dbkey_thickness_].to_numpy()

    @t.setter
    def t(self, value: Union[float, int, ndarray]):
        """Returns the thicknesses of the cells."""
        if isinstance(value, (int, float)):
            value = np.full(len(self), value)
        self._wrapped[self._dbkey_thickness_] = value

    @property
    def A(self) -> ndarray:
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
    def id(self, value: ndarray) -> None:
        """
        Sets global indices of the cells.

        Parameters
        ----------
        value: numpy.ndarray
            An 1d integer array.
        """
        if isinstance(value, int):
            if len(self) == 1:
                value = np.array(
                    [
                        value,
                    ],
                    dtype=int,
                )
            else:
                raise ValueError(f"Expected an array, got {type(value)}")

        if not isinstance(value, ndarray):
            raise TypeError(f"Expected ndarray, got {type(value)}")

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

    def set_nodal_distribution_factors(self, factors: ndarray, key: str = None) -> None:
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
