import operator
from typing import Union, Iterable, Optional
from contextlib import suppress

from numba import njit, prange
from numba.core import types as nbtypes, cgutils
from numba.extending import (
    typeof_impl,
    models,
    make_attribute_wrapper,
    register_model,
    box,
    unbox,
    NativeValue,
    type_callable,
    lower_builtin,
    overload,
    overload_attribute,
)

from numpy import ndarray
import numpy as np

from sigmaepsilon.core.typing import issequence
from sigmaepsilon.math import minmax
from sigmaepsilon.math.linalg import Vector, FrameLike

from .frame import CartesianFrame
from .point import Point
from ..utils.space import index_of_closest_point, index_of_furthest_point

__cache = True


__all__ = ["PointCloud"]


VectorLike = Union[Point, Vector, ndarray]


@njit(nogil=True, parallel=True, cache=__cache)
def show_coords(dcm: ndarray, coords: ndarray) -> ndarray:
    res = np.zeros_like(coords)
    for i in prange(coords.shape[0]):
        res[i] = dcm @ coords[i, :]
    return res


def dcoords(coords: ndarray, v: ndarray) -> ndarray:
    res = np.zeros_like(coords)
    if len(res.shape) == 1:
        with suppress(IndexError):
            res[0] = v[0]
            res[1] = v[1]
            res[2] = v[2]
    elif len(res.shape) == 2:
        with suppress(IndexError):
            res[:, 0] = v[0]
            res[:, 1] = v[1]
            res[:, 2] = v[2]
    else:  # pragma: no cover
        raise ValueError("Only 1 and 2 dimensional arrays are supported here.")
    return res


class PointCloud(Vector):
    """
    A numba-jittable class to support calculations related to points
    in Euclidean space.

    Parameters
    ----------
    frame: FrameLike or numpy.ndarray, Optional
        A numpy array representing coordinate axes of a reference frame.
        Default is None.
    inds: numpy.ndarray, Optional
        An 1d integer array specifying point indices. Default is None.

    Examples
    --------
    Collect the points of a simple triangulation and get the center:

    >>> from sigmaepsilon.mesh.space import PointCloud
    >>> from sigmaepsilon.mesh.triang import triangulate
    >>> coords, *_ = triangulate(size=(800, 600), shape=(10, 10))
    >>> coords = PointCloud(coords)
    >>> coords.center()
    array([400., 300.,   0.])

    Centralize and get center again:

    >>> coords = coords.centralize()
    >>> center = coords.center()  # array([0., 0., 0.])

    Move the points in the global frame:

    >>> coords = coords.move(np.array([1., 0., 0.]))
    >>> center = coords.center()  # array([1., 0., 0.])

    Rotate the points with 90 degrees around global Z.
    Before we do so, let check the boundaries:

    >>> coords.x().min(), coords.x().max()
    (Array(-399.), Array(401.))

    >>> coords.y().min(), coords.y().max()
    (Array(-300.), Array(300.))

    Now centralize wrt. the global frame, rotate and check
    the boundaries again:

    >>> coords = coords.rotate('Body', [0, 0, np.pi], 'XYZ')
    >>> center = coords.center()  # [1., 0., 0.]

    >>> coords.x().min(), coords.x().max()
    (Array(-401.), Array(399.))

    >>> coords.y().min(), coords.y().max()
    (Array(-300.), Array(300.))

    The object keeps track of indices after slicing, always
    referring to the top level array:

    >>> coords[10:50][[1, 2, 10]].inds
    array([11, 12, 20])

    The instances are available in Numba-jitted functions, with
    the coordinates and indices available as 'data' and 'inds':

    >>> from numba import jit
    >>> @jit(nopython=True)
    ... def foo(arr): return arr.data, arr.inds
    >>> c = np.array([[0, 0, 0], [0, 0, 1.], [0, 0, 0]])
    >>> COORD = PointCloud(c, inds=np.array([0, 1, 2, 3]))
    >>> data, inds = foo(COORD)
    """

    _frame_cls_ = CartesianFrame

    def __init__(
        self,
        *args,
        frame: Optional[Union[CartesianFrame, FrameLike, None]] = None,
        inds: Optional[Union[ndarray, None]] = None,
        **kwargs,
    ):
        if frame is None:
            if len(args) > 0 and isinstance(args[0], ndarray):
                frame = self._frame_cls_(dim=args[0].shape[-1])
        super().__init__(*args, frame=frame, **kwargs)
        self.inds = inds if inds is None else np.array(inds, dtype=int)

    def __getitem__(self, key):
        inds = None
        key = (key,) if not isinstance(key, tuple) else key
        if isinstance(key[0], slice):
            slc = key[0]
            start, stop, step = slc.start, slc.stop, slc.step
            start = 0 if start == None else start
            step = 1 if step == None else step
            stop = self.shape[0] if stop == None else stop
            inds = list(range(start, stop, step))
        elif issequence(key[0]):
            inds = key[0]
        elif isinstance(key[0], int):
            inds = [
                key[0],
            ]
        if inds is not None and self.inds is not None:
            inds = self.inds[inds]
        arr = self._array.__getitem__(key)
        return PointCloud(arr, frame=self.frame, inds=inds)

    @property
    def frame(self) -> Union[CartesianFrame, FrameLike]:
        """
        Returns the frame the points are embedded in.
        """
        return self._frame

    @frame.setter
    def frame(self, target: FrameLike):
        """
        Sets the frame. This changes the frame itself and results
        in a transformation of coordinates.

        Parameters
        ----------
        target: FrameLike
            A target frame of reference.
        """
        if isinstance(target, FrameLike):
            self._array = self.show(target)
            self._frame = target
        else:
            raise TypeError("Value must be a {} instance".format(FrameLike))

    @property
    def id(self) -> ndarray:
        """
        Returns the indices of the points.
        """
        return self.inds

    def x(self, target: FrameLike = None) -> ndarray:
        """Returns the `x` coordinates."""
        arr = self.show(target)
        return arr[:, 0] if len(self.shape) > 1 else arr[0]

    def y(self, target: FrameLike = None) -> ndarray:
        """Returns the `y` coordinates."""
        arr = self.show(target)
        return arr[:, 1] if len(self.shape) > 1 else arr[1]

    def z(self, target: FrameLike = None) -> ndarray:
        """Returns the `z` coordinates."""
        arr = self.show(target)
        return arr[:, 2] if len(self.shape) > 1 else arr[2]

    def bounds(self, target: FrameLike = None) -> ndarray:
        """
        Returns the bounds of the pointcloud as a numpy array with
        a shape of (N, 2), where N is 2 for 2d problems and 3 for 3d
        ones.
        """
        arr = self.show(target)
        dim = arr.shape[1]
        res = np.zeros((dim, 2))
        res[0] = minmax(arr[:, 0])
        res[1] = minmax(arr[:, 1])
        if dim > 2:
            res[2] = minmax(arr[:, 2])
        return res

    def center(self, target: FrameLike = None) -> ndarray:
        """
        Returns the center of the points in a specified frame,
        or the root frame if there is no target provided.

        Parameters
        ----------
        target: ReferenceFrame, Optional
            A frame of reference. Default is None.

        Returns
        -------
        numpy.ndarray
            A numpy array.
        """
        arr = self.show(target)

        def mean(i: int) -> float:
            return np.mean(arr[:, i])

        return np.array(list(map(mean, range(self.shape[1]))))

    def index_of_closest(self, p: VectorLike, frame: FrameLike = None) -> int:
        """
        Returns the index of the point being closest to `p`.

        Parameters
        ----------
        p: Vector or Array, Optional
            Vectors or coordinates of one or more points. If provided as
            an array, the `frame` argument can be used to specify the
            parent frame in which the coordinates are to be understood.
        frame: ReferenceFrame, Optional
            A frame in which the input is defined if it is not a Vector.
            Default is None.

        Returns
        -------
        int
        """
        if not isinstance(p, Vector):
            p = np.array(p)
            if frame is None:
                frame = self._frame_cls_(dim=p.shape[-1])
            p = Vector(p, frame=frame)
        return index_of_closest_point(self.show(), p.show())

    def index_of_furthest(self, p: VectorLike, frame: FrameLike = None) -> int:
        """
        Returns the index of the point being furthest from `p`.

        Parameters
        ----------
        p: Vector or Array, Optional
            Vectors or coordinates of one or more points. If provided as
            an array, the `frame` argument can be used to specify the
            parent frame in which the coordinates are to be understood.
        frame: ReferenceFrame, Optional
            A frame in which the input is defined if it is not a Vector.
            Default is None.

        Returns
        -------
        int
        """
        if not isinstance(p, Vector):
            p = Vector(p, frame=frame)
        return index_of_furthest_point(self.show(), p.show())

    def closest(self, p: VectorLike, frame: FrameLike = None) -> Point:
        """
        Returns the point being closest to `p`.

        Parameters
        ----------
        p: Vector or Array, Optional
            Vectors or coordinates of one or more points. If provided as
            an array, the `frame` argument can be used to specify the
            parent frame in which the coordinates are to be understood.
        frame: ReferenceFrame, Optional
            A frame in which the input is defined if it is not a Vector.
            Default is None.

        Returns
        -------
        `~sigmaepsilon.mesh.space.point.Point`
        """
        id = self.index_of_closest(p, frame)
        arr = self._array[id, :]
        if isinstance(self.inds, np.ndarray):
            gid = self.inds[id]
        else:
            gid = id
        if isinstance(id, int):
            return Point(arr, frame=self.frame, id=id, gid=gid)
        else:
            return PointCloud(arr, frame=self.frame, inds=id)

    def furthest(self, p: VectorLike, frame: FrameLike = None) -> Point:
        """
        Returns the point being closest to `p`.

        Parameters
        ----------
        p: Vector or Array, Optional
            Vectors or coordinates of one or more points. If provided as
            an array, the `frame` argument can be used to specify the
            parent frame in which the coordinates are to be understood.
        frame: ReferenceFrame, Optional
            A frame in which the input is defined if it is not a Vector.
            Default is None.

        Returns
        -------
        `~sigmaepsilon.mesh.space.point.Point`
        """
        id = self.index_of_furthest(p, frame)
        arr = self._array[id, :]
        if isinstance(self.inds, np.ndarray):
            gid = self.inds[id]
        else:
            gid = id
        if isinstance(id, int):
            return Point(arr, frame=self.frame, id=id, gid=gid)
        else:
            return PointCloud(arr, frame=self.frame, inds=id)

    def show(self, target: FrameLike = None, *args, **kwargs) -> ndarray:
        """
        Returns the coordinates of the points in a specified frame,
        or the root frame if there is no target provided.

        Parameters
        ----------
        target: ReferenceFrame, Optional
            A frame of reference. Default is None.

        Notes
        -----
        This function returns the coordinates of the points in a target
        frame, but does not make any changes to the points themselves.
        If you want to change the frame of the pointcloud, reset the
        frame of the object by setting the `frame` property.

        See Also
        --------
        :func:`~sigmaepsilon.mesh.space.pointcloud.PointCloud.frame`

        Returns
        -------
        numpy.ndarray
            The coordinates in the desired frame.
        """
        # passing unexpected arguments is necessary here because the
        # function might ocassionally be called from super()
        x = super().show(target, *args, **kwargs)
        frame = self.frame
        if isinstance(frame, CartesianFrame):
            buf = x + dcoords(x, self.frame.relative_origo(target))
        else:
            buf = x
        return self._array_cls_(shape=buf.shape, buffer=buf, dtype=buf.dtype)

    def move(self, v: VectorLike, frame: FrameLike = None) -> "PointCloud":
        """
        Moves the points wrt. to a specified frame, or the root
        frame if there is no target provided. Returns the object
        for continuation.

        Parameters
        ----------
        v: Vector or Array, Optional
            An array of a vector. If provided as an array, the `frame`
            argument can be used to specify the parent frame in which the
            motion is tp be understood.
        frame: ReferenceFrame, Optional
            A frame in which the input is defined if it is not a Vector.
            Default is None.

        Returns
        -------
        PointCloud
            The object the function is called on.

        Examples
        --------
        Collect the points of a simple triangulation and get the center:

        >>> from sigmaepsilon.mesh import triangulate
        >>> coords, *_ = triangulate(size=(800, 600), shape=(10, 10))
        >>> coords = PointCloud(coords)
        >>> coords.center()
        array([400., 300.,   0.])

        Move the points and get the center again:

        >>> d = np.array([0., 1., 0.])
        >>> coords = coords.move(d).move(d)
        >>> coords.center()
        array([400., 302.,   0.])
        """
        if not isinstance(v, Vector) and frame:
            v = Vector(v, frame=frame)
            arr = v.show(self.frame)
        else:
            arr = v
        self._array += dcoords(self._array, arr)
        return self

    def centralize(
        self, target: FrameLike = None, axes: Iterable = None
    ) -> "PointCloud":
        """
        Centralizes the coordinates wrt. to a specified frame,
        or the root frame if there is no target provided.

        Returns the object for continuation.

        Parameters
        ----------
        target: ReferenceFrame, Optional
            A frame of reference. Default is None.
        axes: Iterable, Optional
            The axes on which centralization is to be performed. A `None` value
            means all axes. Default is None.

        Returns
        -------
        PointCloud
            The object the function is called on.
        """
        center = self.center(target)
        d = np.zeros_like(center, dtype=float)
        if not isinstance(axes, Iterable):
            axes = list(range(len(center)))
        d[axes] = center[axes]
        return self.move(-d, target)

    def rotate(self, *args, **kwargs) -> "PointCloud":
        """
        Applies a transformation to the coordinates in-place. All arguments
        are passed to `ReferenceFrame.orient_new`, see its docs to know more.

        Returns the object for continuation.

        Parameters
        ----------
        *args: tuple,
            The first positional argument can be a ReferenceFrame object.
            If it is not, all positional and keyword arguments are forwarded
            to `ReferenceFrame.orient_new`.

        Returns
        -------
        PointCloud
            The object the function is called on.

        Examples
        --------
        To apply a 90 degree rotation about the Z axis:

        >>> from sigmaepsilon.mesh.space import PointCloud
        >>> from sigmaepsilon.mesh import triangulate
        >>> coords, *_ = triangulate(size=(800, 600), shape=(10, 10))
        >>> points = PointCloud(coords)
        >>> rotated_pointcloud = points.rotate('Space', [0, 0, np.pi/2], 'XYZ')
        """
        if isinstance(args[0], FrameLike):
            self.orient(dcm=args[0].dcm())
            return self
        else:
            target = self.frame.orient_new(*args, **kwargs)
            return self.rotate(target)

    def idsort(self) -> ndarray:
        """
        Returns the indices that would sort the array according to
        their indices.
        """
        return np.argsort(self.inds)

    def sort_indices(self) -> "PointCloud":
        """
        Sorts the points according to their indices and returns the
        object.
        """
        s = self.idsort()
        self._array = self._array[s]
        self.inds = self.inds[s]
        return self

    def __repr__(self):
        return f"PointCloud({self._array})"

    def __str__(self):
        return f"PointCloud({self._array})"


class PointCloudType(nbtypes.Type):  # pragma: no cover
    """Numba type."""

    def __init__(self, datatype, indstype=nbtypes.NoneType):
        self.data = datatype
        self.inds = indstype
        super(PointCloudType, self).__init__(name="PointCloud")


make_attribute_wrapper(PointCloudType, "data", "data")
make_attribute_wrapper(PointCloudType, "inds", "inds")


@overload_attribute(PointCloudType, "x")
def attr_x(arr):  # pragma: no cover
    def get(arr):
        return arr.data[:, 0]

    return get


@overload_attribute(PointCloudType, "y")
def attr_y(arr):  # pragma: no cover
    def get(arr):
        return arr.data[:, 1]

    return get


@overload_attribute(PointCloudType, "z")
def attr_z(arr):  # pragma: no cover
    def get(arr):
        return arr.data[:, 2]

    return get


@typeof_impl.register(PointCloud)
def type_of_impl(val, context):  # pragma: no cover
    """`val` is the Python object being typed"""
    datatype = typeof_impl(val._array, context)
    indstype = typeof_impl(val.inds, context)
    return PointCloudType(datatype, indstype)


@type_callable(PointCloud)
def type_of_callable(context):  # pragma: no cover
    def typer(data, inds=None):
        datatype = typeof_impl(data, context)
        indstype = typeof_impl(inds, context) if inds is not None else nbtypes.NoneType
        return PointCloudType(datatype, indstype)

    return typer


@register_model(PointCloudType)
class StructModel(models.StructModel):  # pragma: no cover
    """Data model for nopython mode."""

    def __init__(self, dmm, fe_type):
        """
        fe_type is `PointCloudType`
        """
        members = [
            ("data", fe_type.data),
            ("inds", fe_type.inds),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@overload(operator.getitem)
def overload_getitem(obj, idx):  # pragma: no cover
    if isinstance(obj, PointCloudType):

        def dummy_getitem_impl(obj, idx):
            return obj.data[idx]

        return dummy_getitem_impl


@lower_builtin(PointCloud, nbtypes.Array)
def lower_type(context, builder, sig, args):  # pragma: no cover
    typ = sig.return_type
    data, inds = args
    obj = cgutils.create_struct_proxy(typ)(context, builder)
    obj.data = data
    obj.inds = inds
    return obj._getvalue()


@unbox(PointCloudType)
def unbox_type(typ, obj, c):  # pragma: no cover
    """Convert a python object to a numba-native structure."""
    data_obj = c.pyapi.object_getattr_string(obj, "_array")
    inds_obj = c.pyapi.object_getattr_string(obj, "inds")
    native_obj = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    native_obj.data = c.unbox(typ.data, data_obj).value
    native_obj.inds = c.unbox(typ.inds, inds_obj).value
    c.pyapi.decref(data_obj)
    c.pyapi.decref(inds_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(native_obj._getvalue(), is_error=is_error)


@box(PointCloudType)
def box_type(typ, val, c):  # pragma: no cover
    """Convert a numba-native structure to a python object."""
    native_obj = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(PointCloud))
    data_obj = c.box(typ.data, native_obj.data)
    inds_obj = c.box(typ.inds, native_obj.inds)
    python_obj = c.pyapi.call_function_objargs(class_obj, (data_obj, inds_obj))
    c.pyapi.decref(data_obj)
    c.pyapi.decref(inds_obj)
    return python_obj
