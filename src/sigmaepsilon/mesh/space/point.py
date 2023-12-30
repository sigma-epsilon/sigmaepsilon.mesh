from typing import Union, Iterable
import warnings

import numpy as np

from sigmaepsilon.math.linalg import Vector, FrameLike, CartesianFrame, ReferenceFrame


class Point(Vector):
    """
    .. deprecated:: 2.3.0
       Use :class:`~sigmaepsilon.mesh.space.pointcloud.PointCloud` instead.

    A class a to handle a single point in Euclidean space.

    It inherits :class:`Vector <sigmaepsilon.math.linalg.vector.Vector>`,
    and extends its behaviour with default frame management for domain specific applications.

    If data is provided on object creation, the class can infer an appropriate default frame,
    hence the specification of such can be omitted.

    Parameters
    ----------
    frame: Union[ReferenceFrame, numpy.ndarray, Iterable], Optional
        A suitable reference frame, or an iterable representing coordinate axes of one.
        Default is None.

    Note
    ----
    1) This is class is superseded by :class:`PointCloud <sigmaepsilon.mesh.space.pointcloud.PointCloud>`.
    2) This class does not take the origo of the supporting reference frame into consideration when
    transforming coordinates between frames.

    Examples
    --------
    >>> from sigmaepsilon.mesh.space import Point
    >>> p = Point([1., 1., 1.])
    >>> type(p.frame)
    sigmaepsilon.math.linalg.frame.CartesianFrame

    If we want to handle more than one point:

    >>> import math
    >>> p = Point([[1., 0., 0.], [0., 1., 0.]])
    >>> A = p.frame
    >>> B = A.orient_new('Body', [0, 0, math.pi/2], 'XYZ')
    >>> p.show(B)
    array([[ 0.0, -1.0,  0.0],
           [ 1.0,  0.0,  0.0]])
    """

    _frame_cls_ = CartesianFrame

    def __init__(
        self,
        *args,
        frame: Union[ReferenceFrame, np.ndarray, Iterable] = None,
        id: int = None,
        gid: int = None,
        **kwargs,
    ):
        warnings.warn(
            "The Point class is deprecated and will be removed in a future version. "
            "Use a PointCloud instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if frame is None:
            if len(args) > 0:
                if isinstance(args[0], np.ndarray):
                    frame = self._frame_cls_(dim=args[0].shape[-1])
                else:
                    try:
                        arg = np.array(args[0])
                        frame = self._frame_cls_(dim=arg.shape[-1])
                    except Exception:
                        frame = None
        else:
            if not isinstance(frame, self._frame_cls_):
                if isinstance(frame, FrameLike):
                    frame = self._frame_cls_(frame.axes)
                elif isinstance(frame, np.ndarray):
                    frame = self._frame_cls_(frame)
                elif isinstance(frame, Iterable):
                    frame = self._frame_cls_(np.array(frame, dtype=float))

        if not isinstance(frame, self._frame_cls_):
            raise ValueError("Invalid frame!")

        super().__init__(*args, frame=frame, **kwargs)
        self._id = id
        self._gid = id if gid is None else gid

    @property
    def id(self) -> int:
        return self._id

    @property
    def gid(self) -> int:
        return self._gid
