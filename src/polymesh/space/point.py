from typing import Union, Iterable
import numpy as np

from neumann.linalg import Vector, FrameLike, CartesianFrame, ReferenceFrame


class Point(Vector):
    """
    A class a to handle one or more points in Euclidean space.

    It inherits :class:`Vector <neumann.linalg.vector.Vector>`,
    and extends its behaviour with default frame management for domain
    specific applications.

    If data is provided on object creation, the class can infer an appropriate
    default frame, hence the specification of such can be omitted.

    Parameters
    ----------
    frame: Union[ReferenceFrame, np.ndarray, Iterable], Optional
        A suitable reference frame, or an iterable representing coordinate axes of one.
        Default is None.

    Note
    ----
    This is class is superseded by :class:`PointCloud <polymesh.space.coordarray.PointCloud>`.

    Examples
    --------
    >>> from polymesh.space import Point
    >>> p = Point([1., 1., 1.])
    >>> type(p.frame)
    neumann.linalg.frame.CartesianFrame

    If we want to handle more than one points:

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
        **kwargs
    ):
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
    def id(self):
        return self._id

    @property
    def gid(self):
        return self._gid
