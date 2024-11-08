import numpy as np
from typing import Union, Iterable
from types import NoneType
from copy import deepcopy as dcopy

import numpy as np
from numpy import ndarray

from sigmaepsilon.math.linalg import CartesianFrame as Frame, FrameLike, Vector
from sigmaepsilon.math.linalg.vector import Vector

__all__ = ["CartesianFrame"]


VectorLike = Union[Vector, ndarray]


class CartesianFrame(Frame):
    """
    A domain-specific reference frame to be used in problems related to
    Euclidean geometry.

    It builds on top of :class:`~sigmaepsilon.math.linalg.meta.FrameLike` from
    `sigmaepsilon.math`, but adds the contept of 'origo', and some other applications
    related to the field.

    See Also
    --------
    :class:`~sigmaepsilon.math.linalg.meta.FrameLike`

    Parameters
    ----------
    axes: numpy.ndarray, Optional.
        2d numpy array of floats specifying cartesian reference frames.
        Dafault is None.
    dim: int, Optional
        Dimension of the mesh. Deafult is 3.
    origo: numpy.ndarray, Optional.
        The origo of the mesh. Default is the zero vector.

    Note
    ----
    See the documentation of :class:`~sigmaepsilon.math.linalg.FrameLike` for more control over
    object creation. However, if your problem not very extreme in some sense,
    you are probably good to goo only by following the examples.

    Example
    -------
    Define a standard Cartesian frame and rotate it around axis 'Z'
    with an amount of 180 degrees:

    >>> from sigmaepsilon.mesh.space import CartesianFrame
    >>> import numpy as np

    >>> A = CartesianFrame(dim=3)
    >>> B = A.orient_new('Space', [0, 0, np.pi], 'XYZ')

    To create a third frame that rotates from B the way B rotates from A, we
    can do

    >>> A = CartesianFrame(dim=3)
    >>> C = A.orient_new('Space', [0, 0, 2*np.pi], 'XYZ')

    Then, the *DCM from A to B* , that is :math:`^{A}\\mathbf{R}^{B}` would be

    >>> A_R_B = B.dcm(source=A)

    or equivalently

    >>> A_R_B = A.dcm(target=A)

    """

    def __init__(
        self,
        axes: ndarray | NoneType = None,
        *args,
        dim: int = 3,
        origo: ndarray | NoneType = None,
        **kwargs
    ):
        axes = np.eye(dim) if axes is None else axes
        super().__init__(axes, *args, **kwargs)
        if origo is not None:
            try:
                if not isinstance(origo, ndarray) and isinstance(origo, Iterable):
                    origo = np.array(origo, dtype=float)
            except Exception:
                raise Exception(
                    "Falied to turn 'origo' into a NumPy array! Check the input!"
                )

            if isinstance(origo, ndarray):
                if not len(origo.shape) == 1:
                    raise ValueError("'origo' must be a 1d iterable!")
            else:  # pragma: no cover
                raise TypeError("'origo' must be a NumPy array or an iterable!")

        self._origo = origo

    @property
    def origo(self) -> ndarray[float]:
        if not isinstance(self._origo, ndarray):
            self._origo = np.zeros(len(self.axes))
        return self._origo

    @origo.setter
    def origo(self, value: Iterable):
        value = np.array(value).astype(float)
        if not isinstance(self._origo, ndarray):
            self._origo = np.zeros(len(self.axes))
        if value.shape == self._array.shape:
            if self._weakrefs and len(self._weakrefs) > 0:
                for v in self._weakrefs.values():
                    v.array -= value
            self._array = value
        else:
            raise ValueError("Mismatch in data dimensinons!")

    def relative_origo(self, target: FrameLike | NoneType = None) -> ndarray:
        """
        Returns the origo of the current frame in ambient space
        or with respect to another frame.

        Parameters
        ----------
        target: FrameLike, Optional
            A frame in which we want to get the origo of the current frame.
            A None value returns the origo of the current frame with respect
            to the root. Default is None.

        Returns
        -------
        numpy.ndarray
            A vector defined in ambient space, or the specified frame.

        Examples
        --------
        Define a standard Cartesian frame and rotate it around axis 'Z'
        with an amount of 180 degrees:

        >>> from sigmaepsilon.mesh.space import CartesianFrame
        >>> import numpy as np

        >>> A = CartesianFrame()
        >>> B = A.orient_new('Space', [0, 0, 45*np.pi/180],  'XYZ')

        To get the origin of frame B:

        >>> origo = B.relative_origo()  # array([0., 0., 0.])

        Move frame B (the motion is defined locally) and print the
        new point of origin with respect to A:

        >>> B = B.move(Vector([1, 0, 0], frame=B))
        >>> origo = B.relative_origo(A)  # [0.7071, 0.7071, 0.]

        Of course, the point of origin of a frame with respect to itself
        must be a zero vector:

        >>> origo = B.relative_origo(B)  # [0., 0., 0.]

        Providing with no arguments returns the distance of origin with
        respect to the root frame:

        >>> origo = B.relative_origo()  # same as B.relative_origo(B.root), [0.7071, 0.7071, 0.]
        """
        if not isinstance(self._origo, ndarray):
            self._origo = np.zeros(len(self.axes))

        if target is None:
            return Vector(self._origo).show()
        else:
            s = self.relative_origo()
            if isinstance(target, CartesianFrame):
                t = target.relative_origo()
            else:
                t = np.zeros_like(s)
            return Vector(s - t).show(target)

    def rotate(self, *args, **kwargs) -> "CartesianFrame":
        return super().rotate(*args, **kwargs)

    def move(
        self, d: VectorLike, frame: FrameLike | NoneType = None
    ) -> "CartesianFrame":
        """
        Moves the frame by shifting its origo.

        Parameters
        ----------
        d: VectorLike
            :class:`Vector` or :class:`Array`, the amount of the motion.
        frame: FrameLike, Optional
            A frame in which the input is defined if it is not a Vector.
            Default is None, which assumes the root frame.

        Returns
        -------
        CartesianFrame
            The object the function is called on.

        Examples
        --------
        >>> from sigmaepsilon.mesh.space import CartesianFrame
        >>> import numpy as np

        >>> A = CartesianFrame()
        >>> v = Vector([1., 0., 0.], frame=A)
        >>> B = A.fork('Space', [0, 0, 45*np.pi/180], 'XYZ').move(v)

        Move the frame locally with the same amount

        >>> B.move(v.array, frame=B)
        Array([[ 0.70710678,  0.70710678,  0.        ],
               [-0.70710678,  0.70710678,  0.        ],
               [ 0.        ,  0.        ,  1.        ]])

        """
        if not isinstance(d, Vector):
            if not isinstance(d, ndarray):
                d = np.array(d)
            d = Vector(d, frame=frame)
        if self._origo is None:
            self._origo = np.zeros(len(self.axes))
        self._origo += d.show()
        return self

    def fork(self, *args, **kwargs) -> "CartesianFrame":
        """
        Alias for `orient_new`.
        """
        return self.orient_new(*args, **kwargs)

    def copy(self, deep: bool = False, name: str | NoneType = None) -> "CartesianFrame":
        """
        Returns a shallow or deep copy of this object, depending of the
        argument `deepcopy` (default is False).

        Parameters
        ----------
        deep: bool, Optional
            If True, a deep copy is returned. Default is False.
        name: str, Optional
            The name of the copy. Default is None.
        """
        if deep:
            return self.__class__(dcopy(self.axes), origo=dcopy(self.origo), name=name)
        else:
            return self.__class__(self.axes, origo=self.origo, name=name)

    def deepcopy(self, name: str | NoneType = None) -> "CartesianFrame":
        """
        Returns a deep copy of the instance.

        Parameters
        ----------
        name: str, Optional
            The name of the copy. Default is None.
        """
        return self.copy(deep=True, name=name)
