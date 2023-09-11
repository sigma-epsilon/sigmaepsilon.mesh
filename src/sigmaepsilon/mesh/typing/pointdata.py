from typing import Protocol, runtime_checkable

from numpy import ndarray

from sigmaepsilon.math.linalg import ReferenceFrame as FrameLike

__all__ = ["PointDataProtocol"]


@runtime_checkable
class PointDataProtocol(Protocol):
    """
    Base class for PointData objects.
    """

    @property
    def id(self) -> ndarray:
        """Ought to return global ids of the points."""
        ...

    @property
    def frame(self) -> FrameLike:
        """Ought to return a frame of reference."""
        ...

    @property
    def x(self) -> ndarray:
        """Ought to return the coordinates of the associated pointcloud."""
        ...
