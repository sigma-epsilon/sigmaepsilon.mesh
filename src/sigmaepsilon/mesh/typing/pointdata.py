from abc import abstractproperty

from numpy import ndarray

from sigmaepsilon.core.meta import ABCMeta_Weak

__all__ = ["PointDataBase"]


class ABC(metaclass=ABCMeta_Weak):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()


class PointDataBase(ABC):
    """
    Base class for PointData objects.
    """

    @abstractproperty
    def id(self) -> ndarray:
        """Ought to return global ids of the points."""
        ...

    @abstractproperty
    def frame(self) -> ndarray:
        """Ought to return a frame of reference."""
        ...

    @abstractproperty
    def x(self) -> ndarray:
        """Ought to return the coordinates of the associated pointcloud."""
        ...
