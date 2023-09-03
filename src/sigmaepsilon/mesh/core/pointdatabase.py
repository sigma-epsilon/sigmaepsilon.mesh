from abc import abstractproperty

from numpy import ndarray

from .akwrapper import AkWrapper


class PointDataBase(AkWrapper):
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