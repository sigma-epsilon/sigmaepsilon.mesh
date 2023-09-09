from abc import abstractmethod, abstractproperty

from numpy import ndarray

from sigmaepsilon.core.meta import ABCMeta_Weak
from ..topoarray import TopologyArray

__all__ = ["CellDataBase"]


class ABC(metaclass=ABCMeta_Weak):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """

    __slots__ = ()


class CellDataBase(ABC):
    """
    Base class for CellData objects.
    """

    @abstractproperty
    def id(self) -> ndarray:
        """Ought to return global ids of the cells."""
        ...

    @abstractmethod
    def coords(self, *args, **kwargs) -> ndarray:
        """Ought to return the coordiantes associated with the object."""
        ...

    @abstractmethod
    def topology(self, *args, **kwargs) -> TopologyArray:
        """Ought to return the topology associated with the object."""
        ...

    @abstractmethod
    def measures(self, *args, **kwargs) -> ndarray:
        """Ought to return meaninful measures for each cell."""
        ...

    @abstractmethod
    def measure(self, *args, **kwargs) -> ndarray:
        """Ought to return a single measure for a collection of cells."""
        ...

    def to_simplices(self, *args, **kwargs) -> ndarray:
        """Ought to return a triangular representation of the mesh."""
        raise NotImplementedError
