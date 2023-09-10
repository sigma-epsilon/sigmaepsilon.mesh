from abc import abstractmethod, abstractproperty
from typing import Union, Iterable, Generic, TypeVar

from numpy import ndarray

from sigmaepsilon.core.meta import ABCMeta_Weak
from sigmaepsilon.math.linalg.sparse import csr_matrix

from ..topoarray import TopologyArray
from .pointdata import PointDataType
from .celldata import CellDataType

PD = TypeVar("PD", bound=PointDataType)
CD = TypeVar("CD", bound=CellDataType)

__all__ = ["PolyDataType"]


class ABC(metaclass=ABCMeta_Weak):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """

    __slots__ = ()


class PolyDataType(Generic[PD, CD], ABC):
    """
    Base class for PolyData objects.
    """

    @abstractproperty
    def frame(self) -> ndarray:
        """Ought to return a frame of reference."""
        ...

    @abstractmethod
    def source(self, *args, **kwargs) -> "PolyDataType":
        """Ought to return the object that holds onto point data."""
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
    def nodal_distribution_factors(self) -> Union[ndarray, csr_matrix]:
        """
        Ought to return nodal distribution factors for every node
        of every cell in the block.
        """
        ...

    @abstractmethod
    def pointblocks(self) -> Iterable[PD]:
        """
        Ought to return PolyData blocks with attached PointData.
        """
        ...

    @abstractmethod
    def cellblocks(self) -> Iterable[CD]:
        """
        Ought to return PolyData blocks with attached CellData.
        """
        ...