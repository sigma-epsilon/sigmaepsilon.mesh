from abc import abstractmethod, abstractproperty
from typing import Union, Iterable

from numpy import ndarray

from sigmaepsilon.math.linalg.sparse import csr_matrix

from ..topoarray import TopologyArray
from .pointdata import PointDataBase
from .celldata import CellDataBase


class PolyDataBase:
    """
    Base class for PolyData objects.
    """
    
    @abstractproperty
    def frame(self) -> ndarray:
        """Ought to return a frame of reference."""
        ...

    @abstractmethod
    def source(self, *args, **kwargs) -> "PolyDataBase":
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
    def pointblocks(self) -> Iterable[PointDataBase]:
        """
        Ought to return PolyData blocks with attached PointData.
        """
        ...

    @abstractmethod
    def cellblocks(self) -> Iterable[CellDataBase]:
        """
        Ought to return PolyData blocks with attached CellData.
        """
        ...
        