"""
These base classes meant to resolve circular references
while providing static hints.

This module must not have references from other parts of the library,
to make sure circular refrerences are all avoided.
"""
from abc import abstractmethod, abstractproperty
from typing import Union, Iterable

from numpy import ndarray

from linkeddeepdict import LinkedDeepDict
from sigmaepsilon.math.linalg.sparse import csr_matrix

from .topoarray import TopologyArray
from .akwrap import AkWrapper
from .abcdata import ABC_MeshData


class PointDataBase(AkWrapper, ABC_MeshData):
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


class CellDataBase(AkWrapper, ABC_MeshData):
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

    def to_triangles(self, *args, **kwargs) -> ndarray:
        """Ought to return a triangular representation of the mesh."""
        raise NotImplementedError

    def to_tetrahedra(self, *args, **kwargs) -> ndarray:
        """Ought to return a tetrahedral representation of the mesh."""
        raise NotImplementedError


class PolyDataBase(LinkedDeepDict):
    """
    Base class for PolyData objects.
    """

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
