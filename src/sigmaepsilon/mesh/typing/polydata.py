from typing import Union, Iterable, Protocol, runtime_checkable, TypeVar, Generic

from numpy import ndarray

from sigmaepsilon.math.linalg import ReferenceFrame as FrameLike
from sigmaepsilon.math.linalg.sparse import csr_matrix

from ..topoarray import TopologyArray

__all__ = ["PolyDataProtocol"]


PD = TypeVar("PD")
CD = TypeVar("CD")


@runtime_checkable
class PolyDataProtocol(Generic[PD, CD], Protocol):
    """Protocol for polygonal meshes."""

    @property
    def frame(self) -> FrameLike:
        """Ought to return the frame of the attached pointdata"""
        
    @property
    def pointdata(self) -> PD:
        """Ought to return the attached pointdata."""
        ...

    @property
    def celldata(self) -> CD:
        """Ought to return the attached celldata."""
        ...

    def source(self, *args, **kwargs) -> Union["PolyDataProtocol", None]:
        """Ought to return the object that holds onto point data."""
        ...

    def coords(self, *args, **kwargs) -> ndarray:
        """Ought to return the coordiantes associated with the object."""
        ...

    def topology(self, *args, **kwargs) -> Union[ndarray, TopologyArray]:
        """Ought to return the topology associated with the object."""
        ...

    def nodal_distribution_factors(self, *args, **kwargs) -> Union[ndarray, csr_matrix]:
        """
        Ought to return nodal distribution factors for every node
        of every cell in the block.
        """
        ...

    def pointblocks(self, *args, **kwargs) -> Iterable["PolyDataProtocol"]:
        """
        Ought to return PolyData blocks with attached PointData.
        """
        ...

    def cellblocks(self, *args, **kwargs) -> Iterable["PolyDataProtocol"]:
        """
        Ought to return PolyData blocks with attached CellData.
        """
        ...
