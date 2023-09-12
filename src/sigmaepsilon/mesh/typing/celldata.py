from typing import (
    Protocol,
    runtime_checkable,
    Tuple,
    ClassVar,
    Optional,
    TypeVar,
    Generic,
)

from numpy import ndarray

from .polydata import PolyDataProtocol
from .pointdata import PointDataProtocol
from .geometry import GeometryProtocol
from ..topoarray import TopologyArray

__all__ = ["CellDataProtocol"]

MD = TypeVar("MD", bound=PolyDataProtocol[PointDataProtocol, "CellDataProtocol"])
PD = TypeVar("PD", bound=PointDataProtocol)


@runtime_checkable
class CellDataProtocol(Generic[MD, PD], Protocol):
    """
    Base class for CellData objects.
    """

    label: ClassVar[Optional[str]] = None
    Geometry: ClassVar[GeometryProtocol]

    @property
    def id(self) -> ndarray:
        """Ought to return global ids of the cells."""
        ...

    @property
    def frames(self) -> ndarray:
        """Ought to return the reference frames of the cells."""
        ...

    @property
    def pointdata(self) -> PD:
        """Returns the hosting pointdata."""
    
    @property
    def container(self) -> MD:
        """Returns the container object of the block."""

    def coords(self, *args, **kwargs) -> ndarray:
        """Ought to return the coordiantes associated with the object."""
        ...

    def topology(self, *args, **kwargs) -> TopologyArray:
        """Ought to return the topology associated with the object."""
        ...

    def measures(self, *args, **kwargs) -> ndarray:
        """Ought to return meaninful measures for each cell."""
        ...

    def measure(self, *args, **kwargs) -> float:
        """Ought to return a single measure for a collection of cells."""
        ...

    def to_simplices(self, *args, **kwargs) -> Tuple[ndarray]:
        """Ought to return a triangular representation of the mesh."""
        raise NotImplementedError

    def jacobian_matrix(self, *args, **kwargs) -> ndarray:
        """Ought to return meaninful measures for each cell."""
        ...
