from typing import (
    Tuple,
    ClassVar,
    Optional,
    TypeVar,
    Generic,
    Protocol,
    Union,
    Iterable,
)

from numpy import ndarray

from .polydata import PolyDataProtocol
from .pointdata import PointDataProtocol
from .geometry import GeometryProtocol
from .celldata import CellDataProtocol
from ..topoarray import TopologyArray

__all__ = ["PolyCellProtocol"]

MeshDataLike = TypeVar(
    "MeshDataLike", bound=PolyDataProtocol[PointDataProtocol, "PolyCellProtocol"]
)
PointDataLike = TypeVar("PointDataLike", bound=PointDataProtocol)


class PolyCellProtocol(
    CellDataProtocol[MeshDataLike, PointDataLike],
    Generic[MeshDataLike, PointDataLike],
    Protocol,
):
    """
    Base class for PolyCell objects.
    """

    label: ClassVar[Optional[str]] = None
    Geometry: ClassVar[GeometryProtocol]

    def local_coordinates(self) -> ndarray:
        """Ought to return the coordinates of the cells in their local
        coordinate systems."""
        ...

    def coords(self) -> ndarray:
        """Ought to return the coordiantes associated with the object."""
        ...

    def topology(self) -> TopologyArray:
        """Ought to return the topology associated with the object."""
        ...

    def measures(self) -> ndarray:
        """Ought to return meaninful measures for each cell."""
        ...

    def measure(self) -> float:
        """Ought to return a single measure for a collection of cells."""
        ...

    def to_simplices(self) -> Tuple[ndarray]:
        """Ought to return a triangular representation of the mesh."""
        raise NotImplementedError

    def jacobian_matrix(self) -> ndarray:
        """Ought to return meaninful measures for each cell."""
        ...

    def loc_to_glob(
        self, x: Union[Iterable, ndarray], ec: Optional[Union[ndarray, None]] = None
    ) -> ndarray:
        """
        Maps local coordinates in the master domain to global cooridnates.
        The basis of the transformation is 'ec', which is the node coordinates
        array of the cells of shape (nE, nNE, nD), where 'nE', 'nNE' and 'nD' are
        the number of cells, nodes per cell and local spatial dimensions.
        """
        ...
