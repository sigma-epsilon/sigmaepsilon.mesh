from typing import (
    Protocol,
    runtime_checkable,
    TypeVar,
)

from numpy import ndarray

from .pointdata import PointDataProtocol

__all__ = ["CellDataProtocol"]

MeshDataLike = TypeVar("MeshDataLike")
PointDataLike = TypeVar("PointDataLike", bound=PointDataProtocol)


@runtime_checkable
class CellDataProtocol(Protocol[MeshDataLike, PointDataLike]):
    """
    Base class for CellData objects.
    """

    @property
    def id(self) -> ndarray:
        """Ought to return global ids of the cells."""
        ...

    @property
    def frames(self) -> ndarray:
        """Ought to return the reference frames of the cells."""
        ...

    @property
    def pointdata(self) -> PointDataLike:
        """Returns the hosting pointdata."""

    @property
    def container(self) -> MeshDataLike:
        """Returns the container object of the block."""
