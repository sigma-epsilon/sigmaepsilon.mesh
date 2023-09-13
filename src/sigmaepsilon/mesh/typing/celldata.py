from typing import (
    Protocol,
    runtime_checkable,
    TypeVar,
    Generic,
)

from numpy import ndarray

__all__ = ["CellDataProtocol"]

MeshDataLike = TypeVar("MeshDataLike")
PointDataLike = TypeVar("PointDataLike")


@runtime_checkable
class CellDataProtocol(Generic[MeshDataLike, PointDataLike], Protocol):
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
