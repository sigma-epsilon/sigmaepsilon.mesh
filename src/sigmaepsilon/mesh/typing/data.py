from typing import (
    Union,
    Iterable,
    Protocol,
    runtime_checkable,
    TypeVar,
    Generic,
    Optional,
    Tuple,
    ClassVar,
)

from numpy import ndarray

from sigmaepsilon.math.linalg import ReferenceFrame as FrameLike
from sigmaepsilon.math.linalg.sparse import csr_matrix

from .geometry import GeometryProtocol
from ..space import CartesianFrame
from ..topoarray import TopologyArray


__all__ = [
    "PointDataProtocol",
    "CellDataProtocol",
    "PolyCellProtocol",
    "PolyDataProtocol",
    "PolyDataLike",
    "PointDataLike",
    "CellDataLike",
    "PolyCellLike",
]


PolyDataLike = TypeVar("PolyDataLike", bound="PolyDataProtocol", covariant=True)
PointDataLike = TypeVar("PointDataLike", bound="PointDataProtocol", covariant=True)
CellDataLike = TypeVar("CellDataLike", bound="CellDataProtocol", covariant=True)
PolyCellLike = TypeVar("PolyCellLike", bound="PolyCellProtocol", covariant=True)


@runtime_checkable
class PointDataProtocol(Protocol):  # pragma: no cover
    """
    A protocol class for storing point-related data in a mesh.
    """

    @property
    def id(self) -> ndarray:
        """Ought to return global ids of the points as an 1d integer array."""
        ...

    @property
    def frame(self) -> FrameLike:
        """Ought to return a frame of reference."""
        ...

    @property
    def x(self) -> ndarray:
        """Ought to return the coordinates of the associated pointcloud
        as a 2d float array, where the first axis runs along the points."""
        ...

    def pull(self) -> ndarray:
        """Collects data at the points from the cells meeting at thenodes
        and aggregates it"""
        ...


@runtime_checkable
class CellDataProtocol(Generic[PolyDataLike, PointDataLike], Protocol):
    """
    A generic protocol class for storing cell-related data in a mesh.
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
    def nodes(self) -> ndarray:
        """
        Ought to return the topology of the cells as a 2d NumPy integer array.
        """
        ...

    @property
    def pointdata(self) -> PointDataLike:
        """Returns the hosting pointdata."""

    @property
    def container(self) -> PolyDataLike:
        """Returns the container object of the block."""

    @property
    def has_frames(self) -> bool:
        """
        Ought to return `True` if the cells are equipped with frames,
        `False` if they are not.
        """
        ...

    @property
    def has_nodes(self) -> bool:
        """
        Ought to return `True` if the cells are equipped with nodes,
        `False` if they are not.
        """
        ...


class PolyCellProtocol(
    CellDataProtocol[PolyDataLike, PointDataLike],
    Generic[PolyDataLike, PointDataLike],
    Protocol,
):  # pragma: no cover
    """
    A generic protocol class for polygonal cell containers.
    """

    label: ClassVar[Optional[str]] = None
    Geometry: ClassVar[GeometryProtocol]

    @property
    def db(self) -> CellDataProtocol[PolyDataLike, PointDataLike]:
        """
        Returns the database of the block.
        """
        ...

    @db.setter
    def db(self, value: CellDataProtocol[PolyDataLike, PointDataLike]) -> None:
        """
        Sets the database of the block.
        """
        ...

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

    def normals(self) -> ndarray:
        """Ought to return the normal vectors of the surface of the mesh."""
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


@runtime_checkable
class PolyDataProtocol(
    Generic[PointDataLike, PolyCellLike], Protocol
):  # pragma: no cover
    """A generic protocol class for polygonal mesh containers."""

    @property
    def frame(self) -> Union[FrameLike, CartesianFrame]:
        """Ought to return the frame of the attached pointdata"""

    @property
    def pointdata(self) -> PointDataLike:
        """Ought to return the attached pointdata."""
        ...

    @property
    def celldata(self) -> PolyCellLike:
        """Ought to return the attached celldata."""
        ...

    def root(self: PolyDataLike, *args, **kwargs) -> PolyDataLike:
        """Ought to return the top level container."""
        ...

    def source(self: PolyDataLike, *args, **kwargs) -> Union[PolyDataLike, None]:
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

    def pointblocks(self: PolyDataLike, *args, **kwargs) -> Iterable[PolyDataLike]:
        """
        Ought to return PolyData blocks with attached PointData.
        """
        ...

    def cellblocks(self: PolyDataLike, *args, **kwargs) -> Iterable[PolyDataLike]:
        """
        Ought to return PolyData blocks with attached CellData.
        """
        ...
