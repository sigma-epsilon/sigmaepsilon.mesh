from typing import (
    Union,
    Callable,
    Optional,
    ClassVar,
    Iterable,
    Protocol,
    runtime_checkable,
)

from numpy import ndarray


__all__ = ["GeometryProtocol"]


@runtime_checkable
class GeometryProtocol(Protocol):
    """
    Protocol for Geometry classes.
    """

    number_of_nodes: ClassVar[int]
    number_of_spatial_dimensions: ClassVar[int]
    number_of_nodal_variables: ClassVar[int] = 1
    vtk_cell_id: ClassVar[Optional[int]] = None
    meshio_cell_id: ClassVar[Optional[str]] = None
    boundary_class: ClassVar[Optional["GeometryProtocol"]] = None
    shape_function_evaluator: ClassVar[Optional[Callable]] = None
    shape_function_matrix_evaluator: ClassVar[Optional[Callable]] = None
    shape_function_derivative_evaluator: ClassVar[Optional[Callable]] = None
    monomial_evaluator: ClassVar[Optional[Callable]] = None
    quadrature: ClassVar[Optional[dict]] = None

    @classmethod
    def master_coordinates(cls) -> ndarray:
        """
        Returns the coordinates of the master element.

        Returns
        -------
        numpy.ndarray
        """
        ...

    @classmethod
    def master_center(cls) -> ndarray:
        """
        Returns the coordinates of the master element.

        Returns
        -------
        numpy.ndarray
        """
        ...

    @classmethod
    def shape_function_values(
        cls, x: Union[float, Iterable[float]], *args, **kwargs
    ) -> ndarray:
        """
        Evaluates the shape functions at the specified locations.
        """
        ...

    @classmethod
    def shape_function_derivatives(
        cls, x: Union[float, Iterable[float]], *args, **kwargs
    ) -> ndarray:
        """
        Evaluates shape function derivatives wrt. the master element or the local
        coordinate frames of some cells. To control the behaviour, either 'jac' or 'wrt'
        can be provided.
        """
        ...

    @classmethod
    def shape_function_matrix(
        cls, x: Union[float, Iterable[float]], *args, N: Optional[int] = 1, **kwargs
    ) -> ndarray:
        """
        Evaluates the shape function matrix at the specified locations and
        an N number of nodal variables.
        """
        ...

    @classmethod
    def approximator(cls, *args, **kwargs) -> Callable:
        """
        Returns a callable object that can be used to approximate over
        nodal values of one or more cells.
        """
        ...
