from typing import Iterable, Union, Any, Callable
from functools import partial
import warnings

import numpy as np
from numpy import ndarray

from sigmaepsilon.core.warning import SigmaEpsilonPerformanceWarning
from sigmaepsilon.math.linalg import generalized_inverse

from .utils.cells.approximator import _approximate_multi

__all__ = ["LagrangianCellApproximator"]


def _get_shape_function_evaluator(cls: Any) -> Callable:
    try:
        if hasattr(cls, "Geometry"):
            shp_fnc = cls.Geometry.shape_function_values
        else:
            shp_fnc = cls.shape_function_values
        return shp_fnc
    except AttributeError:
        raise TypeError(
            "Invalid type. The cell must be an instance of PolyCell"
            " or implement the PolyCellGeometry protocol."
        )


def _approximator(
    cls: Any,
    *,
    x_source: Iterable | None = None,
    shp_source_inverse: Iterable | None = None,
    values_source: Iterable = None,
    x_target: Iterable | None = None,
    axis: int | None = None,
) -> Union[float, ndarray]:
    """
    Returns interpolated values from a set of known points and values.
    """
    shp_fnc: Callable = _get_shape_function_evaluator(cls)

    if shp_source_inverse is None:
        assert isinstance(x_source, Iterable)
        shp_source = shp_fnc(x_source)  # (nP_source, nNE)

        num_rows, num_columns = shp_source.shape
        rank = np.linalg.matrix_rank(shp_source)
        square_and_full_rank = (
            num_rows == num_columns
        ) and rank == num_columns == num_rows
        if not square_and_full_rank:  # pragma: no cover
            warnings.warn(
                "The approximation involves the calculation of a generalized inverse "
                "which probably results in loss of precision.",
                SigmaEpsilonPerformanceWarning,
            )

        shp_source_inverse = generalized_inverse(shp_source)

    if not isinstance(values_source, ndarray):
        values_source = np.array(values_source)

    multi_dimensional = len(values_source.shape) > 1
    shp_target = shp_fnc(x_target)  # (nP_target, nNE)

    if len(values_source.shape) > 1 and axis is None:
        axis = -1

    if isinstance(axis, int):
        if not multi_dimensional:
            raise ValueError(
                "If 'axis' is provided, 'values_source' must be multidimensional."
            )

    if multi_dimensional:
        values_source = np.moveaxis(values_source, axis, -1)
        *array_axes, nP_source = values_source.shape
        nX = np.prod(array_axes)
        values_source = np.reshape(values_source, (nX, nP_source))
        nP_target = shp_target.shape[0]
        result = np.zeros((nX, nP_target))
        # (nP_T x nNE) @ (nNE x nP_S) @ (nX, nP_S) -> (nX, nP_T)
        _approximate_multi(shp_target @ shp_source_inverse, values_source, result)
        result = np.reshape(result, tuple(array_axes) + (nP_target,))
        result = np.moveaxis(result, -1, axis)
        values_source = np.moveaxis(values_source, -1, axis)
    else:
        # (nP_T x nNE) @ (nNE x nP_S) @ (nNE)
        result = shp_target @ shp_source_inverse @ values_source

    return result


class LagrangianCellApproximator:
    """
    An approximator for Lagrangian cells. It can be constructed directly or using
    a cell class from the library.

    Parameters
    ----------
    cell_class: :class:`~sigmaepsilon.mesh.data.polycell.PolyCell`
        A Lagrangian cell class that provides the batteries for interpolation.
        The capabilities of this class determines the nature and accuracy of the
        interpolation/extrapolation.
    x_source: Iterable, Optional
        The process of interpolation involves calculating the inverse of a matrix.
        If you plan to use an interpolator many times using the same set of source points,
        it is a good idea to fed the instance with these coordinates at the time of
        instantiation. This way the expensive part of the calculation is only done once,
        and subsequent evaluations are faster. Default is None.

    Notes
    -----
    Depending on the number of nodes of the element (hence the order of the approximation
    functions), the approximation may be exact interpolation or some kind of regression.
    For instance, if you try to extrapolate from 3 values using a 2-noded line element,
    the approximator is overfitted and the approximation is an ecaxt one only if all the
    data values fit a line perfectly.

    Examples
    --------
    Create an approximator using 8-noded hexahedrons.

    >>> from sigmaepsilon.mesh import LagrangianCellApproximator
    >>> from sigmaepsilon.mesh.cells import H8
    >>> approximator = LagrangianCellApproximator(H8)

    The data to feed the approximator:

    >>> source_coordinates = H8.Geometry.master_coordinates() / 2
    >>> source_values = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> target_coordinates = H8.Geometry.master_coordinates() * 2

    The desired data at the target locations:

    >>> target_values = approximator(
    ...     source=source_coordinates,
    ...     target=target_coordinates,
    ...     values=source_values
    ... )

    This approximator can also be created using the class diretly:

    >>> approximator = H8.Geometry.approximator()

    If you want to reuse the approximator with the same set of source coordinates
    many times, you can feed these points to the approximator at instance creation:

    >>> approximator = H8.Geometry.approximator(source_coordinates)
    >>> approximator = LagrangianCellApproximator(H8, source_coordinates)

    Then, only source values and target coordinates have to be provided for
    approximation to happen (in fact, you will get an Exception of you provide
    source coordinates both at creation and approximator):

    >>> target_values = approximator(
    ...     target=target_coordinates,
    ...     values=source_values
    ... )

    To approximator multidimensional data, you have to carefully organize the
    input values for utmost performance. The memory layout is optimal if the axis
    that goes along the input points is the last one:

    >>> approximator = H8.Geometry.approximator()

    >>> source_values = np.random.rand(10, 2, 8)
    >>> approximator(
    ...     source=source_coordinates,
    ...     values=source_values,
    ...     target=target_coordinates[:3]
    ... ).shape
    (10, 2, 3)

    If it is not the last axis, you can use the 'axis' parameter:

    >>> source_values = np.random.rand(8, 2, 10)
    >>> approximator(
    ...     source=source_coordinates,
    ...     values=source_values,
    ...     target=target_coordinates[:3],
    ...     axis=0,
    ... ).shape
    (3, 2, 10)
    """

    approximator_function: Callable = _approximator

    def __init__(self, cell_class: Any, x_source: Iterable = None):
        shp_fnc: Callable = _get_shape_function_evaluator(cell_class)

        if isinstance(x_source, Iterable):
            shp_source = shp_fnc(np.array(x_source))  # (nP_source, nNE)
            self._source_shp_inverse = generalized_inverse(shp_source)
        else:
            self._source_shp_inverse = None

        self._approximator = partial(
            self.__class__.approximator_function,
            cell_class,
            shp_source_inverse=self._source_shp_inverse,
        )

    def __call__(
        self,
        *,
        values: Iterable,
        target: Iterable,
        source: Iterable = None,
        axis: int = None,
    ) -> ndarray:
        if source is not None and self._source_shp_inverse is not None:
            raise Exception("The approximator is already fed with source coordinates.")

        return self._approximator(
            x_source=source,
            x_target=target,
            values_source=values,
            axis=axis,
            shp_source_inverse=self._source_shp_inverse,
        )
