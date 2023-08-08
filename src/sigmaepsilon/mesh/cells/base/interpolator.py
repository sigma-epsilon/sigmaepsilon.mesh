from typing import Iterable, Union, Any
from functools import partial

import numpy as np
from numpy import ndarray

from sigmaepsilon.math.linalg import generalized_inverse
from ...utils.cells.interpolator import _interpolate_multi

__all__ = ["LagrangianCellInterpolator"]


def _interpolator(
    cls,
    *,
    x_source: Iterable = None,
    shp_source_inverse: Iterable = None,
    values_source: Iterable = None,
    x_target: Iterable = None,
    axis: int = None,
) -> Union[float, ndarray]:
    """
    Returns interpolated values from a set of known points and values.
    """
    if shp_source_inverse is None:
        assert isinstance(x_source, Iterable)
        shp_source = cls.shape_function_values(x_source)  # (nP_source, nNE)
        shp_source_inverse = generalized_inverse(shp_source)

    if not isinstance(values_source, ndarray):
        values_source = np.array(values_source)

    multi_dimensional = len(values_source.shape) > 1
    shp_target = cls.shape_function_values(x_target)  # (nP_target, nNE)

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
        # (nP_T x nNE) @ (nNE x nP_S) @ (nX, nP_S)
        _interpolate_multi(shp_target @ shp_source_inverse, values_source, result)
        result = np.reshape(result, tuple(array_axes) + (nP_target,))
        result = np.moveaxis(result, -1, axis)
        values_source = np.moveaxis(values_source, -1, axis)
    else:
        # (nP_T x nNE) @ (nNE x nP_S) @ (nNE)
        result = shp_target @ shp_source_inverse @ values_source

    return result


class LagrangianCellInterpolator:
    """
    An interpolator for Lagrangian cells. It can be constructed directly or using
    a cell class from the library.

    Parameters
    ----------
    cell_class: :class:`~sigmaepsilon.mesh.cells.base.cell.PolyCell`
        A Lagrangian cell class that provides the batteries for interpolation.
        The capabilities of this class determines the nature and accuracy of the
        interpolation/extrapolation.
    x_source: Iterable, Optional
        The process of interpolation involves calculating the inverse of a matrix.
        If you plan to use an interpolator many times using the same set of source points,
        it is a good idea to fed the instance with these coordinates at the time of
        instantiation. This way the expensive part of the calculation is only done once,
        and subsequent evaluations are faster. Default is None.

    Examples
    --------
    Create an interpolator using 8-noded hexahedrons.

    >>> from sigmaepsilon.mesh.cells import LagrangianCellInterpolator, H8
    >>> interpolator = LagrangianCellInterpolator(H8)

    The data to feed the interpolator:

    >>> source_coordinates = H8.master_coordinates() / 2
    >>> source_values = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> target_coordinates = H8.master_coordinates() * 2

    The desired data at the target locations:

    >>> target_values = interpolator(
    ...     source=source_coordinates,
    ...     target=target_coordinates,
    ...     values=source_values
    ... )

    This interpolator can also be created using the class diretly:

    >>> interpolator = H8.interpolator()

    If you want to reuse the interpolator with the same set of source coordinates
    many times, you can feed these points to the interpolator at instance creation:

    >>> interpolator = H8.interpolator(source_coordinates)
    >>> interpolator = LagrangianCellInterpolator(H8, source_coordinates)

    Then, only source values and target coordinates have to be provided for
    interpoaltion to happen (in fact, you will get an Exception of you provide
    source coordinates both at creation and interpolation):

    >>> target_values = interpolator(
    ...     target=target_coordinates,
    ...     values=source_values
    ... )
    
    To interpolate multidimensional data, you have to carefully organize the
    input values for utmost performance. The memory layout is optimal if the axis
    that goes along the input points is the last one:
    
    >>> interpolator = H8.interpolator()
    
    >>> source_values = np.random.rand(10, 2, 8)
    >>> interpolator(
    ...     source=source_coordinates, 
    ...     values=source_values, 
    ...     target=target_coordinates[:3]
    ... ).shape
    (10, 2, 3)
    
    If it is not the last axis, you can use the 'axis' parameter:
    
    >>> source_values = np.random.rand(8, 2, 10)
    >>> interpolator(
    ...     source=source_coordinates, 
    ...     values=source_values, 
    ...     target=target_coordinates[:3],
    ...     axis=0,
    ... ).shape
    (3, 2, 10)
    """

    def __init__(self, cell_class: Any, x_source: Iterable = None):
        if not hasattr(cell_class, "shape_function_values"):
            raise TypeError("'cell_class' must be a cell class")

        if isinstance(x_source, Iterable):
            shp_source = cell_class.shape_function_values(
                np.array(x_source)
            )  # (nP_source, nNE)
            self._source_shp_inverse = generalized_inverse(shp_source)
        else:
            self._source_shp_inverse = None

        self._interpolator = partial(
            _interpolator, cell_class, shp_source_inverse=self._source_shp_inverse
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
            raise Exception("The interpolator is already fed with source coordinates.")

        return self._interpolator(
            x_source=source,
            x_target=target,
            values_source=values,
            axis=axis,
            shp_source_inverse=self._source_shp_inverse,
        )
