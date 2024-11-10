# -*- coding: utf-8 -*-
from typing import Tuple, Iterable
from types import NoneType
from numbers import Number

import numpy as np
from numpy import ndarray

from .data import PolyData, PointData
from .space import CartesianFrame
from .cells import H8
from .grid import grid
from .utils.topology import detach_mesh_bulk
from .utils.topology.cic import H8_in_TET4, H8_in_T3

__all__ = ["voxelize_cylinder", "voxelize_TET4_H8", "voxelize_T3_H8"]


def voxelize_cylinder(
    radius: ndarray | Number | Iterable[Number],
    height: float,
    size: float,  # voxel edge length
    frame: CartesianFrame | NoneType = None,
) -> Tuple[ndarray[float], ndarray[int]]:
    """
    Returns raw mesh data of a voxelized cylinder.

    Parameters
    ----------
    radius: numpy.ndarray or Number or Iterable[Number]
        Radius of the cylinder. If a NumPy array is passed, it should
        contain two values, the inner and outer radii. If a single
        number is passed, it is assumed to be the outer radius.
    height: float
        Height of the cylinder.
    size: float
        Size of the voxel grid.
    frame: CartesianFrame or None, Optional
        Cartesian frame of the voxel grid. Default is None.

    Returns
    -------
    tuple[numpy.ndarray[float], numpy.ndarray[int]]
        The coordinates and topology of the hexahedral mesh.

    Example
    -------
    The following example shows how to create a voxelized cylinder.

    >>> import numpy as np
    >>> from sigmaepsilon.mesh.voxelize import voxelize_cylinder
    >>> radius = 1
    >>> height = 2
    >>> size = 0.1
    >>> coords, topo = voxelize_cylinder(radius, height, size)

    """
    if isinstance(radius, int):
        radius = np.array([0, radius])
    elif not isinstance(radius, ndarray):
        radius = np.array(radius)

    nXY = int(np.ceil(2 * radius[1] / size))
    nZ = int(np.ceil(height / size))
    Lxy, Lz = 2 * radius[1], height
    coords, topo = grid(
        size=(Lxy, Lxy, Lz), shape=(nXY, nXY, nZ), eshape="H8", centralize=True
    )
    frame = CartesianFrame(dim=3) if frame is None else frame
    pd = PointData(coords=coords, frame=frame)
    cd = H8(topo=topo, frames=frame)
    c = PolyData(pd, cd).centers()
    r = (c[:, 0] ** 2 + c[:, 1] ** 2) ** (1 / 2)
    cond = (r <= radius[1]) & (r >= radius[0])
    inds = np.where(cond)[0]
    return detach_mesh_bulk(coords, topo[inds])


def _check_voxelization_arguments(coords, topo, shape, resolution, num_node) -> None:
    if shape is not None and resolution is not None:
        raise ValueError("Both `shape` and `resolution` cannot be provided.")

    if shape is None and resolution is None:
        raise ValueError("Either `shape` or `resolution` should be provided.")

    if isinstance(shape, Iterable):
        if not len(shape) == 3:
            raise ValueError("The shape should be an iterable of length 3.")

    if isinstance(resolution, Iterable):
        if not len(resolution) == 3:
            raise ValueError("`resolution` should be an iterable of length 3.")

    if not isinstance(coords, ndarray):
        raise TypeError("The `coords` should be a NumPy array.")

    if not isinstance(topo, ndarray):
        raise TypeError("The `topo` should be a NumPy array.")

    if not len(coords.shape) == 2:
        raise ValueError("The `coords` should be a 2d array.")

    if not len(topo.shape) == 2:
        raise ValueError("The `topo` should be a 2d array.")

    if not coords.shape[1] == 3:
        raise ValueError("The `coords` should have 3 columns.")

    if not topo.shape[1] == num_node:
        raise ValueError(f"The `topo` should have {num_node} columns.")


def voxelize_TET4_H8(
    coords_TET4: ndarray,
    topo_TET4: ndarray,
    shape: Iterable[int] | int | NoneType = None,
    resolution: Iterable[float] | float | NoneType = None,
    k_max: int = 10,
) -> Tuple[ndarray[float], ndarray[int]]:
    """
    Returns a voxelized version of a tetrahadral mesh.

    The voxelization is carried out by creating a hexahedral grid that
    covers the bounding box of the input mesh. The function then checks
    which hexahedrons are intersected by the input mesh and returns the
    coordinates and topology of the intersected hexahedrons. To make the calculation
    faster, the function uses a k-d tree to find the nearest neighbours.
    The maximum number of nearest neighbours to consider can be controlled with
    the `k_max` parameter.

    .. note::
        The function is not guaranteed to work for all types of meshes. If the
        input mesh is irregular, the function may not be able to find the correct
        voxelization.

    Parameters
    ----------
    coords_TET4: numpy.ndarray[float]
        2d NumPy array of the coordinates of the nodes of the TET4 cells.
    topo_TET4: numpy.ndarray[int]
        2d NumPy array of the topology of the TET4 cells.
    shape: Iterable[int] | int, Optional
        The shape of the voxel grid specified as a single integer or an iterable
        of integers (of length 3). The shape of the grid is the number of
        hexahedrons in each direction. If the value is a single integer, the grid
        will have an equal number of hexahedrons in all 3 spatial directions.
        Default is None.
    resolution: Iterable[float] | float, Optional
        Resolution of the voxel grid specified as a single number or an iterable
        of length 3. The resolution expresses the side lengths of the hexahedron cells.
        If only a single number is provided, the hexahedrons will have uniform edge lengths
        (a voxel). Default is None.
    k_max: int, Optional
        Maximum number of nearest neighbours to consider. Default is 10.
        If the number of TET4 cells is less than `k_max`, the function
        will use the number of TET4 cells as the `k` parameter.

    Returns
    -------
    tuple[numpy.ndarray[float], numpy.ndarray[int]]
        The coordinates and topology of the hexahedral mesh.

    Raises
    ------
    ValueError
        Upon invalid input values.
    TypeError
        Upon invalid input types.

    Example
    -------
    The following example shows how to voxelize a TET4 mesh.

    >>> import numpy as np
    >>> from sigmaepsilon.mesh.voxelize import voxelize_TET4_H8
    >>> coords_TET4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(float)
    >>> topo_TET4 = np.array([[0, 1, 2, 3]])
    >>> shape = (10, 10, 10)
    >>> coords_H8, topo_H8 = voxelize_TET4_H8(coords_TET4, topo_TET4, shape=shape)

    """
    _check_voxelization_arguments(coords_TET4, topo_TET4, shape, resolution, 4)

    size_x = np.max(coords_TET4[:, 0]) - np.min(coords_TET4[:, 0])
    size_y = np.max(coords_TET4[:, 1]) - np.min(coords_TET4[:, 1])
    size_z = np.max(coords_TET4[:, 2]) - np.min(coords_TET4[:, 2])
    size = (size_x, size_y, size_z)

    shift_x = np.min(coords_TET4[:, 0])
    shift_y = np.min(coords_TET4[:, 1])
    shift_z = np.min(coords_TET4[:, 2])
    shift = (shift_x, shift_y, shift_z)

    if shape is None and resolution is not None:
        n_x = int(np.ceil(size_x / resolution))
        n_y = int(np.ceil(size_y / resolution))
        n_z = int(np.ceil(size_z / resolution))
        shape = (n_x, n_y, n_z)

    coords_H8, topo_H8 = grid(size=size, shape=shape, eshape="H8", shift=shift)
    H8_bool = H8_in_TET4(coords_H8, topo_H8, coords_TET4, topo_TET4, k=k_max)
    return coords_H8, topo_H8[H8_bool]


def voxelize_T3_H8(
    coords_T3: ndarray[float],
    topo_T3: ndarray[int],
    shape: Iterable[int] | int | NoneType = None,
    resolution: Iterable[float] | float | NoneType = None,
    k_max: int = 10,
) -> Tuple[ndarray[float], ndarray[int]]:
    """
    Returns a voxelized version of a triangular mesh.

    The voxelization is carried out by creating a hexahedral grid that
    covers the bounding box of the input mesh. The function then checks
    which hexahedrons are intersected by the input mesh and returns the
    coordinates and topology of the intersected hexahedrons. To make the calculation
    faster, the function uses a k-d tree to find the nearest neighbours.
    The maximum number of nearest neighbours to consider can be controlled with
    the `k_max` parameter.

    .. note::
        The function is not guaranteed to work for all types of meshes. If the
        input mesh is irregular, the function may not be able to find the correct
        voxelization.

    .. versionadded:: 3.1.0

    Parameters
    ----------
    coords_T3: numpy.ndarray[float]
        2d NumPy array of the coordinates of the nodes of the T3 cells.
    topo_T3: numpy.ndarray[int]
        2d NumPy array of the topology of the T3 cells.
    shape: Iterable[int] | int, Optional
        The shape of the voxel grid specified as a single integer or an iterable
        of integers (of length 3). The shape of the grid is the number of
        hexahedrons in each direction. If the value is a single integer, the grid
        will have an equal number of hexahedrons in all 3 spatial directions.
        Default is None.
    resolution: Iterable[float] | float, Optional
        Resolution of the voxel grid specified as a single number or an iterable
        of length 3. The resolution expresses the side lengths of the hexahedron cells.
        If only a single number is provided, the hexahedrons will have uniform edge lengths
        (a voxel). Default is None.
    k_max: int, Optional
        Maximum number of nearest neighbours to consider. Default is 10.
        If the number of T3 cells is less than `k_max`, the function
        will use the number of T3 cells as the `k` parameter.

    Returns
    -------
    tuple[numpy.ndarray[float], numpy.ndarray[int]]
        The coordinates and topology of the hexahedral mesh.

    Raises
    ------
    ValueError
        Upon invalid input values.
    TypeError
        Upon invalid input types.

    Example
    -------
    The following example shows how to voxelize a T3 mesh.

    >>> import numpy as np
    >>> from sigmaepsilon.mesh.voxelize import voxelize_T3_H8
    >>> coords_T3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(float)
    >>> topo_T3 = np.array([[0, 1, 2]])
    >>> shape = (10, 10, 10)
    >>> coords_H8, topo_H8 = voxelize_T3_H8(coords_T3, topo_T3, shape=shape)

    """
    _check_voxelization_arguments(coords_T3, topo_T3, shape, resolution, 3)

    size_x = np.max(coords_T3[:, 0]) - np.min(coords_T3[:, 0])
    size_y = np.max(coords_T3[:, 1]) - np.min(coords_T3[:, 1])
    size_z = np.max(coords_T3[:, 2]) - np.min(coords_T3[:, 2])
    size = (size_x, size_y, size_z)

    shift_x = np.min(coords_T3[:, 0])
    shift_y = np.min(coords_T3[:, 1])
    shift_z = np.min(coords_T3[:, 2])
    shift = (shift_x, shift_y, shift_z)

    if isinstance(resolution, (float, int)):
        resolution = 3 * (float(resolution),)

    if isinstance(shape, int):
        shape = 3 * (shape,)

    if shape is None and resolution is not None:
        n_x = int(np.ceil(size_x / resolution[0]))
        n_y = int(np.ceil(size_y / resolution[1]))
        n_z = int(np.ceil(size_z / resolution[2]))
        shape = (n_x, n_y, n_z)

    shape = (max(1, shape[0]), max(1, shape[1]), max(1, shape[2]))

    coords_H8, topo_H8 = grid(size=size, shape=shape, eshape="H8", shift=shift)
    H8_bool = H8_in_T3(coords_H8, topo_H8, coords_T3, topo_T3, k=k_max)
    return coords_H8, topo_H8[H8_bool]
