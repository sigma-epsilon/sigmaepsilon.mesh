# -*- coding: utf-8 -*-
from typing import Tuple, Optional, Union, Iterable
from numbers import Number

import numpy as np
from numpy import ndarray

from sigmaepsilon.math.utils import atleast2d

from .data import PolyData, PointData
from .space import CartesianFrame
from .cells import H8
from .grid import grid
from .utils.topology import detach_mesh_bulk
from .utils import cell_centers_bulk, cells_coords
from .utils.knn import k_nearest_neighbours
from .utils.cic import H8_in_TET4_bulk_knn


__all__ = ["voxelize_cylinder", "voxelize_TET4_H8"]


def voxelize_cylinder(
    radius: ndarray | Number | Iterable[Number],
    height: float,
    size: float,  # voxel edge length
    frame: Optional[Union[CartesianFrame, None]] = None,
) -> Tuple[ndarray, ndarray]:
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


def voxelize_TET4_H8(
    coords_TET4: ndarray,
    topo_TET4: ndarray,
    shape: tuple | None = None,
    resolution: float | None = None,
    k_max: int = 4,
    tol: float = 1e-12,
) -> Tuple[ndarray, ndarray]:
    """
    Returns a voxelized version of a tetrahadral mesh.

    The function is expected to behave well, if the input mesh is
    regular. If it contains extremely skew cells, the function may
    struggle to find the correct voxelization.

    Parameters
    ----------
    coords_TET4: numpy.ndarray
        2d NumPy array of the coordinates of the nodes of the TET4 cells.
    topo_TET4: numpy.ndarray
        2d NumPy array of the topology of the TET4 cells.
    shape: tuple, Optional
        Tuple of the shape of the voxel grid. Default is None.
    resolution: float, Optional
        Resolution of the voxel grid. Default is None.
    k_max: int, Optional
        Maximum number of nearest neighbours to consider. Default is 4.
        If the number of TET4 cells is less than `k_max`, the function
        will use the number of TET4 cells as the `k` parameter.
    tol: float, Optional
        Tolerance to consider a point inside a cell. Default is 1e-12.

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
    centers_H8 = cell_centers_bulk(coords_H8, topo_H8)
    centers_TET4 = cell_centers_bulk(coords_TET4, topo_TET4)
    k = min(k_max, len(centers_TET4))
    neighbours = k_nearest_neighbours(centers_TET4, centers_H8, k=k)
    neighbours = atleast2d(neighbours, back=True)
    cell_coords_TET4 = cells_coords(coords_TET4, topo_TET4)
    cell_coords_H8 = cells_coords(coords_H8, topo_H8)
    H8_bool = H8_in_TET4_bulk_knn(cell_coords_H8, cell_coords_TET4, neighbours, tol=tol)
    return coords_H8, topo_H8[H8_bool]
