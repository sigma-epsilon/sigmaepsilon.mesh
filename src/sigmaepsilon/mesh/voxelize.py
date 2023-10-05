# -*- coding: utf-8 -*-
from typing import Tuple, Optional, Union

import numpy as np
from numpy import ndarray

from .data import PolyData, PointData
from .cells import H8
from .grid import grid
from .utils.topology import detach_mesh_bulk
from .space import CartesianFrame

__all__ = ["voxelize_cylinder"]


def voxelize_cylinder(
    radius: ndarray,
    height: float,
    size: float,  # voxel edge length
    frame: Optional[Union[CartesianFrame, None]] = None,
) -> Tuple[ndarray, ndarray]:
    """
    Returns raw mesh data of a voxelized cylinder.
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
