# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit

from neumann import atleast3d, repeat

from .space import frames_of_lines
from .grid import rgridMT as grid


__cache = True


__all__ = ["mesh1d_uniform"]


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _mesh1d_uniform_(
    coords: ndarray, topo: ndarray, eshape: ndarray, N: int, frames: ndarray
):
    origo = np.zeros(1)
    subcoords_, subtopo_ = grid((1,), N, eshape, origo, 0)
    num_node_sub = len(subcoords_)
    N_new = len(coords) + len(topo) * (N - 1)
    coords_new = np.zeros((N_new, coords.shape[1]), dtype=coords.dtype)
    coords_new[: len(coords)] = coords
    frames_new = dict()
    topo_new = dict()
    cN = len(coords)  # node counter
    for i in range(topo.shape[0]):
        subtopo = subtopo_ + cN - 1
        subtopo[0, 0] = topo[i, 0]
        subtopo[-1, -1] = topo[i, -1]
        p1 = coords[topo[i, 0]]
        p2 = coords[topo[i, -1]]
        for j in range(1, num_node_sub - 1):
            p = p1 * (1 - subcoords_[j]) + p2 * subcoords_[j]
            coords_new[cN] = p
            cN += 1
        topo_new[i] = np.copy(subtopo)
        frames_new[i] = repeat(frames[i], N)
    return coords_new, topo_new, frames_new


def mesh1d_uniform(
    coords: ndarray,
    topo: ndarray,
    eshape: ndarray,
    *args,
    N: int = 2,
    refZ=None,
    return_frames=False,
    **kwargs
):
    """
    Returns the representation of a uniform 1d mesh as a tuple of numpy arrays.
    """
    frames = atleast3d(frames_of_lines(coords, topo, refZ))
    coords, topo, frames = _mesh1d_uniform_(coords, topo, eshape, N, frames)
    if return_frames:
        return coords, topo, frames
    return coords, topo
