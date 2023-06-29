# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray as array
from numba import njit, prange

from neumann import minmax

from .utils.topology import detach_mesh_bulk as detach_mesh, remap_topo

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def tile2d(coords: array, topo: array, shape: tuple, tol: float = 1e-8):
    """
    Tiles a rectangle with the mesh defined by the coordinate
    array 'coords' and topology array 'topo'.

    Parameters
    ----------
    coords: ndarray

    topo: ndaray

    shape : Tupel

    tol : float

    """
    nX, nY = shape
    nP, nD = coords.shape
    nE, nNE = topo.shape

    # identify points on the boundaries
    xmin, xmax = minmax(coords[:, 0])
    ymin, ymax = minmax(coords[:, 1])
    dx = xmax - xmin
    dy = ymax - ymin
    inds_xmin = np.where(np.abs(coords[:, 0] - xmin) < tol)[0]
    inds_xmax = np.where(np.abs(coords[:, 0] - xmax) < tol)[0]
    inds_ymin = np.where(np.abs(coords[:, 1] - ymin) < tol)[0]
    inds_ymax = np.where(np.abs(coords[:, 1] - ymax) < tol)[0]

    # allocate space for the results
    nP_new = nX * nY * nP
    nE_new = nX * nY * nE
    coords_new = np.zeros((nP_new, nD), dtype=coords.dtype)
    topo_new = np.zeros((nE_new, nNE), dtype=topo.dtype)

    # tile mesh and prepare mapping to renumber coincident nodes
    imap = np.arange(nP_new, dtype=topo.dtype)
    for i in prange(nX):
        for j in prange(nY):
            blockId = i * nY + j
            i_start = blockId * nE
            i_stop = i_start + nE
            topo_new[i_start:i_stop, :] = topo + blockId * nP
            i_start = blockId * nP
            i_stop = i_start + nP
            coords_new[i_start:i_stop, 0] = coords[:, 0] + i * dx
            coords_new[i_start:i_stop, 1] = coords[:, 1] + j * dy
            coords_new[i_start:i_stop, 2] = coords[:, 2]
            # mapping
            if i > 0:
                inds_old = inds_xmin + blockId * nP
                inds_new = inds_xmax + ((i - 1) * nY + j) * nP
                imap[inds_old] = inds_new
            if j > 0:
                inds_old = inds_ymin + blockId * nP
                inds_new = inds_ymax + (i * nY + (j - 1)) * nP
                imap[inds_old] = inds_new

    # renumber and return detached mesh
    return detach_mesh(coords_new, remap_topo(topo_new, imap))
