# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

from neumann import tile

from .utils.tri import edges_tri
from .utils.topology import unique_topo_data

__cache = True


def populate_trimesh_T3(
    points: np.ndarray,
    topo: np.ndarray,
    dp: np.ndarray,
    N=1,
    return_lines=False,
    edges=None,
    edgeIDs=None,
):
    points_pop, topo_pop = populate_trimesh_T3_njit(points, topo, dp, N)
    if return_lines:
        if edges is None or edgeIDs is None:
            edges, edgeIDs = unique_topo_data(edges_tri(topo))
        nEdge = len(edges)
        nP = len(points)
        edges_pop = tile(edges, nP, N)
        edgeIDs_pop = tile(edgeIDs, nEdge, N)
        return points_pop, edges_pop, topo_pop, edgeIDs_pop
    else:
        return points_pop, topo_pop


@njit(nogil=True, cache=__cache)
def populate_trimesh_T3_njit(points: np.ndarray, topo: np.ndarray, dp: np.ndarray, N=1):
    points_pop = tile(points, dp, N)
    topo_pop = tile(topo, len(points), N)
    return points_pop, topo_pop
