# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def extrude_T3_TET4(points: ndarray, triangles: ndarray, h: float = 1.0, zres: int = 1):
    nT = triangles.shape[0]
    nP = points.shape[0]
    nE = nT * zres * 3
    nC = nP * (zres + 1)
    coords = np.zeros((nC, 3), dtype=points.dtype)
    topo = np.zeros((nE, 4), dtype=triangles.dtype)
    coords[:nP, :2] = points[:, :2]
    for i in prange(zres):
        coords[nP * (i + 1) : nP * (i + 2), :2] = points[:, :2]
        coords[nP * (i + 1) : nP * (i + 2), 2] = h * (i + 1) / zres
        for j in prange(nT):
            id = i * nT * 3 + j * 3
            i_0, j_0, k_0 = triangles[j] + i * nP
            i_1, j_1, k_1 = triangles[j] + (i + 1) * nP
            #
            topo[id, 0] = i_0
            topo[id, 1] = j_0
            topo[id, 2] = k_0
            topo[id, 3] = k_1
            #
            topo[id + 1, 0] = i_0
            topo[id + 1, 1] = j_0
            topo[id + 1, 2] = k_1
            topo[id + 1, 3] = j_1
            #
            topo[id + 2, 0] = i_0
            topo[id + 2, 1] = j_1
            topo[id + 2, 2] = k_1
            topo[id + 2, 3] = i_1
    return coords, topo


@njit(nogil=True, parallel=True, cache=__cache)
def extrude_T3_W6(points: ndarray, triangles: ndarray, h: float = 1.0, zres: int = 1):
    nT = triangles.shape[0]
    nP = points.shape[0]
    nE = nT * zres
    nC = nP * (zres + 1)
    coords = np.zeros((nC, 3), dtype=points.dtype)
    topo = np.zeros((nE, 6), dtype=triangles.dtype)
    coords[:nP, :2] = points[:, :2]
    for i in prange(zres):
        coords[nP * (i + 1) : nP * (i + 2), :2] = points[:, :2]
        coords[nP * (i + 1) : nP * (i + 2), 2] = h * (i + 1) / zres
        for j in prange(nT):
            id = i * nT + j
            i_0, j_0, k_0 = triangles[j] + i * nP
            i_1, j_1, k_1 = triangles[j] + (i + 1) * nP
            #
            topo[id, 0] = i_0
            topo[id, 1] = j_0
            topo[id, 2] = k_0
            topo[id, 3] = i_1
            topo[id, 4] = j_1
            topo[id, 5] = k_1
    return coords, topo
