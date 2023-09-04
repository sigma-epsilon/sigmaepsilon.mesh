# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
from numpy import ndarray
from numba import njit

__cache = True


@njit(nogil=True, cache=__cache)
def edges_Q4(quads: ndarray) -> ndarray:
    nE = len(quads)
    edges = np.zeros((nE, 4, 2), dtype=quads.dtype)
    edges[:, 0, 0] = quads[:, 0]
    edges[:, 0, 1] = quads[:, 1]
    edges[:, 1, 0] = quads[:, 1]
    edges[:, 1, 1] = quads[:, 2]
    edges[:, 2, 0] = quads[:, 2]
    edges[:, 2, 1] = quads[:, 3]
    edges[:, 3, 0] = quads[:, 3]
    edges[:, 3, 1] = quads[:, 0]
    return edges


def edgeIds_T3() -> ndarray:
    return np.array([[0, 1], [1, 2], [2, 0]], dtype=int)


def edgeIds_TET4() -> ndarray:
    return np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]], dtype=int)


@njit(nogil=True, cache=__cache)
def edges_TET4(topo: ndarray) -> ndarray:
    nE = len(topo)
    edges = np.zeros((nE, 6, 2), dtype=topo.dtype)
    edges[:, 0, 0] = topo[:, 0]
    edges[:, 0, 1] = topo[:, 1]
    edges[:, 1, 0] = topo[:, 1]
    edges[:, 1, 1] = topo[:, 2]
    edges[:, 2, 0] = topo[:, 2]
    edges[:, 2, 1] = topo[:, 0]
    edges[:, 3, 0] = topo[:, 0]
    edges[:, 3, 1] = topo[:, 3]
    edges[:, 4, 0] = topo[:, 1]
    edges[:, 4, 1] = topo[:, 3]
    edges[:, 5, 0] = topo[:, 2]
    edges[:, 5, 1] = topo[:, 3]
    return edges


@njit(nogil=True, cache=__cache)
def faces_TET4(topo: ndarray) -> ndarray:
    nE = len(topo)
    faces = np.zeros((nE, 4, 3), dtype=topo.dtype)
    faces[:, 0, 0] = topo[:, 0]
    faces[:, 0, 1] = topo[:, 1]
    faces[:, 0, 2] = topo[:, 3]
    faces[:, 1, 0] = topo[:, 1]
    faces[:, 1, 1] = topo[:, 2]
    faces[:, 1, 2] = topo[:, 3]
    faces[:, 2, 0] = topo[:, 2]
    faces[:, 2, 1] = topo[:, 0]
    faces[:, 2, 2] = topo[:, 3]
    faces[:, 3, 0] = topo[:, 0]
    faces[:, 3, 1] = topo[:, 2]
    faces[:, 3, 2] = topo[:, 1]
    return faces


def edgeIds_H8() -> ndarray:
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=int,
    )


@njit(nogil=True, cache=__cache)
def edges_H8(topo: ndarray) -> ndarray:
    nE = len(topo)
    edges = np.zeros((nE, 12, 2), dtype=topo.dtype)
    edges[:, 0, 0] = topo[:, 0]
    edges[:, 0, 1] = topo[:, 1]
    edges[:, 1, 0] = topo[:, 1]
    edges[:, 1, 1] = topo[:, 2]
    edges[:, 2, 0] = topo[:, 2]
    edges[:, 2, 1] = topo[:, 3]
    edges[:, 3, 0] = topo[:, 3]
    edges[:, 3, 1] = topo[:, 0]
    edges[:, 4, 0] = topo[:, 4]
    edges[:, 4, 1] = topo[:, 5]
    edges[:, 5, 0] = topo[:, 5]
    edges[:, 5, 1] = topo[:, 6]
    edges[:, 6, 0] = topo[:, 6]
    edges[:, 6, 1] = topo[:, 7]
    edges[:, 7, 0] = topo[:, 7]
    edges[:, 7, 1] = topo[:, 4]
    edges[:, 8, 0] = topo[:, 0]
    edges[:, 8, 1] = topo[:, 4]
    edges[:, 9, 0] = topo[:, 1]
    edges[:, 9, 1] = topo[:, 5]
    edges[:, 10, 0] = topo[:, 2]
    edges[:, 10, 1] = topo[:, 6]
    edges[:, 11, 0] = topo[:, 3]
    edges[:, 11, 1] = topo[:, 7]
    return edges


@njit(nogil=True, cache=__cache)
def faces_H8(topo: ndarray) -> ndarray:
    nE = len(topo)
    faces = np.zeros((nE, 6, 4), dtype=topo.dtype)
    faces[:, 0, 0] = topo[:, 0]
    faces[:, 0, 1] = topo[:, 4]
    faces[:, 0, 2] = topo[:, 7]
    faces[:, 0, 3] = topo[:, 3]
    faces[:, 1, 0] = topo[:, 1]
    faces[:, 1, 1] = topo[:, 2]
    faces[:, 1, 2] = topo[:, 6]
    faces[:, 1, 3] = topo[:, 5]
    faces[:, 2, 0] = topo[:, 0]
    faces[:, 2, 1] = topo[:, 1]
    faces[:, 2, 2] = topo[:, 5]
    faces[:, 2, 3] = topo[:, 4]
    faces[:, 3, 0] = topo[:, 2]
    faces[:, 3, 1] = topo[:, 3]
    faces[:, 3, 2] = topo[:, 7]
    faces[:, 3, 3] = topo[:, 6]
    faces[:, 4, 0] = topo[:, 0]
    faces[:, 4, 1] = topo[:, 3]
    faces[:, 4, 2] = topo[:, 2]
    faces[:, 4, 3] = topo[:, 1]
    faces[:, 5, 0] = topo[:, 4]
    faces[:, 5, 1] = topo[:, 5]
    faces[:, 5, 2] = topo[:, 6]
    faces[:, 5, 3] = topo[:, 7]
    return faces


@njit(nogil=True, cache=__cache)
def edges_W6(topo: ndarray) -> ndarray:
    nE = len(topo)
    edges = np.zeros((nE, 9, 2), dtype=topo.dtype)
    edges[:, 0, 0] = topo[:, 0]
    edges[:, 0, 1] = topo[:, 1]
    edges[:, 1, 0] = topo[:, 1]
    edges[:, 1, 1] = topo[:, 2]
    edges[:, 2, 0] = topo[:, 2]
    edges[:, 2, 1] = topo[:, 0]
    edges[:, 3, 0] = topo[:, 3]
    edges[:, 3, 1] = topo[:, 4]
    edges[:, 4, 0] = topo[:, 4]
    edges[:, 4, 1] = topo[:, 5]
    edges[:, 5, 0] = topo[:, 5]
    edges[:, 5, 1] = topo[:, 3]
    edges[:, 6, 0] = topo[:, 0]
    edges[:, 6, 1] = topo[:, 3]
    edges[:, 7, 0] = topo[:, 1]
    edges[:, 7, 1] = topo[:, 4]
    edges[:, 8, 0] = topo[:, 2]
    edges[:, 8, 1] = topo[:, 5]
    return edges


@njit(nogil=True, cache=__cache)
def faces_W6(topo: ndarray) -> Tuple[ndarray, ndarray]:
    nE = len(topo)

    faces2 = np.zeros((nE, 2, 3), dtype=topo.dtype)
    faces2[:, 0, 0] = topo[:, 0]
    faces2[:, 0, 1] = topo[:, 2]
    faces2[:, 0, 2] = topo[:, 1]
    faces2[:, 1, 2] = topo[:, 3]
    faces2[:, 1, 0] = topo[:, 4]
    faces2[:, 1, 1] = topo[:, 5]

    faces4 = np.zeros((nE, 3, 4), dtype=topo.dtype)
    faces4[:, 0, 2] = topo[:, 0]
    faces4[:, 0, 3] = topo[:, 1]
    faces4[:, 0, 0] = topo[:, 4]
    faces4[:, 0, 1] = topo[:, 3]
    faces4[:, 1, 2] = topo[:, 1]
    faces4[:, 1, 3] = topo[:, 2]
    faces4[:, 1, 0] = topo[:, 5]
    faces4[:, 1, 1] = topo[:, 4]
    faces4[:, 2, 2] = topo[:, 2]
    faces4[:, 2, 3] = topo[:, 0]
    faces4[:, 2, 0] = topo[:, 3]
    faces4[:, 2, 1] = topo[:, 5]

    return faces4, faces2
