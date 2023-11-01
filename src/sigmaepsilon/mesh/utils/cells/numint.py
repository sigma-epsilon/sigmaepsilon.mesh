from typing import Tuple, Iterable
from numbers import Number

import numpy as np
from numpy import ndarray

from sigmaepsilon.math.numint import gauss_points as gp


class Quadrature:
    
    def __init__(self, x: Iterable[Number], w: Iterable[Number]):
        self._pos = x
        self._weight = w
    
    @property
    def pos(self) -> Iterable[Number]:
        return self._pos
    
    @property
    def weight(self) -> Iterable[Number]:
        return self._weight

# LINES


def Gauss_Legendre_Line_Grid(n: int) -> Tuple[ndarray]:
    return gp(n)


#  TRIANGLES


def Gauss_Legendre_Tri_1() -> Tuple[ndarray]:
    return np.array([[0.0, 0.0]]), np.array([1 / 2])


def Gauss_Legendre_Tri_3a() -> Tuple[ndarray]:
    p = np.array([[-1 / 6, -1 / 6], [1 / 3, -1 / 6], [-1 / 6, 1 / 3]])
    w = np.array([1 / 6, 1 / 6, 1 / 6])
    return p, w


def Gauss_Legendre_Tri_3b() -> Tuple[ndarray]:
    p = np.array([[1 / 6, 1 / 6], [-1 / 3, 1 / 6], [1 / 6, -1 / 3]])
    w = np.array([1 / 6, 1 / 6, 1 / 6])
    return p, w


#  QUADRILATERALS


def Gauss_Legendre_Quad_Grid(i: int, j: int = None) -> Tuple[ndarray]:
    j = i if j is None else j
    return gp(i, j)


def Gauss_Legendre_Quad_1() -> Tuple[ndarray]:
    return gp(1, 1)


def Gauss_Legendre_Quad_4() -> Tuple[ndarray]:
    return gp(2, 2)


def Gauss_Legendre_Quad_9() -> Tuple[ndarray]:
    return gp(3, 3)


#  TETRAHEDRA


def Gauss_Legendre_Tet_1() -> Tuple[ndarray]:
    p = np.array([[-1 / 12, -1 / 12, -1 / 12]])
    w = np.array([1 / 6])
    return p, w


def Gauss_Legendre_Tet_4() -> Tuple[ndarray]:
    a = ((5 + 3 * np.sqrt(5)) / 20) - 1 / 3
    b = ((5 - np.sqrt(5)) / 20) - 1 / 3
    p = np.array([[a, b, b], [b, a, b], [b, b, a], [b, b, b]])
    w = np.full(4, 1 / 24)
    return p, w


def Gauss_Legendre_Tet_5() -> Tuple[ndarray]:
    p = np.array(
        [
            [-1 / 12, -1 / 12, -1 / 12],
            [1 / 6, -1 / 6, -1 / 6],
            [-1 / 6, 1 / 6, -1 / 6],
            [-1 / 6, -1 / 6, 1 / 6],
            [-1 / 6, -1 / 6, -1 / 6],
        ]
    )
    w = np.array([-4 / 30, 9 / 120, 9 / 120, 9 / 120, 9 / 120])
    return p, w


def Gauss_Legendre_Tet_11() -> Tuple[ndarray]:
    a = ((1 + 3 * np.sqrt(5 / 15)) / 4) - 1 / 3
    b = ((1 - np.sqrt(5 / 14)) / 4) - 1 / 3
    p = np.array(
        [
            [-1 / 12, -1 / 12, -1 / 12],
            [19 / 42, -11 / 42, -11 / 42],
            [-11 / 42, 19 / 42, -11 / 42],
            [-11 / 42, -11 / 42, 19 / 42],
            [-11 / 42, -11 / 42, -11 / 42],
            [a, a, b],
            [a, b, a],
            [a, b, b],
            [b, a, a],
            [b, a, b],
            [b, b, a],
        ]
    )
    w = np.array(
        [
            -74 / 5625,
            343 / 45000,
            343 / 45000,
            343 / 45000,
            343 / 45000,
            56 / 2250,
            56 / 2250,
            56 / 2250,
            56 / 2250,
            56 / 2250,
            56 / 2250,
        ]
    )
    return p, w


#  HEXAHEDRA


def Gauss_Legendre_Hex_Grid(i: int, j: int = None, k: int = None) -> Tuple[ndarray]:
    j = i if j is None else j
    k = j if k is None else k
    return gp(i, j, k)


# WEDGES


def Gauss_Legendre_Wedge_3x2() -> Tuple[ndarray]:
    p_tri, w_tri = Gauss_Legendre_Tri_3a()
    p_line, w_line = Gauss_Legendre_Line_Grid(2)
    p = np.zeros((6, 3), dtype=float)
    w = np.zeros((6,), dtype=float)
    p[:3, :2] = p_tri
    p[:3, 2] = p_line[0]
    w[:3] = w_tri * w_line[0]
    p[3:6, :2] = p_tri
    p[3:6, 2] = p_line[0]
    w[3:6] = w_tri * w_line[1]
    return p, w


def Gauss_Legendre_Wedge_3x3() -> Tuple[ndarray]:
    p_tri, w_tri = Gauss_Legendre_Tri_3a()
    p_line, w_line = Gauss_Legendre_Line_Grid(3)
    n = len(w_line) * len(w_tri)
    p = np.zeros((n, 3), dtype=float)
    w = np.zeros((n,), dtype=float)
    for i in range(len(w_line)):
        p[i * 3 : (i + 1) * 3, :2] = p_tri
        p[i * 3 : (i + 1) * 3, 2] = p_line[i]
        w[i * 3 : (i + 1) * 3] = w_tri * w_line[i]
    return p, w
