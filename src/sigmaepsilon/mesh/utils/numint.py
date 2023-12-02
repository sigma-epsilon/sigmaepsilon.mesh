from typing import Tuple, Iterable, Optional, Union
from numbers import Number

import numpy as np
from numpy import ndarray

from sigmaepsilon.math.numint import gauss_points as gp
from .tri import nat_to_loc_tri as n2l_tri, loc_to_nat_tri as l2n_tri
from .tet import nat_to_loc_tet as n2l_tet


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


def Gauss_Legendre_Line_Grid(n: int) -> Tuple[ndarray, ndarray]:
    return gp(n)


#  TRIANGLES

# https://mathsfromnothing.au/triangle-quadrature-rules/?i=1


def _complete_natural_coordinates(nat: ndarray) -> ndarray:
    res = np.zeros((len(nat), 3), dtype=nat.dtype)
    for i in range(len(res)):
        res[i, 2] = 1 - res[i, 0] - res[i, 1]
    return res


def Gauss_Legendre_Tri_1(
    center: Optional[Union[ndarray, None]] = None, natural: Optional[bool] = False
) -> Tuple[ndarray, ndarray]:
    p, w = np.array([[0.0, 0.0]]), np.array([1 / 2])
    if isinstance(center, ndarray):
        p += center
    if natural:
        p = np.array([l2n_tri(x, center=center) for x in p], dtype=float)
    return p, w


def Gauss_Legendre_Tri_3a(
    center: Optional[Union[ndarray, None]] = None, natural: Optional[bool] = False
) -> Tuple[ndarray, ndarray]:
    nat = np.array(
        [
            [2 / 3, 1 / 6, 1 / 6],
            [1 / 6, 2 / 3, 1 / 6],
            [1 / 6, 1 / 6, 2 / 3],
        ],
        dtype=float,
    )
    if not natural:
        p = np.array([n2l_tri(n, center=center) for n in nat], dtype=float)
    else:
        p = nat
    w = np.array([1 / 6, 1 / 6, 1 / 6])
    return p, w


def Gauss_Legendre_Tri_3b(
    center: Optional[Union[ndarray, None]] = None, natural: Optional[bool] = False
) -> Tuple[ndarray, ndarray]:
    nat = np.array(
        [
            [0.0, 1 / 2, 1 / 2],
            [1 / 2, 0.0, 1 / 2],
            [1 / 2, 1 / 2, 0.0],
        ],
        dtype=float,
    )
    if not natural:
        p = np.array([n2l_tri(n, center=center) for n in nat], dtype=float)
    else:
        p = nat
    w = np.array([1 / 6, 1 / 6, 1 / 6])
    return p, w


def Gauss_Legendre_Tri_4(
    center: Optional[Union[ndarray, None]] = None, natural: Optional[bool] = False
) -> Tuple[ndarray, ndarray]:
    nat = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.6, 0.2, 0.2],
        ],
        dtype=float,
    )
    if not natural:
        p = np.array([n2l_tri(n, center=center) for n in nat], dtype=float)
    else:
        p = nat
    w = np.array([-0.5625, 0.520833333333333, 0.520833333333333, 0.520833333333333]) / 2
    return p, w


def Gauss_Legendre_Tri_6(
    center: Optional[Union[ndarray, None]] = None, natural: Optional[bool] = False
) -> Tuple[ndarray, ndarray]:
    nat = np.array(
        [
            [0.445948490915965, 0.108103018168070],
            [0.445948490915965, 0.445948490915965],
            [0.108103018168070, 0.445948490915965],
            [0.091576213509771, 0.816847572980459],
            [0.091576213509771, 0.091576213509771],
            [0.816847572980459, 0.091576213509771],
        ],
        dtype=float,
    )
    nat = _complete_natural_coordinates(nat)
    if not natural:
        p = np.array([n2l_tri(n, center=center) for n in nat], dtype=float)
    else:
        p = nat
    w = (
        np.array(
            [
                0.223381589678011,
                0.223381589678011,
                0.223381589678011,
                0.109951743655322,
                0.109951743655322,
                0.109951743655322,
            ]
        )
        / 2
    )
    return p, w


#  QUADRILATERALS


def Gauss_Legendre_Quad_Grid(i: int, j: int = None) -> Tuple[ndarray, ndarray]:
    j = i if j is None else j
    return gp(i, j)


def Gauss_Legendre_Quad_1() -> Tuple[ndarray, ndarray]:
    return gp(1, 1)


def Gauss_Legendre_Quad_4() -> Tuple[ndarray, ndarray]:
    return gp(2, 2)


def Gauss_Legendre_Quad_9() -> Tuple[ndarray, ndarray]:
    return gp(3, 3)


#  TETRAHEDRA


def Gauss_Legendre_Tet_1(
    center: Optional[Union[ndarray, None]] = None, natural: Optional[bool] = False
) -> Tuple[ndarray, ndarray]:
    nat = np.array([[0.25, 0.25, 0.25, 0.25]])
    if not natural:
        p = np.array([n2l_tet(n, center=center) for n in nat], dtype=float)
    else:
        p = nat
    w = np.array([1 / 6])
    return p, w


def Gauss_Legendre_Tet_4(
    center: Optional[Union[ndarray, None]] = None, natural: Optional[bool] = False
) -> Tuple[ndarray, ndarray]:
    nat = np.array(
        [
            [
                0.585410196624968,
                0.138196601125010,
                0.138196601125010,
                0.138196601125010,
            ],
            [
                0.138196601125010,
                0.585410196624968,
                0.138196601125010,
                0.138196601125010,
            ],
            [
                0.138196601125010,
                0.138196601125010,
                0.585410196624968,
                0.138196601125010,
            ],
            [
                0.138196601125010,
                0.138196601125010,
                0.138196601125010,
                0.585410196624968,
            ],
        ]
    )
    if not natural:
        p = np.array([n2l_tet(n, center=center) for n in nat], dtype=float)
    else:
        p = nat
    w = np.full(4, 1 / 24)
    return p, w


def Gauss_Legendre_Tet_5(
    center: Optional[Union[ndarray, None]] = None, natural: Optional[bool] = False
) -> Tuple[ndarray, ndarray]:
    nat = np.array(
        [
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
            [1 / 2, 1 / 6, 1 / 6, 1 / 6],
            [1 / 6, 1 / 2, 1 / 6, 1 / 6],
            [1 / 6, 1 / 6, 1 / 2, 1 / 6],
            [1 / 6, 1 / 6, 1 / 6, 1 / 2],
        ]
    )
    if not natural:
        p = np.array([n2l_tet(n, center=center) for n in nat], dtype=float)
    else:
        p = nat
    w = np.array([-4 / 30, 9 / 120, 9 / 120, 9 / 120, 9 / 120])
    return p, w


def Gauss_Legendre_Tet_11(
    center: Optional[Union[ndarray, None]] = None, natural: Optional[bool] = False
) -> Tuple[ndarray, ndarray]:
    nat = np.array(
        [
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
            [
                0.785714285714286,
                0.0714285714285714,
                0.0714285714285714,
                0.0714285714285714,
            ],
            [
                0.0714285714285714,
                0.785714285714286,
                0.0714285714285714,
                0.0714285714285714,
            ],
            [
                0.0714285714285714,
                0.0714285714285714,
                0.785714285714286,
                0.0714285714285714,
            ],
            [
                0.0714285714285714,
                0.0714285714285714,
                0.0714285714285714,
                0.785714285714286,
            ],
            [
                0.399403576166799,
                0.399403576166799,
                0.100596423833201,
                0.100596423833201,
            ],
            [
                0.399403576166799,
                0.100596423833201,
                0.399403576166799,
                0.100596423833201,
            ],
            [
                0.399403576166799,
                0.100596423833201,
                0.100596423833201,
                0.399403576166799,
            ],
            [
                0.100596423833201,
                0.399403576166799,
                0.399403576166799,
                0.100596423833201,
            ],
            [
                0.100596423833201,
                0.399403576166799,
                0.100596423833201,
                0.399403576166799,
            ],
            [
                0.100596423833201,
                0.100596423833201,
                0.399403576166799,
                0.399403576166799,
            ],
        ]
    )
    if not natural:
        p = np.array([n2l_tet(n, center=center) for n in nat], dtype=float)
    else:
        p = nat
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


def Gauss_Legendre_Hex_Grid(
    i: int, j: int = None, k: int = None
) -> Tuple[ndarray, ndarray]:
    j = i if j is None else j
    k = j if k is None else k
    return gp(i, j, k)


# WEDGES


def Gauss_Legendre_Wedge_3x2() -> Tuple[ndarray, ndarray]:
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


def Gauss_Legendre_Wedge_3x3() -> Tuple[ndarray, ndarray]:
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
