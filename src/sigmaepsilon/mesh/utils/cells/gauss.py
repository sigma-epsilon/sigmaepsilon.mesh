import numpy as np

from neumann.numint import gauss_points as gp


# LINES


def Gauss_Legendre_Line_Grid(n: int):
    return gp(n)


#  TRIANGLES


def Gauss_Legendre_Tri_1():
    return np.array([[1 / 3, 1 / 3]]), np.array([1 / 2])


def Gauss_Legendre_Tri_3a():
    p = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
    w = np.array([1 / 6, 1 / 6, 1 / 6])
    return p, w


def Gauss_Legendre_Tri_3b():
    p = np.array([[1 / 2, 1 / 2], [0, 1 / 2], [1 / 2, 0]])
    w = np.array([1 / 6, 1 / 6, 1 / 6])
    return p, w


#  QUADRILATERALS


def Gauss_Legendre_Quad_Grid(n: int, m: int = None):
    m = n if m is None else m
    return gp(n, m)


def Gauss_Legendre_Quad_1():
    return gp(1, 1)


def Gauss_Legendre_Quad_4():
    return gp(2, 2)


def Gauss_Legendre_Quad_9():
    return gp(3, 3)


#  TETRAHEDRA


def Gauss_Legendre_Tet_1():
    p = np.array([[1 / 4, 1 / 4, 1 / 4]])
    w = np.array([1 / 6])
    return p, w


def Gauss_Legendre_Tet_4():
    a = (5 + 3 * np.sqrt(5)) / 20
    b = (5 - np.sqrt(5)) / 20
    p = np.array([[a, b, b], [b, a, b], [b, b, a], [b, b, b]])
    w = np.full(4, 1 / 24)
    return p, w


def Gauss_Legendre_Tet_5():
    p = np.array(
        [
            [1 / 4, 1 / 4, 1 / 4],
            [1 / 2, 1 / 6, 1 / 6],
            [1 / 6, 1 / 2, 1 / 6],
            [1 / 6, 1 / 6, 1 / 2],
            [1 / 6, 1 / 6, 1 / 6],
        ]
    )
    w = np.array([-4 / 30, 9 / 120, 9 / 120, 9 / 120, 9 / 120])
    return p, w


def Gauss_Legendre_Tet_11():
    a = (1 + 3 * np.sqrt(5 / 15)) / 4
    b = (1 - np.sqrt(5 / 14)) / 4
    p = np.array(
        [
            [1 / 4, 1 / 4, 1 / 4],
            [11 / 14, 1 / 14, 1 / 14],
            [1 / 14, 11 / 14, 1 / 14],
            [1 / 14, 1 / 14, 11 / 14],
            [1 / 14, 1 / 14, 1 / 14],
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


def Gauss_Legendre_Hex_Grid(n: int, m: int = None, k: int = None):
    m = n if m is None else m
    k = m if k is None else k
    return gp(n, m, k)


# WEDGES


def Gauss_Legendre_Wedge_3x2():
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


def Gauss_Legendre_Wedge_3x3():
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
