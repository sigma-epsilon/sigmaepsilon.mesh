import unittest

from polymesh.triang import triangulate
from polymesh.grid import grid
from polymesh.extrude import extrude_T3_W6
from polymesh.utils.topology import (
    T3_to_T6,
    T6_to_T3,
    Q9_to_Q4,
    Q4_to_T3,
    Q4_to_Q9,
    H8_to_H27,
    Q4_to_Q8,
    Q9_to_T6,
    H8_to_TET4,
    TET4_to_TET10,
    H8_to_L2,
    TET4_to_L2,
    L2_to_L3,
    Q4_to_Q8,
    Q8_to_T3,
    H27_to_H8,
    W6_to_TET4,
    W18_to_W6,
    W6_to_W18,
)


class TestTopoTR(unittest.TestCase):
    def test_1(self):
        def test_1(Lx, Ly, nx, ny):
            """T3 -> T6 -> T3"""
            coords, topo, _ = triangulate(size=(Lx, Ly), shape=(nx, ny))
            nE1 = topo.shape[0]
            coords, topo = T3_to_T6(coords, topo)
            coords, topo = T6_to_T3(coords, topo)
            nE2 = topo.shape[0]
            return nE1 * 4 == nE2

        assert test_1(1, 1, 2, 2)

    def test_2(self):
        def test_2(Lx, Ly, nx, ny):
            """Q9 -> Q4 -> Q9 -> T6"""
            coords, topo = grid(size=(Lx, Ly), shape=(nx, ny), eshape="Q9")
            nE1 = topo.shape[0]
            coords, topo = Q9_to_Q4(coords, topo)
            coords, topo = Q4_to_Q9(coords, topo)
            coords, topo = Q9_to_T6(coords, topo)
            nE2 = topo.shape[0]
            return nE1 * 8 == nE2

        assert test_2(1, 1, 2, 2)

    def test_3(self):
        def test_3(Lx, Ly, nx, ny):
            """Q9 -> Q4 -> T3 -> T6"""
            coords, topo = grid(size=(Lx, Ly), shape=(nx, ny), eshape="Q9")
            nE1 = topo.shape[0]
            coords, topo = Q9_to_Q4(coords, topo)
            Q8_to_T3(*Q4_to_Q8(coords, topo))
            coords, topo = Q4_to_T3(coords, topo)
            coords, topo = T3_to_T6(coords, topo)
            nE2 = topo.shape[0]
            return nE1 * 8 == nE2

        assert test_3(1, 1, 2, 2)

    def test_4(self):
        def test_4(Lx, Ly, nx, ny):
            """Q9 -> Q4 -> Q8"""
            coords, topo = grid(size=(Lx, Ly), shape=(nx, ny), eshape="Q9")
            nE1 = topo.shape[0]
            coords, topo = Q9_to_Q4(coords, topo)
            coords, topo = Q4_to_Q8(coords, topo)
            nE2 = topo.shape[0]
            return nE1 * 4 == nE2

        assert test_4(1, 1, 2, 2)

    def test_5(self):
        def test_5(Lx, Ly, Lz, nx, ny, nz):
            """H8 -> H27"""
            coords, topo = grid(size=(Lx, Ly, Lz), shape=(nx, ny, nz), eshape="H8")
            nE1 = topo.shape[0]
            coords, topo = H8_to_H27(coords, topo)
            nE2 = topo.shape[0]
            return nE1 == nE2

        assert test_5(1, 1, 1, 2, 2, 2)

    def test_6(self):
        def test_6(Lx, Ly, Lz, nx, ny, nz):
            """H8 -> TET4 -> TET10"""
            coords, topo = grid(size=(Lx, Ly, Lz), shape=(nx, ny, nz), eshape="H8")
            nE1 = topo.shape[0]
            coords, topo = H8_to_TET4(coords, topo)
            coords, topo = TET4_to_TET10(coords, topo)
            nE2 = topo.shape[0]
            return nE1 * 5 == nE2

        assert test_6(1, 1, 1, 2, 2, 2)

    def test_7(self):
        def test_7(Lx, Ly, Lz, nx, ny, nz):
            """H27 -> H8 -> TET4 -> L2 -> L3"""
            coords, topo = grid(size=(Lx, Ly, Lz), shape=(nx, ny, nz), eshape="H27")
            H8_to_L2(*H27_to_H8(coords, topo))
            coords, topo = H8_to_TET4(*H27_to_H8(coords, topo))
            coords, topo = TET4_to_L2(coords, topo)
            coords, topo = L2_to_L3(coords, topo)
            return True

        assert test_7(1, 1, 1, 2, 2, 2)

    def test_8(self):
        def test_8(Lx, Ly, Lz, nx, ny, nz):
            """T3 -> W6 -> W18 -> W6 -> TET4"""
            coords, topo, _ = triangulate(size=(Lx, Ly), shape=(nx, ny))
            coords, topo = extrude_T3_W6(coords, topo, Lz, nz)
            coords, topo = W6_to_W18(coords, topo)
            coords, topo = W18_to_W6(coords, topo)
            coords, topo = W6_to_TET4(coords, topo)
            return True

        assert test_8(1, 1, 1, 2, 2, 2)


if __name__ == "__main__":
    unittest.main()
