# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.utils.topology.tr import H8_to_TET4
from sigmaepsilon.mesh.utils.topology.cic import (
    T3_in_T3,
    TET4_in_TET4,
    TET4_in_H8,
    H8_in_TET4,
    TET4_in_T3,
    H8_in_T3,
)
from sigmaepsilon.mesh import PolyData, PointData
from sigmaepsilon.mesh.cells import TET4


class TestCellInCell(unittest.TestCase):
    def test_T3_in_T3(self):
        topoA = np.array([[0, 1, 2]])
        topoB = np.array([[3, 4, 5]])

        coords = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
        coords = np.array(coords).astype(float)
        self.assertTrue(T3_in_T3(coords, topoA, coords, topoB)[0])
        self.assertTrue(T3_in_T3(coords, topoB, coords, topoA)[0])

        coords = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.5, 0, 0],
            [1.5, 0, 0],
            [0.5, 1, 0],
        ]
        coords = np.array(coords).astype(float)
        self.assertTrue(T3_in_T3(coords, topoA, coords, topoB)[0])
        self.assertTrue(T3_in_T3(coords, topoB, coords, topoA)[0])

        coords = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [2, 0, 0],
            [1, 1, 0],
        ]
        coords = np.array(coords).astype(float)
        self.assertTrue(T3_in_T3(coords, topoA, coords, topoB)[0])
        self.assertTrue(T3_in_T3(coords, topoB, coords, topoA)[0])

        coords = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1.1, 0, 0],
            [2, 0, 0],
            [1, 1, 0],
        ]
        coords = np.array(coords).astype(float)
        self.assertFalse(T3_in_T3(coords, topoA, coords, topoB)[0])
        self.assertFalse(T3_in_T3(coords, topoB, coords, topoA)[0])

    def test_T3_in_T3_ValueError(self):
        topo_OK = np.arange(3)
        topo_FAIL = np.arange(4)
        coords_OK = np.ones((3, 3))
        coords_FAIL = np.ones((3, 2))
        with self.assertRaises(ValueError):
            T3_in_T3(coords_FAIL, topo_OK, coords_OK, topo_OK)
        with self.assertRaises(ValueError):
            T3_in_T3(coords_OK, topo_OK, coords_FAIL, topo_OK)
        with self.assertRaises(ValueError):
            T3_in_T3(coords_OK, topo_OK, coords_OK, topo_FAIL)
        with self.assertRaises(ValueError):
            T3_in_T3(coords_OK, topo_FAIL, coords_OK, topo_OK)

    def test_TET4_in_TET4(self):
        size = 80, 60, 40
        shape = 2, 2, 2
        coordsA, topoA = grid(size=size, shape=shape, eshape="H8")
        coordsB, topoB = grid(size=size, shape=shape, eshape="H8")
        coordsA, topoA = H8_to_TET4(coordsA, topoA)
        coordsB, topoB = H8_to_TET4(coordsB, topoB)
        coordsB[:, 0] += size[0] * 2 / 3
        coordsB[:, 1] += size[1] * 2 / 3
        coordsB[:, 2] += size[2] * 2 / 3

        A_in_B = TET4_in_TET4(coordsA, topoA, coordsB, topoB)
        where_A_in_B = np.argwhere(A_in_B).flatten()

        pd = PointData(coordsA)
        cd = TET4(topoA)
        meshA = PolyData(pd, cd)

        pd = PointData(coordsB)
        cd = TET4(topoB)
        meshB = PolyData(pd, cd)

        pd = PointData(coordsA)
        cd = TET4(topoA[where_A_in_B])
        meshC = PolyData(pd, cd)

        mesh = PolyData()
        mesh["meshA"] = meshA
        mesh["meshB"] = meshB
        mesh["meshC"] = meshC

        volume_ratio = mesh["meshA"].volume() / mesh["meshC"].volume()

        self.assertTrue(np.isclose(volume_ratio, 8.0, atol=1e-3))

    def test_TET4_in_TET4_ValueError(self):
        topo_OK = np.arange(4)
        topo_FAIL = np.arange(5)
        coords_OK = np.ones((3, 3))
        coords_FAIL = np.ones((3, 2))
        with self.assertRaises(ValueError):
            TET4_in_TET4(coords_FAIL, topo_OK, coords_OK, topo_OK)
        with self.assertRaises(ValueError):
            TET4_in_TET4(coords_OK, topo_OK, coords_FAIL, topo_OK)
        with self.assertRaises(ValueError):
            TET4_in_TET4(coords_OK, topo_OK, coords_OK, topo_FAIL)
        with self.assertRaises(ValueError):
            TET4_in_TET4(coords_OK, topo_FAIL, coords_OK, topo_OK)

    def test_H8_in_TET4(self):
        coords_H8 = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        )
        topo_H8 = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        coords_TET4, topo_TET4 = H8_to_TET4(coords_H8, topo_H8)
        result = H8_in_TET4(coords_H8, topo_H8, coords_TET4, topo_TET4)
        self.assertTrue(np.all(result))

    def test_H8_in_TET4_ValueError(self):
        topo_H8_OK = np.arange(8)
        topo_H8_FAIL = np.arange(6)
        topo_TET4_OK = np.arange(4)
        topo_TET4_FAIL = np.arange(8)
        coords_OK = np.ones((3, 3))
        coords_FAIL = np.ones((3, 2))
        with self.assertRaises(ValueError):
            H8_in_TET4(coords_FAIL, topo_H8_OK, coords_OK, topo_TET4_OK)
        with self.assertRaises(ValueError):
            H8_in_TET4(coords_OK, topo_H8_OK, coords_FAIL, topo_TET4_OK)
        with self.assertRaises(ValueError):
            H8_in_TET4(coords_OK, topo_H8_OK, coords_OK, topo_TET4_FAIL)
        with self.assertRaises(ValueError):
            H8_in_TET4(coords_OK, topo_H8_FAIL, coords_OK, topo_TET4_OK)

    def test_TET4_in_H8(self):
        coords_H8 = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        )
        topo_H8 = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        coords_TET4, topo_TET4 = H8_to_TET4(coords_H8, topo_H8)
        result = TET4_in_H8(coords_TET4, topo_TET4, coords_H8, topo_H8)
        self.assertTrue(np.all(result))

    def test_TET4_in_H8_ValueError(self):
        topo_H8_OK = np.arange(8)
        topo_H8_FAIL = np.arange(6)
        topo_TET4_OK = np.arange(4)
        topo_TET4_FAIL = np.arange(8)
        coords_OK = np.ones((3, 3))
        coords_FAIL = np.ones((3, 2))
        with self.assertRaises(ValueError):
            TET4_in_H8(coords_FAIL, topo_TET4_OK, coords_OK, topo_H8_OK)
        with self.assertRaises(ValueError):
            TET4_in_H8(coords_OK, topo_TET4_OK, coords_FAIL, topo_H8_OK)
        with self.assertRaises(ValueError):
            TET4_in_H8(coords_OK, topo_TET4_OK, coords_OK, topo_H8_FAIL)
        with self.assertRaises(ValueError):
            TET4_in_H8(coords_OK, topo_TET4_FAIL, coords_OK, topo_H8_OK)

    def test_TET4_in_T3(self):
        coords_H8 = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        )
        topo_H8 = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        coords_TET4, topo_TET4 = H8_to_TET4(coords_H8, topo_H8)
        topo_T3 = np.array(
            [
                [0, 1, 2],
                [0, 1, 7],
                [4, 5, 6],
                [4, 5, 7],
                [0, 3, 4],
                [0, 3, 7],
                [1, 2, 5],
                [1, 2, 6],
                [2, 3, 6],
                [3, 4, 7],
                [0, 1, 5],
                [0, 4, 5],
            ]
        ).astype(int)
        result = TET4_in_T3(coords_TET4, topo_TET4, coords_H8, topo_T3)
        self.assertTrue(np.all(result))

    def test_TET4_in_T3_ValueError(self):
        topo_T3_OK = np.arange(3)
        topo_T3_FAIL = np.arange(6)
        topo_TET4_OK = np.arange(4)
        topo_TET4_FAIL = np.arange(8)
        coords_OK = np.ones((3, 3))
        coords_FAIL = np.ones((3, 2))
        with self.assertRaises(ValueError):
            TET4_in_T3(coords_FAIL, topo_TET4_OK, coords_OK, topo_T3_OK)
        with self.assertRaises(ValueError):
            TET4_in_T3(coords_OK, topo_TET4_OK, coords_FAIL, topo_T3_OK)
        with self.assertRaises(ValueError):
            TET4_in_T3(coords_OK, topo_TET4_OK, coords_OK, topo_T3_FAIL)
        with self.assertRaises(ValueError):
            TET4_in_T3(coords_OK, topo_TET4_FAIL, coords_OK, topo_T3_OK)

    def test_H8_in_T3(self):
        coords_H8 = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        )
        topo_H8 = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        topo_T3 = np.array(
            [
                [0, 1, 2],
                [0, 1, 7],
                [4, 5, 6],
                [4, 5, 7],
                [0, 3, 4],
                [0, 3, 7],
                [1, 2, 5],
                [1, 2, 6],
                [2, 3, 6],
                [3, 4, 7],
                [0, 1, 5],
                [0, 4, 5],
            ]
        ).astype(int)
        result = H8_in_T3(coords_H8, topo_H8, coords_H8, topo_T3)
        self.assertTrue(np.all(result))

    def test_H8_in_T3_ValueError(self):
        topo_T3_OK = np.arange(3)
        topo_T3_FAIL = np.arange(6)
        topo_H8_OK = np.arange(8)
        topo_H8_FAIL = np.arange(2)
        coords_OK = np.ones((3, 3))
        coords_FAIL = np.ones((3, 2))
        with self.assertRaises(ValueError):
            H8_in_T3(coords_FAIL, topo_H8_OK, coords_OK, topo_T3_OK)
        with self.assertRaises(ValueError):
            H8_in_T3(coords_OK, topo_H8_OK, coords_FAIL, topo_T3_OK)
        with self.assertRaises(ValueError):
            H8_in_T3(coords_OK, topo_H8_OK, coords_OK, topo_T3_FAIL)
        with self.assertRaises(ValueError):
            H8_in_T3(coords_OK, topo_H8_FAIL, coords_OK, topo_T3_OK)


if __name__ == "__main__":
    unittest.main()
