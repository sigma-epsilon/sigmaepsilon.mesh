# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.utils.topology.tr import H8_to_TET4
from sigmaepsilon.mesh.utils.logical import T3_in_T3, TET4_in_TET4
from sigmaepsilon.mesh import PolyData, PointData
from sigmaepsilon.mesh.cells import TET4


class TestUtilsLogical(unittest.TestCase):
    def test_T3_in_T3_inplane(self):
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


if __name__ == "__main__":
    unittest.main()
