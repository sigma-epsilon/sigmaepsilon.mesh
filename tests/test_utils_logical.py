# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.mesh.utils.logical import T3_in_T3


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


if __name__ == "__main__":
    unittest.main()
