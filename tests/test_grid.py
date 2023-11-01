# -*- coding: utf-8 -*-
import numpy as np
import unittest, doctest

import sigmaepsilon.mesh.grid
from sigmaepsilon.mesh.grid import grid, knngridL2


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(sigmaepsilon.mesh.grid))
    return tests


class TestGrid(unittest.TestCase):
    def test_grid_Q4(self):
        size = 80, 60
        shape = 8, 6
        topo = grid(size=size, shape=shape, eshape="Q4")[1]

        self.assertEqual(topo.shape[0], np.prod(shape))
        self.assertEqual(topo.shape[1], 4)

    def test_grid_Q8(self):
        size = 80, 60
        shape = 8, 6
        topo = grid(size=size, shape=shape, eshape="Q8")[1]

        self.assertEqual(topo.shape[0], np.prod(shape))
        self.assertEqual(topo.shape[1], 8)

    def test_grid_Q9(self):
        size = 80, 60
        shape = 8, 6
        topo = grid(size=size, shape=shape, eshape="Q9")[1]

        self.assertEqual(topo.shape[0], np.prod(shape))
        self.assertEqual(topo.shape[1], 9)

    def test_grid_H8(self):
        size = 80, 60, 20
        shape = 8, 6, 2
        coords, topo = grid(size=size, shape=shape, eshape="H8")

        self.assertEqual(topo.shape[0], np.prod(shape))
        self.assertEqual(topo.shape[1], 8)
        self.assertEqual(coords.shape[0], np.prod([x + 1 for x in shape]))
        self.assertEqual(coords.shape[1], 3)

    def test_grid_H27(self):
        size = 80, 60, 20
        shape = 8, 6, 2
        topo = grid(size=size, shape=shape, eshape="H27")[1]

        self.assertEqual(topo.shape[0], np.prod(shape))
        self.assertEqual(topo.shape[1], 27)

    def test_knngridL2(self):
        size = 80, 60, 20
        shape = 8, 6, 2
        coords, topo = knngridL2(size=size, shape=shape, eshape="H8")
        coords_, topo_ = knngridL2(X=coords)

        self.assertEqual(coords.shape, coords_.shape)
        self.assertEqual(topo.shape, topo_.shape)


if __name__ == "__main__":
    unittest.main()
