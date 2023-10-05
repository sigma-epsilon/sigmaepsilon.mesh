# -*- coding: utf-8 -*-
import unittest, doctest

from sigmaepsilon.core.testing import SigmaEpsilonTestCase

import sigmaepsilon.mesh.tetrahedralize
from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh import tetrahedralize, PointData, PolyData
from sigmaepsilon.mesh.cells import H8


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(sigmaepsilon.mesh.tetrahedralize))
    return tests


class TestTetrahedralize(SigmaEpsilonTestCase):
    def test_tetrahedralize(self):
        coords, topo = grid(size=(80, 60, 20), shape=(8, 6, 2), eshape="H8")
        pd = PointData(coords=coords)
        cd = H8(topo=topo)
        mesh = PolyData(pd, cd)
        tetrahedralize(mesh)
        tetrahedralize(mesh, order=2)
        self.assertFailsProperly(ValueError, tetrahedralize, mesh, order=3)


if __name__ == "__main__":
    unittest.main()
