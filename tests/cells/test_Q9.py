# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame
from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.cells import Q9


class TestQ9(SigmaEpsilonTestCase):
    def test_Q9(self):
        Lx, Ly = 800, 600
        nx, ny = 8, 6
        xbins = np.linspace(0, Lx, nx + 1)
        ybins = np.linspace(0, Ly, ny + 1)
        bins = xbins, ybins
        coords, topo = grid(bins=bins, eshape="Q9")
        frame = CartesianFrame(dim=3)

        pd = PointData(coords=coords)
        cd = Q9(topo=topo, frames=frame)
        _ = PolyData(pd, cd)

        self.assertTrue(np.isclose(cd.area(), Lx * Ly))
        self.assertTrue(np.isclose(cd.volume(), Lx * Ly))
        self.assertEqual(cd.jacobian_matrix().shape, (topo.shape[0], 9, 2, 2))

    def test_Q9_to_triangles(self):
        Lx, Ly = 800, 600
        nx, ny = 8, 6
        xbins = np.linspace(0, Lx, nx + 1)
        ybins = np.linspace(0, Ly, ny + 1)
        bins = xbins, ybins
        _, topo = grid(bins=bins, eshape="Q9")
        frame = CartesianFrame(dim=3)

        cd = Q9(topo=topo, frames=frame)
        triangles = cd.to_triangles()
        self.assertEqual(triangles.shape[1], 3)

    def test_Q9_Geometry(self):
        Q9.Geometry.polybase()
        Q9.Geometry.master_coordinates()
        Q9.Geometry.master_center()
        trimap = Q9.Geometry.trimap()
        self.assertTrue(trimap.shape[1] == 3)


if __name__ == "__main__":
    unittest.main()
