# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame
from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.cells import Q4


class TestQ4(SigmaEpsilonTestCase):
    def test_Q4(self):
        Lx, Ly = 800, 600
        nx, ny = 8, 6
        xbins = np.linspace(0, Lx, nx + 1)
        ybins = np.linspace(0, Ly, ny + 1)
        bins = xbins, ybins
        coords, topo = grid(bins=bins, eshape="Q4")
        frame = CartesianFrame(dim=3)

        pd = PointData(coords=coords)
        cd = Q4(topo=topo, frames=frame)
        _ = PolyData(pd, cd)

        self.assertTrue(np.isclose(cd.area(), Lx * Ly))
        self.assertTrue(np.isclose(cd.volume(), Lx * Ly))
        self.assertEqual(cd.jacobian_matrix().shape, (topo.shape[0], 4, 2, 2))

    def test_Q4_to_triangles(self):
        Lx, Ly = 800, 600
        nx, ny = 8, 6
        xbins = np.linspace(0, Lx, nx + 1)
        ybins = np.linspace(0, Ly, ny + 1)
        bins = xbins, ybins
        _, topo = grid(bins=bins, eshape="Q4")
        frame = CartesianFrame(dim=3)

        cd = Q4(topo=topo, frames=frame)
        triangles = cd.to_triangles()
        self.assertEqual(triangles.shape[1], 3)
    
    def test_Q4_Geometry(self):
        Q4.Geometry.polybase()
        Q4.Geometry.master_coordinates()
        Q4.Geometry.master_center()
        trimap = Q4.Geometry.trimap()
        self.assertTrue(trimap.shape[1] == 3)
        
if __name__ == "__main__":
    unittest.main()
