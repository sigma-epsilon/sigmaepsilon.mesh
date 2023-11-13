# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame
from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.cells import Q8


class TestQ8(SigmaEpsilonTestCase):
    def test_Q8(self):
        Lx, Ly = 800, 600
        nx, ny = 8, 6
        xbins = np.linspace(0, Lx, nx + 1)
        ybins = np.linspace(0, Ly, ny + 1)
        bins = xbins, ybins
        coords, topo = grid(bins=bins, eshape="Q8")
        frame = CartesianFrame(dim=3)

        pd = PointData(coords=coords)
        cd = Q8(topo=topo, frames=frame)
        _ = PolyData(pd, cd)

        self.assertTrue(np.isclose(cd.area(), Lx * Ly))
        self.assertTrue(np.isclose(cd.volume(), Lx * Ly))
        self.assertEqual(cd.jacobian_matrix().shape, (topo.shape[0], 8, 2, 2))


if __name__ == "__main__":
    unittest.main()
