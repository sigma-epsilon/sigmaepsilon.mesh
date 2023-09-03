# -*- coding: utf-8 -*-
import unittest

import numpy as np

from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame
from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.cells import L2
from sigmaepsilon.mesh.utils.topology import H8_to_L2
from sigmaepsilon.mesh.utils.space import frames_of_lines


class TestLineCells(unittest.TestCase):
    def test_L2_shape_function_derivatives(self):
        Lx, Ly, Lz = 1, 1, 1
        nx, ny, nz = 2, 2, 2
        coords, topo = grid(size=(Lx, Ly, Lz), shape=(nx, ny, nz), eshape="H8")
        coords, topo = H8_to_L2(coords, topo)
        frame = CartesianFrame(dim=3)
        frames = frames_of_lines(coords, topo)

        pd = PointData(coords=coords)
        cd = L2(topo=topo, frames=frames)
        mesh = PolyData(pd, cd, frame=frame)

        jac = cd.jacobian_matrix()
        self.assertTrue(isinstance(jac, np.ndarray))

        pcoords = L2.Geometry.master_coordinates()
        self.assertTrue(isinstance(pcoords, np.ndarray))

        gdshp = L2.Geometry.shape_function_derivatives(pcoords, jac=jac)
        self.assertTrue(
            gdshp.shape, (topo.shape[0], pcoords.shape[0], topo.shape[1], 1)
        )

        del gdshp, mesh


if __name__ == "__main__":
    unittest.main()
