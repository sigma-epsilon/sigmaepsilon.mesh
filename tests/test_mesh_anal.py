# -*- coding: utf-8 -*-
import numpy as np
import unittest

import pyvista as pv
import numpy as np
from sigmaepsilon.mesh import PolyData, PointData
from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.cells import H8


class TestMeshAnalysis(unittest.TestCase):
    def test_nodal_adjacency(self):
        d, h, a = 6.0, 15.0, 15.0
        cyl = pv.CylinderStructured(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            radius=np.linspace(d / 2, a / 2, 15),
            height=h,
            theta_resolution=4,
            z_resolution=4,
        )
        pd: PolyData = PolyData.from_pv(cyl)
        pd.nodal_adjacency(frmt="scipy-csr")
        pd.nodal_adjacency(frmt="nx")
        pd.nodal_adjacency(frmt="jagged")
        pd.nodal_adjacency_matrix()
        
    def test_neighbourhood_matrix(self):
        d, h, a = 6.0, 15.0, 15.0
        cyl = pv.CylinderStructured(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            radius=np.linspace(d / 2, a / 2, 15),
            height=h,
            theta_resolution=4,
            z_resolution=4,
        )
        pd: PolyData = PolyData.from_pv(cyl)
        nnm = pd.nodal_neighbourhood_matrix()
        self.assertEqual(nnm.min(), 0)
        self.assertEqual(nnm.max(), 1)
        self.assertEqual(np.sum(nnm.diagonal()), 0)

    def test_knn(self):
        size = 80, 60, 20
        shape = 10, 8, 4
        coords, topo = grid(size=size, shape=shape, eshape="H8")
        pd = PointData(coords=coords)
        cd = H8(topo=topo)
        mesh = PolyData(pd, cd)
        mesh.k_nearest_cell_neighbours(k=3, knn_options=dict(max_distance=10.0))[:5]


if __name__ == "__main__":
    unittest.main()
