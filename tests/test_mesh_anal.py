# -*- coding: utf-8 -*-
import numpy as np
import unittest

import pyvista as pv
import numpy as np
from polymesh import PolyData
from polymesh.grid import Grid


class TestMeshAnalysis(unittest.TestCase):
    def test_nodal_adjacency(self):
        d, h, a = 6.0, 15.0, 15.0
        cyl = pv.CylinderStructured(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            radius=np.linspace(d / 2, a / 2, 15),
            height=h,
            theta_resolution=100,
            z_resolution=40,
        )
        pd = PolyData.from_pv(cyl)
        pd.nodal_adjacency(frmt="scipy-csr")
        pd.nodal_adjacency(frmt="nx")
        pd.nodal_adjacency(frmt="jagged")
        pd.nodal_adjacency_matrix()

    def test_knn(self):
        size = 80, 60, 20
        shape = 10, 8, 4
        grid = Grid(size=size, shape=shape, eshape="H8")
        grid.k_nearest_cell_neighbours(k=3, knn_options=dict(max_distance=10.0))[:5]


if __name__ == "__main__":
    unittest.main()
