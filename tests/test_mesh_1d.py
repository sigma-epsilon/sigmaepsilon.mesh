# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.mesh.mesh1d import mesh1d_uniform


class TestMesh1d(unittest.TestCase):
    def test_mesh_1d(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        topo = np.array([[0, 1], [1, 2]], dtype=int)
        mesh1d_uniform(coords, topo, (2,))
        mesh1d_uniform(coords, topo, (2,), return_frames=True)


if __name__ == "__main__":
    unittest.main()
