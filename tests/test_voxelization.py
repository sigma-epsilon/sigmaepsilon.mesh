import unittest
import numpy as np

from sigmaepsilon.mesh.voxelize import voxelize_TET4_H8, voxelize_T3_H8


class TestVoxelize(unittest.TestCase):
    def test_voxelize_TET4_H8_1__SMOKE(self):
        coords_TET4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(
            float
        )
        topo_TET4 = np.array([[0, 1, 2, 3]])
        shape = (10, 10, 10)
        voxelize_TET4_H8(coords_TET4, topo_TET4, shape=shape, k_max=1)

    def test_voxelize_TET4_H8_2__SMOKE(self):
        coords_TET4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(
            float
        )
        topo_TET4 = np.array([[0, 1, 2, 3]])
        resolution = 0.1
        voxelize_TET4_H8(coords_TET4, topo_TET4, resolution=resolution, k_max=1)

    def test_voxelize_T3_H8_1__SMOKE(self):
        coords_T3 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).astype(float)
        topo_T3 = np.array([[0, 1, 2]])
        shape = (10, 10, 10)
        voxelize_T3_H8(coords_T3, topo_T3, shape=shape, k_max=10)


if __name__ == "__main__":
    unittest.main()
