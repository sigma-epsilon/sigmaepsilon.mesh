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

    def test_voxelize_TET4_H8_TypeError(self):
        coords_TET4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(
            float
        )
        topo_TET4 = np.array([[0, 1, 2, 3]])
        shape = (10, 10, 10)
        with self.assertRaises(TypeError):
            voxelize_TET4_H8([[1, 2, 3]], topo_TET4, shape=shape, k_max=1)
        with self.assertRaises(TypeError):
            voxelize_TET4_H8(coords_TET4, [[1, 2, 3]], shape=shape, k_max=1)

    def test_voxelize_TET4_H8_ValueError(self):
        coords = np.zeros((4, 3))
        topo = np.zeros((1, 4))
        shape = (10, 10, 10)
        resolution = (1, 1, 1)
        with self.assertRaises(ValueError):
            voxelize_TET4_H8(coords, topo, shape=shape, resolution=resolution)
        with self.assertRaises(ValueError):
            voxelize_TET4_H8(coords, topo)
        with self.assertRaises(ValueError):
            voxelize_TET4_H8(coords, topo, shape=(10, 10, 10, 10))
        with self.assertRaises(ValueError):
            voxelize_TET4_H8(coords, topo, resolution=(10, 10, 10, 10))
        with self.assertRaises(ValueError):
            voxelize_TET4_H8(np.zeros((4, 4)), topo, shape=shape)
        with self.assertRaises(ValueError):
            voxelize_TET4_H8(coords, np.zeros((1, 5)), shape=shape)
        with self.assertRaises(ValueError):
            voxelize_TET4_H8(np.zeros((1, 1, 1)), topo, shape=shape)
        with self.assertRaises(ValueError):
            voxelize_TET4_H8(coords, np.zeros((1, 1, 1)), shape=shape)

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
        voxelize_T3_H8(coords_T3, topo_T3, shape=10, k_max=10)
        voxelize_T3_H8(coords_T3, topo_T3, resolution=0.1, k_max=10)
        voxelize_T3_H8(coords_T3, topo_T3, resolution=100, k_max=10)

    def test_voxelize_T3_H8_TypeError(self):
        coords_T3 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).astype(float)
        topo_T3 = np.array([[0, 1, 2]])
        shape = (10, 10, 10)
        with self.assertRaises(TypeError):
            voxelize_T3_H8([[1, 2, 3]], topo_T3, shape=shape, k_max=1)
        with self.assertRaises(TypeError):
            voxelize_T3_H8(coords_T3, [[1, 2, 3]], shape=shape, k_max=1)

    def test_voxelize_T3_H8_ValueError(self):
        coords = np.zeros((3, 3))
        topo = np.zeros((1, 3))
        shape = (10, 10, 10)
        resolution = (1, 1, 1)
        with self.assertRaises(ValueError):
            voxelize_T3_H8(coords, topo, shape=shape, resolution=resolution)
        with self.assertRaises(ValueError):
            voxelize_T3_H8(coords, topo)
        with self.assertRaises(ValueError):
            voxelize_T3_H8(coords, topo, shape=(10, 10, 10, 10))
        with self.assertRaises(ValueError):
            voxelize_T3_H8(coords, topo, resolution=(10, 10, 10, 10))
        with self.assertRaises(ValueError):
            voxelize_T3_H8(np.zeros((4, 4)), topo, shape=shape)
        with self.assertRaises(ValueError):
            voxelize_T3_H8(coords, np.zeros((1, 5)), shape=shape)
        with self.assertRaises(ValueError):
            voxelize_T3_H8(np.zeros((1, 1, 1)), topo, shape=shape)
        with self.assertRaises(ValueError):
            voxelize_T3_H8(coords, np.zeros((1, 1, 1)), shape=shape)


if __name__ == "__main__":
    unittest.main()
