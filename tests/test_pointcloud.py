import unittest
from polymesh.space import PointCloud
from polymesh.triang import triangulate
from numba import njit
import numpy as np


class TestPointCloud(unittest.TestCase):
    def test_1(self):
        coords, *_ = triangulate(size=(800, 600), shape=(3, 3))
        inds = np.arange(len(coords))
        coords = PointCloud(coords, inds=inds)
        coords.bounds()
        coords.closest(coords[:3])
        coords.furthest(coords[:3])
        coords.frame
        coords.x
        coords.y
        coords.z
        coords.center()
        coords.idsort()

        self.assertTrue(np.all(np.isclose(coords[3:6].inds, [3, 4, 5])))
        self.assertTrue(np.isclose(coords.index_of_closest(coords.center()), 4))
        self.assertTrue(
            np.all(np.isclose(coords.closest(coords.center()).show(), coords.center()))
        )

        @njit
        def numba_nopython(arr):
            return arr[0], arr.x, arr.data, arr.inds

        c = np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 0]])
        COORD = PointCloud(c, inds=np.array([0, 1, 2, 3]))
        numba_nopython(COORD)

        coords.centralize()
        coords.rotate("Body", [0, 0, np.pi / 2], "XYZ").bounds()
        coords.sort_indices()
        d = np.array([0.0, 1.0, 0.0])
        coords.move(d).move(d)


if __name__ == "__main__":
    unittest.main()
