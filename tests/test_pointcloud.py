import unittest
from numba import njit
import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.math.linalg import rotation_matrix
from sigmaepsilon.mesh import CartesianFrame
from sigmaepsilon.mesh.space import PointCloud
from sigmaepsilon.mesh.triang import triangulate


class TestPointCloud(SigmaEpsilonTestCase):
    def test_1(self):
        coords, *_ = triangulate(size=(800, 600), shape=(3, 3))
        inds = np.arange(len(coords))
        coords = PointCloud(coords, inds=inds)
        coords.bounds()
        coords.closest(coords[:3])
        coords.furthest(coords[:3])
        coords.frame
        coords.x()
        coords.y()
        coords.z()
        coords.center()
        coords.idsort()
        coords.id
        coords[0]

        self.assertTrue(np.all(np.isclose(coords[3:6].inds, [3, 4, 5])))
        self.assertTrue(np.isclose(coords.index_of_closest(coords.center()), 4))
        self.assertTrue(
            np.all(np.isclose(coords.closest(coords.center()).show(), coords.center()))
        )

        @njit
        def numba_nopython(arr):
            return arr[0], arr.x, arr.y, arr.z, arr.data, arr.inds

        c = np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 0]])
        COORD = PointCloud(c, inds=np.array([0, 1, 2, 3]))
        numba_nopython(COORD)

        coords.centralize()
        coords.rotate("Body", [0, 0, np.pi / 2], "XYZ").bounds()
        coords.sort_indices()
        d = np.array([0.0, 1.0, 0.0])
        coords.move(d).move(d)

    def test_frames(self):
        frame = CartesianFrame(dim=3)
        coords, *_ = triangulate(size=(800, 600), shape=(3, 3))
        inds = np.arange(len(coords))
        points = PointCloud(coords, inds=inds)
        points.frame = frame

        self.assertFailsProperly(TypeError, setattr, points, "frame", "_")

    def _rotation_frames(self):
        ambient = CartesianFrame(dim=3)
        rotated = ambient.orient_new("Body", [0.3, -0.2, 0.5], "XYZ")
        return {"ambient": ambient, "rotated": rotated}

    def test_rotate_chained_about_different_axes(self):
        # Issue #174: the rotation must be about the ambient axes for any frame.
        X = np.random.default_rng(0).random((6, 3))
        r1 = ("Space", [0, 0, np.pi / 2], "XYZ")
        r2 = ("Space", [np.pi / 2, 0, 0], "XYZ")
        r3 = ("Body", [0.1, 0.7, -0.4], "XYZ")
        R1, R2, R3 = (rotation_matrix(*r) for r in (r1, r2, r3))
        for name, frame in self._rotation_frames().items():
            with self.subTest(frame=name):
                points = PointCloud(X.copy(), frame=frame)
                x = points.show()
                points.rotate(*r1).rotate(*r2).rotate(*r3)
                self.assertTrue(np.allclose(points.show(), x @ R1 @ R2 @ R3))
                self.assertIs(points.frame, frame)

    def test_rotate_keeps_origin_of_frame(self):
        X = np.random.default_rng(1).random((5, 3))
        r = ("Space", [0.2, 0, np.pi / 3], "XYZ")
        frame = CartesianFrame(dim=3, origo=[1.0, 2.0, 3.0])
        points = PointCloud(X.copy(), frame=frame)
        x = points.show()
        o = np.array(frame.origo)
        points.rotate(*r)
        # the coordinates in the frame of the point cloud rotate about its origin
        self.assertTrue(np.allclose(points.show() - o, (x - o) @ rotation_matrix(*r)))

    def test_rotate_with_frame(self):
        X = np.random.default_rng(2).random((6, 3))
        r = ("Space", [0.4, 0.1, np.pi / 2], "XYZ")
        for name, frame in self._rotation_frames().items():
            with self.subTest(frame=name):
                expected = PointCloud(X.copy(), frame=frame).rotate(*r).show()
                target = frame.orient_new(*r)
                points = PointCloud(X.copy(), frame=frame).rotate(target)
                self.assertTrue(np.allclose(points.show(), expected))
                self.assertIs(points.frame, frame)


if __name__ == "__main__":
    unittest.main()
