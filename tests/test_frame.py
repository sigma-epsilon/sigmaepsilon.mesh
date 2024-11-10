import numpy as np
import unittest

from sigmaepsilon.math.linalg import Vector
from sigmaepsilon.mesh import CartesianFrame


class TestCartesianFrame(unittest.TestCase):
    def test_frame_instantiation(self):
        frame = CartesianFrame()
        frame = CartesianFrame(origo=[1, 1, 1])
        frame = CartesianFrame(origo=np.array([1, 1, 1]))

        self.assertIsInstance(frame.origo, np.ndarray)

        with self.assertRaises(Exception):
            frame = CartesianFrame(origo=dict(a=1))

        with self.assertRaises(ValueError):
            frame = CartesianFrame(origo=np.zeros((3, 3)))

        del frame

    def test_frame_move(self):
        A = CartesianFrame()
        B = A.orient_new("Body", [0, 0, 45 * np.pi / 180], "XYZ")
        B.move(Vector(np.array([1.0, 0, 0]), frame=B))
        B.move(-Vector(np.array([np.sqrt(2) / 2, 0, 0])))
        B.move(-Vector(np.array([0, np.sqrt(2) / 2, 0])))

    def test_frame_tr(self):
        A = CartesianFrame()
        B = A.orient_new("Body", [0, 0, 30 * np.pi / 180], "XYZ")
        C = B.orient_new("Body", [0, 0, 30 * np.pi / 180], "XYZ")
        A = CartesianFrame()
        B = A.orient_new("Body", [0, 0, 30 * np.pi / 180], "XYZ")
        C = B.orient_new("Body", [0, 0, 30 * np.pi / 180], "XYZ")
        A = CartesianFrame()
        v = Vector(np.array([1.0, 0.0, 0.0]), frame=A)
        B = A.fork("Body", [0, 0, 45 * np.pi / 180], "XYZ").move(v)


if __name__ == "__main__":
    unittest.main()
