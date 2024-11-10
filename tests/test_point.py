import unittest
import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.math.linalg import ReferenceFrame
from sigmaepsilon.mesh import CartesianFrame
from sigmaepsilon.mesh.space import Point


class TestPoint(SigmaEpsilonTestCase):
    def test_instantiation(self):
        point = Point([1, 1, 1])
        point = Point(np.array([1, 1, 1]))

        with self.assertRaises(Exception):
            point = Point(dict(a=1))

        frame = CartesianFrame(dim=3)
        _frame = ReferenceFrame(dim=3)

        point = Point([1, 1, 1], frame=frame)
        point = Point([1, 1, 1], frame=_frame)
        point = Point([1, 1, 1], frame=frame.axes)
        point = Point([1, 1, 1], frame=frame.axes.tolist())

        point.id
        point.gid


if __name__ == "__main__":
    unittest.main()
