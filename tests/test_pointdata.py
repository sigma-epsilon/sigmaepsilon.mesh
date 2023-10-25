# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.math.linalg import FrameLike
from sigmaepsilon.mesh import CartesianFrame, PointData, triangulate
from sigmaepsilon.mesh import PointData


class TestPointData(SigmaEpsilonTestCase):
    def test_pointdata(self):
        A = CartesianFrame(dim=3)
        coords = triangulate(size=(800, 600), shape=(10, 10))[0]
        pd = PointData(coords=coords)
        self.assertIsInstance(pd.frame, FrameLike)
        pd = PointData(coords=coords, frame=A)
        self.assertIsInstance(pd.frame, FrameLike)
        nP = len(pd)

        pd.activity = np.ones((nP), dtype=bool)
        self.assertTrue(pd.has_activity)
        self.assertRaises(TypeError, setattr, pd, "activity", "a")
        self.assertRaises(
            ValueError, setattr, pd, "activity", np.ones((nP), dtype=float)
        )
        self.assertRaises(
            ValueError, setattr, pd, "activity", np.ones((nP, 2), dtype=bool)
        )
        self.assertRaises(
            ValueError, setattr, pd, "activity", np.ones((nP - 1), dtype=bool)
        )

        pd.id = np.arange(nP)
        self.assertTrue(pd.has_id)
        self.assertRaises(TypeError, setattr, pd, "id", "a")
        self.assertRaises(ValueError, setattr, pd, "id", np.ones((nP), dtype=float))
        self.assertRaises(ValueError, setattr, pd, "id", np.ones((nP, 2), dtype=int))
        self.assertRaises(ValueError, setattr, pd, "id", np.ones((nP - 1), dtype=int))

        pd.x = coords
        self.assertTrue(pd.has_x)
        self.assertRaises(TypeError, setattr, pd, "x", "_")
        self.assertRaises(ValueError, setattr, pd, "x", np.zeros((3, 3, 3)))


if __name__ == "__main__":
    unittest.main()
