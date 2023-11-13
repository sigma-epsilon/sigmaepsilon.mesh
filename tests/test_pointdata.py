# -*- coding: utf-8 -*-
import numpy as np
import unittest
from awkward import Array, Record

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.math.linalg import FrameLike
from sigmaepsilon.mesh import CartesianFrame, PointData, triangulate


class TestPointData(SigmaEpsilonTestCase):
    def test_pointdata(self):
        A = CartesianFrame(dim=3)
        coords = triangulate(size=(10, 10), shape=(10, 10))[0]
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

        self.assertIsInstance(pd["x"], Array)


class TestPointDataMagicFunctions(SigmaEpsilonTestCase):
    def test_contains(self):
        A = CartesianFrame(dim=3)
        coords, *_ = triangulate(size=(100, 100), shape=(4, 4))
        pd = PointData(coords=coords, frame=A)
        self.assertIn("x", pd)

        x = np.array([[0.0, 0.0, 0.0]], dtype=float)
        pd = PointData(coords=x)
        pd["random_data"] = [1.0]
        self.assertIn("random_data", pd)
        self.assertFalse("__" in pd)

    def test_setitem(self):
        x = np.array([[0.0, 0.0, 0.0]], dtype=float)
        pd = PointData(coords=x)
        pd["random_data"] = [1.0]

        with self.assertRaises(ValueError) as cm:
            pd["random_data"] = [0.0, 0.0]
        the_exception = cm.exception
        self.assertEqual(
            the_exception.args[0],
            "The provided value must have the same length as the database.",
        )

        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
        pd = PointData(coords=x)
        pd["random_data"] = [[0.0], [0.0, 0.0]]

        with self.assertRaises(TypeError) as cm:
            pd["random_data"] = "_"
        the_exception = cm.exception
        self.assertEqual(
            the_exception.args[0],
            "Expected a sequence, got <class 'str'>",
        )

        with self.assertRaises(TypeError) as cm:
            pd[0] = "_"
        the_exception = cm.exception
        self.assertEqual(
            the_exception.args[0],
            "Expected a string, got <class 'int'>",
        )

    def test_getitem(self):
        x = np.array([[0.0, 0.0, 0.0]], dtype=float)
        pd = PointData(coords=x)
        pd["random_data"] = [1.0]

        self.assertTrue(len(pd["x"]) == 1)
        self.assertTrue(len(pd["random_data"]) == 1)
        self.assertIsInstance(pd[0], Record)

    def test_hasattr(self):
        x = np.array([[0.0, 0.0, 0.0]], dtype=float)
        pd = PointData(coords=x)
        pd["random_data"] = [1.0]

        self.assertTrue(hasattr(pd, "x"))
        self.assertTrue(hasattr(pd, "random_data"))

    def test_getattr(self):
        x = np.array([[10.0, 110.0, 50.0]], dtype=float)
        pd = PointData(coords=x)
        random_data = [66.0]
        pd["random_data"] = random_data
        random_data = np.array(random_data)
        
        self.assertTrue(np.allclose(getattr(pd, "x"), x))
        self.assertTrue(np.allclose(getattr(pd, "random_data").to_numpy(), random_data))
        self.assertTrue(np.allclose(pd.random_data.to_numpy(), random_data))


class TestPointDataExports(SigmaEpsilonTestCase):
    def setUp(self) -> None:
        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
        pd = PointData(coords=x)
        pd["random_data"] = [[0.0], [0.0, 0.0]]
        pd["random_data_2"] = [[0.0], [0.0]]
        self.pd = pd
        
    def test_to_numpy(self):
        self.assertIsInstance(self.pd.to_numpy("random_data_2"), np.ndarray)

    def test_to_ak(self):
        arr = self.pd.to_ak()
        self.assertIsInstance(arr, Array)
        arr = self.pd.to_ak(fields=["random_data"])
        self.assertIsInstance(arr, Array)
        
        arr = self.pd.to_ak(asarray=True)
        self.assertIsInstance(arr, Array)
        arr = self.pd.to_ak(asarray=True, fields=["random_data"])
        self.assertIsInstance(arr, Array)
    
    def test_to_akarray(self):
        arr = self.pd.to_akarray()
        self.assertIsInstance(arr, Array)
        arr[0]["random_data"]
        
        arr = self.pd.to_akarray(fields=["random_data"])
        self.assertIsInstance(arr, Array)
        
    def test_to_akrecord(self):
        arr = self.pd.to_akrecord()
        self.assertIsInstance(arr, Array)
        arr[0]["random_data"]
        
        arr = self.pd.to_akrecord(fields=["random_data"])
        self.assertIsInstance(arr, Array)
        
    def test_to_dict(self):
        d = self.pd.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("random_data", d)
        self.assertEqual(len(d), len(self.pd.db.fields))
        
        d = self.pd.to_dict(fields=["random_data"])
        self.assertIsInstance(d, dict)
        self.assertIn("random_data", d)
        self.assertEqual(len(d), 1)
        
    def test_to_list(self):
        res = self.pd.to_list()
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), len(self.pd))
        self.assertIsInstance(res[0], dict)
        self.assertIn("random_data", res[0])
        
        res = self.pd.to_list(fields=["random_data"])
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), len(self.pd))
        self.assertIsInstance(res[0], dict)
        self.assertIn("random_data", res[0])
        self.assertEqual(len(res[0]), 1)


if __name__ == "__main__":
    unittest.main()
