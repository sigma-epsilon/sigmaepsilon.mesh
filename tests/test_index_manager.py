import unittest, doctest

import numpy as np

import sigmaepsilon.mesh.indexmanager
from sigmaepsilon.mesh.indexmanager import IndexManager


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(sigmaepsilon.mesh.indexmanager))
    return tests


class TestIndexManager(unittest.TestCase):
    def test_single(self):
        im = IndexManager()
        self.assertEqual(im.generate(1), 0)
        self.assertEqual(im.generate(1), 1)
        im.recycle(0)
        self.assertEqual(im.generate(1), 0)
        self.assertEqual(im.generate_np(1), 2)

    def test_multi(self):
        im = IndexManager()
        self.assertEqual(im.generate(5), list(range(5)))
        im.recycle(list(range(5)))
        self.assertTrue(np.allclose(im.generate_np(5), np.arange(5)))
        im.recycle(list(range(5)))
        self.assertTrue(np.allclose(im.generate_np(10), np.arange(10)))


if __name__ == "__main__":
    unittest.main()
