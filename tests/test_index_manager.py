import unittest

import numpy as np

from sigmaepsilon.mesh.indexmanager import IndexManager


class TestIndexManager(unittest.TestCase):
    def test_main(self):
        im = IndexManager()
        self.assertEqual(im.generate(1), 0)
        self.assertEqual(im.generate(1), 1)
        im.recycle(0)
        self.assertEqual(im.generate(1), 0)
        self.assertEqual(im.generate_np(1), 2)
        
        im = IndexManager()
        self.assertEqual(im.generate(5), list(range(5)))
        im.recycle(list(range(5)))
        self.assertTrue(np.allclose(im.generate_np(5), np.arange(5)))
        im.recycle(list(range(5)))
        self.assertTrue(np.allclose(im.generate_np(10), np.arange(10)))


if __name__ == "__main__":
    unittest.main()
        