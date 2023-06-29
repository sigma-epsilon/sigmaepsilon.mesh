# -*- coding: utf-8 -*-
import numpy as np
import unittest


from polymesh.topoarray import TopologyArray


class TestTopo(unittest.TestCase):
    def test_topo_array(self):
        topo1 = np.array([[0, 1], [1, 2], [2, 3]])
        topo2 = np.array([[0, 1, 4], [1, 2, 5], [2, 3, 6]])
        topo = TopologyArray(topo1, topo2)
        topo[1, 1]
        topo[4, 2]


if __name__ == "__main__":
    unittest.main()
