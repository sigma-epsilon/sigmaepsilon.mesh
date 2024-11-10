import unittest

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase

from sigmaepsilon.mesh import LineData, PointData
from sigmaepsilon.mesh.cells import L2
from sigmaepsilon.mesh.utils.space import frames_of_lines


class TestLineData(SigmaEpsilonTestCase):
    def setUp(self):
        coords = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float
        )
        topology = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
        frames = frames_of_lines(coords, topology)

        pd = PointData(coords)
        cd = L2(topo=topology, frames=frames)

        self.mesh: LineData = LineData(pd, cd)

    def test_linedata_plot_with_plotly(self):
        self.mesh.plot(backend="plotly")

    def test_linedata_plot_with_pyvista(self):
        self.mesh.plot(backend="pyvista", return_plotter=True)

    def test_linedata_plot_invalid_backend_NotImplrmentedError(self):
        with self.assertRaises(NotImplementedError):
            self.mesh.plot(backend="-$_8%")


if __name__ == "__main__":
    unittest.main()
