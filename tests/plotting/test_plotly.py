# -*- coding: utf-8 -*-
import unittest
import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.utils.topology.tr import Q4_to_T3, H8_to_L2
from sigmaepsilon.mesh.plotting import (
    triplot_plotly,
    plot_lines_plotly,
    scatter_points_plotly,
)


class TestPlotly(SigmaEpsilonTestCase):
    def test_points(self):
        gridparams = {
            "size": (1200, 600),
            "shape": (4, 4),
            "eshape": (2, 2),
        }
        points, _ = grid(**gridparams)
        data = np.random.rand(len(points))
        scatter_points_plotly(points)
        scatter_points_plotly(points, scalars=data)

    def test_lines(self):
        gridparams = {
            "size": (10, 10, 10),
            "shape": (4, 4, 4),
            "eshape": "H8",
        }
        coords, topo = grid(**gridparams)
        coords, topo = H8_to_L2(coords, topo)
        data = np.random.rand(len(coords), 2)
        plot_lines_plotly(coords, topo)
        plot_lines_plotly(coords, topo, scalars=data)
        plot_lines_plotly(coords, topo, scalars=data, scalar_labels=["X", "Y"])

    def test_triplot(self):
        gridparams = {
            "size": (1200, 600),
            "shape": (4, 4),
            "eshape": (2, 2),
        }
        coordsQ4, topoQ4 = grid(**gridparams)
        points, triangles = Q4_to_T3(coordsQ4, topoQ4, path="grid")
        data = np.random.rand(len(points))
        triplot_plotly(points, triangles, plot_edges=False)
        triplot_plotly(points, triangles, data, plot_edges=False)
        triplot_plotly(points, triangles, data, plot_edges=True)


if __name__ == "__main__":
    unittest.main()
