import unittest
import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.utils.topology.tr import Q4_to_T3
from sigmaepsilon.mesh import triangulate
from sigmaepsilon.mesh.recipes import circular_disk
from sigmaepsilon.mesh.plotting import (
    triplot_mpl_data,
    triplot_mpl_mesh,
    triplot_mpl_hinton,
)
import matplotlib.tri as mpltri


class TestMplTriplot(SigmaEpsilonTestCase):
    def test_triplot(self):
        gridparams = {
            "size": (1200, 600),
            "shape": (30, 15),
            "eshape": (2, 2),
            "origo": (0, 0),
            "start": 0,
        }
        coordsQ4, topoQ4 = grid(**gridparams)
        points, triangles = Q4_to_T3(coordsQ4, topoQ4, path="grid")
        triobj = triangulate(points=points[:, :2], triangles=triangles)[-1]
        triplot_mpl_mesh(triobj)
        triplot_mpl_mesh(triobj, zorder=1)
        triplot_mpl_mesh(triobj, fcolor="b")

        data = np.random.rand(len(triangles))
        triplot_mpl_data(triobj, data=data)
        triplot_mpl_hinton(triobj, data=data)

        data = np.random.rand(len(triangles), 3)
        triplot_mpl_data(triobj, data=data)

        data = np.random.rand(len(points))
        triplot_mpl_data(triobj, data=data, cmap="bwr")
        triplot_mpl_data(triobj, data=data, cmap="bwr", refine=True, draw_contours=True)
        refiner = mpltri.UniformTriRefiner(triobj)
        triplot_mpl_data(triobj, data=data, cmap="bwr", refiner=refiner, nlevels=10)

    def circular_disk(self):
        n_angles = 60
        n_radii = 30
        min_radius = 5
        max_radius = 25
        disk = circular_disk(n_angles, n_radii, min_radius, max_radius)
        triobj = triangulate(points=disk.coords()[:, :2], triangles=disk.topology())[-1]
        triplot_mpl_mesh(triobj)


if __name__ == "__main__":
    unittest.main()
