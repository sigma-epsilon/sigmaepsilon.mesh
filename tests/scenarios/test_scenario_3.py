import unittest, sys

import numpy as np

from sigmaepsilon.mesh import PolyData, PointData, LineData
from sigmaepsilon.mesh.space import CartesianFrame
from sigmaepsilon.mesh.cells import H8, TET4, L2
from sigmaepsilon.mesh.utils.topology import H8_to_TET4, H8_to_L2
from sigmaepsilon.mesh.utils.space import frames_of_lines
from sigmaepsilon.mesh.grid import grid as _grid

import pyvista

if sys.platform.startswith("linux"):
    pyvista.start_xvfb()


class TestScenario3(unittest.TestCase):
    def test_scenario_3(self):
        size = 10, 10, 5
        shape = 2, 2, 2
        coords, topo = _grid(size=size, shape=shape, eshape="H8")
        pd = PointData(coords=coords)
        cd = H8(topo=topo)
        grid = PolyData(pd, cd)
        grid.centralize()

        coords = grid.coords()
        topo = grid.topology().to_numpy()
        centers = grid.centers()

        b_left = centers[:, 0] < 0
        b_right = centers[:, 0] >= 0
        b_front = centers[:, 1] >= 0
        b_back = centers[:, 1] < 0
        iTET4 = np.where(b_left)[0]
        iH8 = np.where(b_right & b_back)[0]
        iL2 = np.where(b_right & b_front)[0]
        _, tTET4 = H8_to_TET4(coords, topo[iTET4])
        _, tL2 = H8_to_L2(coords, topo[iL2])
        tH8 = topo[iH8]

        # crate supporting pointcloud
        frame = CartesianFrame(dim=3)
        pd = PointData(coords=coords, frame=frame)
        mesh = PolyData(pd, frame=frame)

        # tetrahedra
        cdTET4 = TET4(topo=tTET4, frames=frame)
        mesh["tetra"] = PolyData(cdTET4, frame=frame)
        mesh["tetra"].config["A", "color"] = "green"

        # hexahedra
        cdH8 = H8(topo=tH8, frames=frame)
        mesh["hex"] = PolyData(cdH8, frame=frame)
        mesh["hex"].config["A", "color"] = "blue"

        # lines
        cdL2 = L2(topo=tL2, frames=frames_of_lines(coords, tL2))
        mesh["line"] = LineData(cdL2, frame=frame)
        mesh["line"].config["A", "color"] = "red"
        mesh["line"].config["A", "line_width"] = 3
        mesh["line"].config["A", "render_lines_as_tubes"] = True

        # finalize the mesh and lock the layout
        mesh.to_standard_form()
        mesh.lock(create_mappers=True)

        # plot with PyVista
        mesh.plot(
            notebook=False,
            jupyter_backend="static",
            config_key=["A"],
            show_edges=True,
            theme="document",
            return_plotter=True,
        )

        # --------------- DATA FROM CELLS TO POINTS ---------------------

        scalars_TET4 = 100 * np.random.rand(len(cdTET4))
        cdTET4.db["scalars"] = scalars_TET4

        scalars_H8 = 100 * np.random.rand(len(cdH8))
        cdH8.db["scalars"] = scalars_H8

        scalars_L2 = 100 * np.random.rand(len(cdL2))
        cdL2.db["scalars"] = scalars_L2
        mesh["line"].config["B", "render_lines_as_tubes"] = True
        mesh["line"].config["B", "line_width"] = 3

        mesh.plot(
            notebook=False,
            jupyter_backend="static",
            config_key=["A"],
            cmap="plasma",
            show_edges=True,
            scalars="scalars",
            theme="document",
            return_plotter=True,
        )

        scalars = mesh.pd.pull("scalars")  # or simply pd.pull('scalars')
        # print(scalars.shape)
        # print(mesh.coords().shape)
        self.assertEqual(scalars.shape, (27,))
        self.assertEqual(mesh.coords().shape, (27, 3))

        mesh.plot(
            notebook=False,
            jupyter_backend="static",
            config_key=["A"],
            show_edges=True,
            scalars=scalars,
            cmap="plasma",
            theme="document",
            return_plotter=True,
        )

        # --------------- DATA FROM POINTS TO CELLS ---------------------

        scalars_on_points = 100 * np.random.rand(len(coords))
        mesh.pd.db["scalars"] = scalars_on_points

        hex_data = mesh["hex"].cd.pull("scalars")
        # print(hex_data.shape)
        # print(mesh["hex"].topology().shape)
        self.assertEqual(hex_data.shape, (2, 8, 1))
        self.assertEqual(mesh["hex"].topology().shape, (2, 8))

        # ------------- CUSTOMIZING THE DISTRIBUTION MECHANISM ----------------

        cdTET4.db["scalars"] = np.full(len(cdTET4), -100)
        cdH8.db["scalars"] = np.full(len(cdH8), 100)
        cdL2.db["scalars"] = np.full(len(cdL2), 0)
        mesh.plot(
            notebook=False,
            jupyter_backend="static",
            config_key=["B"],
            cmap="plasma",
            show_edges=True,
            scalars="scalars",
            theme="document",
            return_plotter=True,
        )

        scalars = mesh.pd.pull("scalars")
        mesh.plot(
            notebook=False,
            jupyter_backend="static",
            config_key=["A"],
            show_edges=True,
            scalars=scalars,
            cmap="jet",
            theme="document",
            return_plotter=True,
        )

        v = mesh.volumes()
        idH8 = mesh["hex"].cd.id  # cell indices of hexahedra
        v[idH8] *= 5  # 500% of original weight
        ndf = mesh.nodal_distribution_factors(weights=v)
        scalars = mesh.pd.pull("scalars", ndf=ndf)
        mesh.plot(
            notebook=False,
            jupyter_backend="static",
            config_key=["A"],
            show_edges=True,
            scalars=scalars,
            cmap="jet",
            theme="document",
            return_plotter=True,
        )


if __name__ == "__main__":
    unittest.main()
