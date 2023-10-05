import unittest, sys

from sigmaepsilon.mesh.space import CartesianFrame
from sigmaepsilon.mesh.recipes import cylinder
from sigmaepsilon.mesh import PolyData, LineData, PointData
from sigmaepsilon.mesh.cells import H8, Q4, L2
from sigmaepsilon.mesh.utils.topology import H8_to_L2, H8_to_Q4
from sigmaepsilon.mesh.utils.topology import detach_mesh_bulk
from sigmaepsilon.mesh.utils.space import frames_of_lines, frames_of_surfaces
from sigmaepsilon.math import minmax

import numpy as np
from k3d.colormaps import matplotlib_color_maps
import pyvista
from pyvista import themes


if sys.platform.startswith("linux"):
    pyvista.start_xvfb()


class TestPolyDataPlot(unittest.TestCase):
    def setUp(self):
        min_radius = 5
        max_radius = 25
        h = 50
        angle = 1

        shape = (min_radius, max_radius), angle, h
        frame = CartesianFrame(dim=3)
        cyl = cylinder(shape, size=5.0, voxelize=True, frame=frame)

        coords = cyl.coords()
        topo = cyl.topology().to_numpy()
        centers = cyl.centers()

        cxmin, cxmax = minmax(centers[:, 0])
        czmin, czmax = minmax(centers[:, 2])
        cxavg = (cxmin + cxmax) / 2
        czavg = (czmin + czmax) / 2
        b_upper = centers[:, 2] > czavg
        b_lower = centers[:, 2] <= czavg
        b_left = centers[:, 0] < cxavg
        b_right = centers[:, 0] >= cxavg
        iL2 = np.where(b_upper & b_right)[0]
        iTET4 = np.where(b_upper & b_left)[0]
        iH8 = np.where(b_lower)[0]
        _, tL2 = H8_to_L2(coords, topo[iL2])
        _, tQ4 = H8_to_Q4(coords, topo[iTET4])
        tH8 = topo[iH8]

        pd = PointData(coords=coords, frame=frame)
        mesh = PolyData(pd)

        cdL2 = L2(topo=tL2, frames=frames_of_lines(coords, tL2))
        mesh["lines", "L2"] = LineData(cdL2)

        cdQ4 = Q4(topo=tQ4, frames=frames_of_surfaces(coords, tQ4))
        mesh["surfaces", "Q4"] = PolyData(cdQ4)

        cH8, tH8 = detach_mesh_bulk(coords, tH8)
        pdH8 = PointData(coords=cH8, frame=frame)
        cdH8 = H8(topo=tH8, frames=frame)
        mesh["bodies", "H8"] = PolyData(pdH8, cdH8)

        mesh.to_standard_form()
        mesh.lock(create_mappers=True)

        self.cdL2 = cdL2
        self.cdQ4 = cdQ4
        self.cdH8 = cdH8
        self.tH8 = tH8
        self.mesh = mesh

    def test_pyvista(self):
        mesh: PolyData = self.mesh

        mesh.plot(
            notebook=False,
            jupyter_backend="static",
            show_edges=True,
            theme="document",
            return_plotter=True,
        )

        mesh["surfaces", "Q4"].plot(
            notebook=False,
            jupyter_backend="static",
            show_edges=True,
            theme="document",
            return_plotter=True,
        )

        mesh["bodies", "H8"].surface().pvplot(
            notebook=False,
            jupyter_backend="static",
            show_edges=True,
            theme="document",
            return_plotter=True,
        )

        # plotting with data
        mesh["lines", "L2"].config["pyvista", "plot", "color"] = "red"
        mesh["lines", "L2"].config["pyvista", "plot", "line_width"] = 1.5
        mesh["lines", "L2"].config["pyvista", "plot", "render_lines_as_tubes"] = True

        mesh["surfaces", "Q4"].config["pyvista", "plot", "show_edges"] = True
        mesh["surfaces", "Q4"].config["pyvista", "plot", "color"] = "yellow"
        mesh["surfaces", "Q4"].config["pyvista", "plot", "opacity"] = 0.3

        mesh["bodies", "H8"].config["pyvista", "plot", "show_edges"] = True
        mesh["bodies", "H8"].config["pyvista", "plot", "color"] = "cyan"
        mesh["bodies", "H8"].config["pyvista", "plot", "opacity"] = 1.0

        mesh.pvplot(
            notebook=False,
            jupyter_backend="static",
            cmap="plasma",
            window_size=(600, 400),
            config_key=["pyvista", "plot"],
            theme="document",
            return_plotter=True,
        )

        mesh["bodies", "H8"].config["pyvista", "plot", "scalars"] = np.random.rand(
            self.tH8.shape[0]
        )
        ncTET4 = mesh["surfaces", "Q4"].coords(from_cells=True).shape[0]
        mesh["surfaces", "Q4"].config[
            "pyvista", "plot", "scalars"
        ] = 2 * np.random.rand(ncTET4)
        mesh["surfaces", "Q4"].config["pyvista", "plot", "opacity"] = 1.0
        mesh.pvplot(
            notebook=False,
            jupyter_backend="static",
            window_size=(600, 400),
            config_key=["pyvista", "plot"],
            cmap="plasma",
            theme="document",
            return_plotter=True,
        )

        block = mesh.blocks_of_cells(2345)[2345]
        block.pvplot(
            notebook=False,
            jupyter_backend="static",
            window_size=(600, 400),
            config_key=["pyvista", "plot"],
            cmap="jet",
            theme="document",
            return_plotter=True,
        )

        block.pvplot(
            notebook=False,
            jupyter_backend="static",
            window_size=(600, 400),
            config_key=["pyvista", "plot"],
            cmap="jet",
            theme="dark",
            return_plotter=True,
        )

        block.pvplot(
            notebook=False,
            jupyter_backend="static",
            window_size=(600, 400),
            config_key=["pyvista", "plot"],
            cmap="jet",
            theme="bw",
            return_plotter=True,
        )

        block.pvplot(
            notebook=False,
            jupyter_backend="static",
            window_size=(600, 400),
            config_key=["pyvista", "plot"],
            cmap="jet",
            theme=themes.Theme(),
            return_plotter=True,
        )

        block.pvplot(
            notebook=False,
            jupyter_backend="static",
            window_size=(600, 400),
            config_key=["pyvista", "plot"],
            cmap="jet",
            edge_color="black",
            return_plotter=True,
        )

    def test_k3d(self):
        mesh: PolyData = self.mesh

        mesh["lines", "L2"].config["k3d", "plot", "color"] = "red"
        mesh["lines", "L2"].config["k3d", "plot", "width"] = 0.2

        mesh["surfaces", "Q4"].config["k3d", "plot", "color"] = "yellow"
        mesh["surfaces", "Q4"].config["k3d", "plot", "opacity"] = 0.3

        mesh["bodies", "H8"].config["k3d", "plot", "color"] = "cyan"
        mesh["bodies", "H8"].config["k3d", "plot", "opacity"] = 1.0

        _ = mesh["lines", "L2"].k3dplot(
            config_key=["k3d", "plot"], menu_visibility=False
        )
        _ = mesh.k3dplot(config_key=["k3d", "plot"], menu_visibility=False)

        mesh["lines", "L2"].cd.db["scalars"] = 100 * np.random.rand(len(self.cdL2))
        mesh["surfaces", "Q4"].cd.db["scalars"] = 100 * np.random.rand(len(self.cdQ4))
        mesh["bodies", "H8"].cd.db["scalars"] = 100 * np.random.rand(len(self.cdH8))
        scalars = mesh.pd.pull("scalars")

        cmap = matplotlib_color_maps.Jet
        _ = mesh.plot(
            backend="k3d",
            scalars=scalars,
            menu_visibility=False,
            cmap=cmap,
            notebook=True,
        )


if __name__ == "__main__":
    unittest.main()
