# -*- coding: utf-8 -*-
import unittest
import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.mesh.examples import compound_mesh


class TestPyVistaPlotter(SigmaEpsilonTestCase):
    def test_compound_mesh(self):
        mesh = compound_mesh()
        mesh["lines", "L2"].config["pyvista", "plot", "label"] = "L2"
        mesh["surfaces", "Q4"].config["pyvista", "plot", "label"] = "Q4"
        mesh["bodies", "H8"].config["pyvista", "plot", "label"] = "H8"

        camera_position = [
            (0.3914, 0.4542, 0.7670),
            (0.0243, 0.0336, -0.0222),
            (-0.2148, 0.8998, -0.3796),
        ]

        mesh.pvplot(
            theme="document", return_plotter=True, config_key=("pyvista", "plot")
        )
        mesh.pvplot(
            camera_position=camera_position,
            show_scalar_bar=False,
            opacity=0.5,
            add_legend=True,
            return_img=True,
            config_key=("pyvista", "plot"),
        )


if __name__ == "__main__":
    unittest.main()
