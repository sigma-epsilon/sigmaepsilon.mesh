import numpy as np
import unittest

from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame
from sigmaepsilon.mesh.cells import H8
from sigmaepsilon.mesh.grid import grid


class TestScenario1(unittest.TestCase):
    def _generate_base_mesh(self):
        gridparams = {
            "size": (1, 1, 1),
            "shape": (4, 4, 4),
            "eshape": "H8",
        }
        coords, topo = grid(**gridparams)

        frame = CartesianFrame(dim=3)

        pd = PointData(coords=coords, frame=frame)
        cd = H8(topo=topo, frames=frame)
        return PolyData(pd, cd)

    def test_scenario_1a(self):
        mesh = PolyData()

        mesh["mesh_1"] = self._generate_base_mesh().centralize()

        # obtain the second mesh by rotating and moving the first one
        mesh["mesh_2"] = (
            mesh["mesh_1"]
            .spin("Space", [0, 0, np.pi / 2], "XYZ", inplace=False)
            .move([0.2, 0, 0])
        )

        # obtain the third mesh by rotating and moving the second one
        mesh["mesh_3"] = (
            mesh["mesh_2"]
            .spin("Space", [0, 0, np.pi / 2], "XYZ", inplace=False)
            .move([0.2, 0, 0])
        )

        # obtain the fourth mesh by absolute transformations
        mesh["mesh_4"] = (
            self._generate_base_mesh()
            .centralize()
            .rotate("Space", [0, 0, 3 * np.pi / 2], "XYZ")
            .move([0.6, 0, 0])
        )

    def test_scenario_1b(self):
        mesh = PolyData()

        ambient_frame = CartesianFrame(dim=3)

        frame_1 = ambient_frame

        frame_2 = ambient_frame.rotate(
            "Space", [0, 0, np.pi / 2], "XYZ", inplace=False
        ).move([0.2, 0, 0], frame=ambient_frame)

        frame_3 = ambient_frame.rotate(
            "Space", [0, 0, np.pi], "XYZ", inplace=False
        ).move([0.4, 0, 0], frame=ambient_frame)

        frame_4 = ambient_frame.rotate(
            "Space", [0, 0, 3 * np.pi / 2], "XYZ", inplace=False
        ).move([0.6, 0, 0], frame=ambient_frame)

        mesh = PolyData()

        base = self._generate_base_mesh().centralize()
        coords, topo = base.coords(), base.topology().to_numpy()

        mesh["mesh_1"] = PolyData(
            PointData(coords=coords), H8(topo=topo), frame=frame_1
        )

        mesh["mesh_2"] = PolyData(
            PointData(coords=coords), H8(topo=topo), frame=frame_2
        )

        mesh["mesh_3"] = PolyData(
            PointData(coords=coords), H8(topo=topo), frame=frame_3
        )

        mesh["mesh_4"] = PolyData(
            PointData(coords=coords), H8(topo=topo), frame=frame_4
        )


if __name__ == "__main__":
    unittest.main()
