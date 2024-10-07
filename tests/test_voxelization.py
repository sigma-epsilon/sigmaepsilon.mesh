import unittest
import numpy as np

from sigmaepsilon.mesh import PolyData, PointData
from sigmaepsilon.mesh.space import StandardFrame
from sigmaepsilon.mesh.cells import H8

from sigmaepsilon.mesh.voxelize import voxelize_TET4_H8


class Test_voxelize_TET4_H8(unittest.TestCase):
    def test_1(self):
        coords_TET4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(
            float
        )
        topo_TET4 = np.array([[0, 1, 2, 3]])
        shape = (10, 10, 10)
        hexa_mesh = voxelize_TET4_H8(coords_TET4, topo_TET4, shape=shape, k_max=1)

        frame = StandardFrame(dim=3)
        mesh = PolyData(
            pd=PointData(coords=hexa_mesh[0], frame=frame),
            cd=H8(topo=hexa_mesh[1], frames=frame),
        )

        self.assertTrue(np.isclose(mesh.volume(), 0.2829999999999999))

    def test_2(self):
        coords_TET4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(
            float
        )
        topo_TET4 = np.array([[0, 1, 2, 3]])
        resolution = 0.1
        voxelize_TET4_H8(coords_TET4, topo_TET4, resolution=resolution, k_max=1)


if __name__ == "__main__":
    unittest.main()
