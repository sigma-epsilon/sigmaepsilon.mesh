import unittest
from polymesh import PolyData
from polymesh.trimesh import TriMesh
from polymesh.grid import Grid
from polymesh.space import StandardFrame
import numpy as np


class TestCompoundMesh(unittest.TestCase):
    def test_1(self):
        A = StandardFrame(dim=3)
        tri = TriMesh(size=(100, 100), shape=(4, 4), frame=A)
        grid2d = Grid(size=(100, 100), shape=(4, 4), eshape="Q4", frame=A)
        grid3d = Grid(size=(100, 100, 20), shape=(4, 4, 2), eshape="H8", frame=A)
        mesh = PolyData(frame=A)
        mesh["tri", "T3"] = tri.move(np.array([0.0, 0.0, -50]))
        mesh["grids", "Q4"] = grid2d.move(np.array([0.0, 0.0, 50]))
        mesh["grids", "H8"] = grid3d
        mesh.to_standard_form(inplace=True)
        aT3 = mesh["tri"].area()
        aQ4 = mesh["grids", "Q4"].area()
        V0 = aT3 + aQ4 + mesh["grids", "H8"].volume()
        V1 = mesh.volume()
        ndf = mesh.nodal_distribution_factors()
        self.assertTrue(np.isclose(ndf.data.min(), 0.125))
        self.assertTrue(np.isclose(ndf.data.max(), 1.0))
        self.assertTrue(np.isclose(aT3, 10000.0))
        self.assertTrue(np.isclose(aQ4, 10000.0))
        self.assertTrue(np.isclose(V1, 220000.0))
        self.assertTrue(np.isclose(V0, V1))
        self.assertTrue(np.all(np.isclose(mesh["tri"].center(), [50.0, 50.0, -50.0])))
        self.assertTrue(
            np.all(np.isclose(mesh["grids", "Q4"].center(), [50.0, 50.0, 50.0]))
        )
        self.assertTrue(
            np.all(np.isclose(mesh["grids", "H8"].center(), [50.0, 50.0, 10.0]))
        )
        self.assertTrue(mesh.topology().is_jagged())


if __name__ == "__main__":
    unittest.main()
