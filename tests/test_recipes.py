import unittest
import numpy as np


from polymesh.recipes import circular_disk, ribbed_plate, perforated_cube, cylinder


class TestRecipes(unittest.TestCase):
    def test_circular_disk(self):
        mesh = circular_disk(120, 60, 5, 25)
        area = np.pi * (25**2 - 5**2)
        self.assertTrue(np.isclose(mesh.area(), area, rtol=1e-3, atol=1e-3))

    def test_ribbed_plate(self):
        mesh = ribbed_plate(lx=5.0, ly=5.0, t=1.0)
        self.assertTrue(np.isclose(mesh.volume(), 25.0, rtol=1e-3, atol=1e-3))
        ribbed_plate(
            lx=5.0, ly=5.0, t=1.0, wx=1.0, hx=2.0, ex=0.05, wy=1.0, hy=2.0, ey=-0.05
        )

    def test_perforated_plate(self):
        params = dict(lx=1, ly=1, lz=0.05, radius=0.2, lmax=0.05)
        vol = (params["lx"] * params["ly"] - np.pi * params["radius"] ** 2) * params[
            "lz"
        ]

        mesh = perforated_cube(**params, order=1, prismatic=False)
        self.assertTrue(np.isclose(mesh.volume(), vol, rtol=1e-3, atol=1e-3))

        mesh = perforated_cube(**params, order=2, prismatic=False)
        self.assertTrue(np.isclose(mesh.volume(), vol, rtol=1e-3, atol=1e-3))

        mesh = perforated_cube(**params, order=1, prismatic=True)
        self.assertTrue(np.isclose(mesh.volume(), vol, rtol=1e-3, atol=1e-3))

        mesh = perforated_cube(**params, order=2, prismatic=True)
        self.assertTrue(np.isclose(mesh.volume(), vol, rtol=1e-3, atol=1e-3))

    def test_cylinder(self):
        n_angles = 60
        n_radii = 30
        min_radius = 5
        max_radius = 25
        n_z = 20
        h = 50
        angle = 1

        vol = np.pi * (max_radius**2 - min_radius**2) * h

        shape = (min_radius, max_radius), angle, h
        size = n_radii, n_angles, n_z

        cyl = cylinder(shape, size, voxelize=True)
        self.assertTrue(np.isclose(cyl.volume(), vol, rtol=1e-3, atol=1e-3))

        cyl = cylinder(shape, size, voxelize=False)
        self.assertTrue(np.isclose(cyl.volume(), vol, rtol=1e-2, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
