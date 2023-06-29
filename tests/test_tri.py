import numpy as np
import unittest

from polymesh.trimesh import TriMesh
from polymesh.space import CartesianFrame
from polymesh.recipes import circular_disk
from polymesh.cells import T3, T6


class TestTri(unittest.TestCase):
    def test_area_T3(self):
        def test_area_T3(Lx, Ly, nx, ny):
            try:
                A = CartesianFrame(dim=3)
                mesh = TriMesh(size=(Lx, Ly), shape=(nx, ny), frame=A, celltype=T3)
                assert np.isclose(mesh.area(), Lx * Ly)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e

        assert test_area_T3(1.0, 1.0, 2, 2)

    def test_area_T6(self):
        def test_area_T6(Lx, Ly, nx, ny):
            try:
                A = CartesianFrame(dim=3)
                mesh = TriMesh(size=(Lx, Ly), shape=(nx, ny), frame=A, celltype=T6)
                assert np.isclose(mesh.area(), Lx * Ly)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e

        assert test_area_T6(1.0, 1.0, 2, 2)

    def test_area_circular_disk_T3(self):
        def test_area_circular_disk_T3(min_radius, max_radius, n_angles, n_radii):
            try:
                mesh = circular_disk(n_angles, n_radii, min_radius, max_radius)
                a = np.pi * (max_radius**2 - min_radius**2)
                assert np.isclose(mesh.area(), a, atol=0, rtol=a / 1000)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e

        assert test_area_circular_disk_T3(1.0, 10.0, 120, 80)

    def test_area_circular_disk_T6(self):
        def test_area_circular_disk_T6(min_radius, max_radius, n_angles, n_radii):
            try:
                mesh = circular_disk(n_angles, n_radii, min_radius, max_radius)
                a = np.pi * (max_radius**2 - min_radius**2)
                assert np.isclose(mesh.area(), a, atol=0, rtol=a / 1000)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e

        assert test_area_circular_disk_T6(1.0, 10.0, 120, 80)


if __name__ == "__main__":
    unittest.main()
