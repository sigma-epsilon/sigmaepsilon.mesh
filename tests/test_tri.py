import numpy as np
import unittest

from sigmaepsilon.mesh import PointData, TriMesh, CartesianFrame, triangulate
from sigmaepsilon.mesh.recipes import circular_disk
from sigmaepsilon.mesh.cells import T3, T6
from sigmaepsilon.mesh.utils.topology.tr import T3_to_T6


class TestTri(unittest.TestCase):
    def test_area_T3(self):
        def test_area_T3(Lx, Ly, nx, ny):
            try:
                A = CartesianFrame(dim=3)
                coords, topo, _ = triangulate(size=(Lx, Ly), shape=(nx, ny))
                pd = PointData(coords=coords, frame=A)
                cd = T3(topo=topo)
                mesh = TriMesh(pd, cd)
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
                coords, topo, _ = triangulate(size=(Lx, Ly), shape=(nx, ny))
                coords, topo = T3_to_T6(coords, topo)
                pd = PointData(coords=coords, frame=A)
                cd = T6(topo=topo)
                mesh = TriMesh(pd, cd)
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
                a = np.pi * (max_radius ** 2 - min_radius ** 2)
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
                a = np.pi * (max_radius ** 2 - min_radius ** 2)
                assert np.isclose(mesh.area(), a, atol=0, rtol=a / 1000)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e

        assert test_area_circular_disk_T6(1.0, 10.0, 120, 80)


if __name__ == "__main__":
    unittest.main()
