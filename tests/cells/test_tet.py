# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.mesh import PointData, TriMesh, CartesianFrame, triangulate
from sigmaepsilon.mesh.recipes import circular_disk
from sigmaepsilon.mesh.cells import T3, TET4, TET10


class TestTet(unittest.TestCase):
    def test_vol_TET4(self):
        def test_vol_TET4(Lx, Ly, Lz, nx, ny, nz):
            try:
                A = CartesianFrame(dim=3)
                coords, topo, _ = triangulate(size=(Lx, Ly), shape=(nx, ny))
                pd = PointData(coords=coords, frame=A)
                cd = T3(topo=topo)
                mesh2d = TriMesh(pd, cd)
                mesh3d = mesh2d.extrude(h=Lz, N=nz)
                assert np.isclose(mesh3d.volume(), Lx * Ly * Lz)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e

        self.assertTrue(test_vol_TET4(1.0, 1.0, 1.0, 2, 2, 2))

    def test_vol_cylinder_TET4(self):
        def test_vol_cylinder_TET4(
            min_radius, max_radius, height, n_angles, n_radii, n_z
        ):
            try:
                mesh2d = circular_disk(n_angles, n_radii, min_radius, max_radius)
                mesh3d = mesh2d.extrude(h=height, N=n_z)
                a = np.pi * (max_radius ** 2 - min_radius ** 2) * height
                assert np.isclose(mesh3d.volume(), a, atol=0, rtol=a / 1000)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e

        self.assertTrue(test_vol_cylinder_TET4(1.0, 10.0, 10.0, 120, 80, 5))

    def test_shp_TET4(self):
        pcoords = TET4.Geometry.master_coordinates()
        shpf, shpmf, dshpf = TET4.Geometry.generate_class_functions(
            return_symbolic=False
        )
        shpf(pcoords)
        shpmf(pcoords)
        dshpf(pcoords)

    def test_shp_TET10(self):
        pcoords = TET10.Geometry.master_coordinates()
        shpf, shpmf, dshpf = TET10.Geometry.generate_class_functions(
            return_symbolic=False
        )
        shpf(pcoords)
        shpmf(pcoords)
        dshpf(pcoords)


if __name__ == "__main__":
    unittest.main()
