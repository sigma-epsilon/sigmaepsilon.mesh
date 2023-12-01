# -*- coding: utf-8 -*-
import numpy as np
import unittest
from sympy import symbols

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.math import atleast2d
from sigmaepsilon.mesh import PointData, TriMesh, CartesianFrame, triangulate
from sigmaepsilon.mesh.recipes import circular_disk
from sigmaepsilon.mesh.cells import T3, TET4, TET10
from sigmaepsilon.mesh.utils.tet import nat_to_loc_tet


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
        
    def test_tet_surface(self):
        Lx, Ly, Lz, nx, ny, nz = 1.0, 1.0, 1.0, 2, 2, 2
        A = CartesianFrame(dim=3)
        coords, topo, _ = triangulate(size=(Lx, Ly), shape=(nx, ny))
        pd = PointData(coords=coords, frame=A)
        cd = T3(topo=topo)
        mesh2d = TriMesh(pd, cd)
        mesh3d = mesh2d.extrude(h=Lz, N=nz)
        
        surface: TriMesh = mesh3d.surface(mesh_class=TriMesh)
        normals = surface.normals()
        
        self.assertEqual(len(normals), len(surface.topology()))
        self.assertTrue(surface.is_2d_mesh())


class TestTET4(SigmaEpsilonTestCase):
    def test_TET4(self, N: int = 3):
        shp, dshp, shpf, shpmf, dshpf = TET4.Geometry.generate_class_functions(
            return_symbolic=True
        )
        r, s, t = symbols("r, s, t", real=True)

        for _ in range(N):
            A1, A2, A3 = np.random.rand(3)
            A4 = 1 - A1 - A2 - A3
            x_nat = np.array([A1, A2, A3, A4])
            x_loc = atleast2d(nat_to_loc_tet(x_nat))

            shpA = shpf(x_loc)
            shpB = TET4.Geometry.shape_function_values(x_loc)
            shp_sym = shp.subs({r: x_loc[0, 0], s: x_loc[0, 1], t: x_loc[0, 2]})
            self.assertTrue(np.allclose(shpA, shpB))
            self.assertTrue(
                np.allclose(shpA, np.array(shp_sym.tolist(), dtype=float).T)
            )

            dshpA = dshpf(x_loc)
            dshpB = TET4.Geometry.shape_function_derivatives(x_loc)
            dshp_sym = dshp.subs({r: x_loc[0, 0], s: x_loc[0, 1], t: x_loc[0, 2]})
            self.assertTrue(np.allclose(dshpA, dshpB))
            self.assertTrue(
                np.allclose(dshpA, np.array(dshp_sym.tolist(), dtype=float))
            )

            shpmfA = shpmf(x_loc)
            shpmfB = TET4.Geometry.shape_function_matrix(x_loc)
            self.assertTrue(np.allclose(shpmfA, shpmfB))

        nX = 2
        shpmf = TET4.Geometry.shape_function_matrix(x_loc, N=nX)
        self.assertEqual(shpmf.shape, (1, nX, 4 * nX))


if __name__ == "__main__":
    unittest.main()
