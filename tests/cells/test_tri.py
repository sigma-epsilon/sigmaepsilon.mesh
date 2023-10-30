# -*- coding: utf-8 -*-
import numpy as np
import unittest
from sympy import symbols

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.math import atleast2d
from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame
from sigmaepsilon.mesh.cells import T3, T6
from sigmaepsilon.mesh.utils.tri import (
    nat_to_loc_tri,
    loc_to_nat_tri,
    loc_to_glob_tri,
    nat_to_glob_tri,
    glob_to_loc_tri,
    glob_to_nat_tri,
    lcoords_tri,
    ncenter_tri,
    lcenter_tri,
    center_tri_2d,
    area_tri,
)


class TestT3(SigmaEpsilonTestCase):
    def test_T3(self, N: int = 3):
        shp, dshp, shpf, shpmf, dshpf = T3.Geometry.generate_class_functions(
            return_symbolic=True
        )
        r, s = symbols("r, s", real=True)

        for _ in range(N):
            A1, A2 = np.random.rand(2)
            A3 = 1 - A1 - A2
            x_nat = np.array([A1, A2, A3])
            x_loc = atleast2d(nat_to_loc_tri(x_nat))

            shpA = shpf(x_loc)
            shpB = T3.Geometry.shape_function_values(x_loc)
            shp_sym = shp.subs({r: x_loc[0, 0], s: x_loc[0, 1]})
            self.assertTrue(np.allclose(shpA, shpB))
            self.assertTrue(
                np.allclose(shpA, np.array(shp_sym.tolist(), dtype=float).T)
            )

            dshpA = dshpf(x_loc)
            dshpB = T3.Geometry.shape_function_derivatives(x_loc)
            dshp_sym = dshp.subs({r: x_loc[0, 0], s: x_loc[0, 1]})
            self.assertTrue(np.allclose(dshpA, dshpB))
            self.assertTrue(
                np.allclose(dshpA, np.array(dshp_sym.tolist(), dtype=float))
            )

            shpmfA = shpmf(x_loc)
            shpmfB = T3.Geometry.shape_function_matrix(x_loc)
            self.assertTrue(np.allclose(shpmfA, shpmfB))

        nX = 2
        shpmf = T3.Geometry.shape_function_matrix(x_loc, N=nX)
        self.assertEqual(shpmf.shape, (1, nX, 3 * nX))
        
    def test_T6(self, N: int = 3):
        shp, dshp, shpf, shpmf, dshpf = T6.Geometry.generate_class_functions(
            return_symbolic=True
        )
        r, s = symbols("r, s", real=True)

        for _ in range(N):
            A1, A2 = np.random.rand(2)
            A3 = 1 - A1 - A2
            x_nat = np.array([A1, A2, A3])
            x_loc = atleast2d(nat_to_loc_tri(x_nat))

            shpA = shpf(x_loc)
            shpB = T6.Geometry.shape_function_values(x_loc)
            shp_sym = shp.subs({r: x_loc[0, 0], s: x_loc[0, 1]})
            self.assertTrue(np.allclose(shpA, shpB))
            self.assertTrue(
                np.allclose(shpA, np.array(shp_sym.tolist(), dtype=float).T)
            )

            dshpA = dshpf(x_loc)
            dshpB = T6.Geometry.shape_function_derivatives(x_loc)
            dshp_sym = dshp.subs({r: x_loc[0, 0], s: x_loc[0, 1]})
            self.assertTrue(np.allclose(dshpA, dshpB))
            self.assertTrue(
                np.allclose(dshpA, np.array(dshp_sym.tolist(), dtype=float))
            )

            shpmfA = shpmf(x_loc)
            shpmfB = T6.Geometry.shape_function_matrix(x_loc)
            self.assertTrue(np.allclose(shpmfA, shpmfB))

        nX = 2
        shpmf = T6.Geometry.shape_function_matrix(x_loc, N=nX)
        self.assertEqual(shpmf.shape, (1, nX, 6 * nX))


class TestTriutils(SigmaEpsilonTestCase):
    def test_triutils(self):
        frame = CartesianFrame()
        coords = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
        topo = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        pd = PointData(coords=coords, frame=frame)
        cd = T3(topo=topo, frames=frame)
        _ = PolyData(pd, cd)
        ec = cd.local_coordinates()
        nE, nNE = topo.shape
        
        self.assertTrue(np.allclose(nat_to_loc_tri(ncenter_tri()), lcenter_tri()))
        self.assertTrue(np.allclose(loc_to_nat_tri(lcenter_tri()), ncenter_tri()))
        
        x_tri_loc = lcoords_tri()
        x_tri_nat = np.eye(3).astype(float)
        c_tri_loc = lcenter_tri()
        
        for iNE in range(nNE):
            x_nat = loc_to_nat_tri(x_tri_loc[iNE])
            self.assertTrue(np.allclose(x_nat, x_tri_nat[iNE]))
        
        for iE in range(nE):
            x_glob = loc_to_glob_tri(c_tri_loc, ec[iE])
            self.assertTrue(np.allclose(center_tri_2d(ec[iE]), x_glob))
        
        for iE in range(nE):
            self.assertAlmostEqual(area_tri(ec[iE]), 2.0, delta=1e-5)
        
        for iE in range(nE):
            for iNE in range(nNE):
                x_glob = loc_to_glob_tri(x_tri_loc[iNE], ec[iE])
                self.assertTrue(np.allclose(x_glob, ec[iE, iNE]))
                
        for iE in range(nE):
            for iNE in range(nNE):
                x_glob = nat_to_glob_tri(x_tri_nat[iNE], ec[iE])
                self.assertTrue(np.allclose(x_glob, ec[iE, iNE]))
                
        for iE in range(nE):
            for iNE in range(nNE):
                x_loc = glob_to_loc_tri(ec[iE, iNE], ec[iE])
                self.assertTrue(np.allclose(x_loc, x_tri_loc[iNE]))

        for iE in range(nE):
            for iNE in range(nNE):
                x_nat = glob_to_nat_tri(ec[iE, iNE], ec[iE])
                self.assertTrue(np.allclose(x_nat, x_tri_nat[iNE]))


if __name__ == "__main__":
    unittest.main()
