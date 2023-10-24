# -*- coding: utf-8 -*-
import numpy as np
import unittest
from sympy import symbols

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.math import atleast2d
from sigmaepsilon.mesh.cells import T3
from sigmaepsilon.mesh.utils.tri import nat_to_loc_tri


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


if __name__ == "__main__":
    unittest.main()
