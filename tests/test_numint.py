# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.mesh.utils.numint import (
    Gauss_Legendre_Line_Grid,
    Gauss_Legendre_Tri_1,
    Gauss_Legendre_Tri_3a,
    Gauss_Legendre_Tri_3b,
    Gauss_Legendre_Tri_4,
    Gauss_Legendre_Tri_6,
    Gauss_Legendre_Quad_Grid,
    Gauss_Legendre_Quad_1,
    Gauss_Legendre_Quad_4,
    Gauss_Legendre_Quad_9,
    Gauss_Legendre_Tet_1,
    Gauss_Legendre_Tet_4,
    Gauss_Legendre_Tet_5,
    Gauss_Legendre_Tet_11,
    Gauss_Legendre_Hex_Grid,
    Gauss_Legendre_Wedge_3x2,
    Gauss_Legendre_Wedge_3x3,
)
from sigmaepsilon.mesh.data import PolyCell


class TestNumint(SigmaEpsilonTestCase):
    
    def test_numint_main(self):
        Gauss_Legendre_Line_Grid(2)
        
        Gauss_Legendre_Tri_1()
        Gauss_Legendre_Tri_1(natural=True)
        Gauss_Legendre_Tri_1(np.array([0.0, 0.0]))
        Gauss_Legendre_Tri_3a()
        Gauss_Legendre_Tri_3a(natural=True)
        Gauss_Legendre_Tri_3b()
        Gauss_Legendre_Tri_3b(natural=True)
        Gauss_Legendre_Tri_4()
        Gauss_Legendre_Tri_4(natural=True)
        Gauss_Legendre_Tri_6()
        Gauss_Legendre_Tri_6(natural=True)
        
        Gauss_Legendre_Quad_Grid(2, 2)
        Gauss_Legendre_Quad_1()
        Gauss_Legendre_Quad_4()
        Gauss_Legendre_Quad_9()
        
        Gauss_Legendre_Tet_1()
        Gauss_Legendre_Tet_1(natural=True)
        Gauss_Legendre_Tet_4()
        Gauss_Legendre_Tet_4(natural=True)
        Gauss_Legendre_Tet_5()
        Gauss_Legendre_Tet_5(natural=True)
        Gauss_Legendre_Tet_11()
        Gauss_Legendre_Tet_11(natural=True)
        
        Gauss_Legendre_Hex_Grid(2, 2, 2)
        Gauss_Legendre_Wedge_3x2()
        Gauss_Legendre_Wedge_3x3()
        
    def test_gauss_parser(self):
        pc = PolyCell()
        parser = pc._parse_gauss_data
        
        quadratures = {
            "1": Gauss_Legendre_Tri_1(),
            "2": "1",
            "3": "2",
            "4": Gauss_Legendre_Tri_1
        }
        
        parser(quadratures, "1")
        parser(quadratures, "2")
        parser(quadratures, "3")
        parser(quadratures, "4")
        
        for q1, q2 in zip(parser(quadratures, "1"), parser(quadratures, "2")):
            self.assertTrue(np.allclose(q1.pos, q2.pos))
            self.assertTrue(np.allclose(q1.weight, q2.weight))
            
        for q1, q2 in zip(parser(quadratures, "1"), parser(quadratures, "3")):
            self.assertTrue(np.allclose(q1.pos, q2.pos))
            self.assertTrue(np.allclose(q1.weight, q2.weight))
            
        for q1, q2 in zip(parser(quadratures, "1"), parser(quadratures, "4")):
            self.assertTrue(np.allclose(q1.pos, q2.pos))
            self.assertTrue(np.allclose(q1.weight, q2.weight))


if __name__ == "__main__":
    unittest.main()
