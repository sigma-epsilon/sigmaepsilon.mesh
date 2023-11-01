import unittest

import numpy as np
from sympy import symbols

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.mesh.data import PolyCell
from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame, grid
from sigmaepsilon.mesh.helpers import vtk_to_celltype


class TestPolyCell1d(SigmaEpsilonTestCase):
    def _test_polycell_1d_single_evaluation(self, CellData: PolyCell[PolyData, PointData]):
        """
        Tests the cells for a single point of evaluation.
        """
        nNE = CellData.Geometry.number_of_nodes
        nD = CellData.Geometry.number_of_spatial_dimensions
        self.assertEqual(nD, 1)

        gridparams = {
            "size": (1,),
            "shape": (2,),
            "eshape": (nNE,),
        }
        coords, topo = grid(**gridparams)
        frame = CartesianFrame(dim=3)

        pd = PointData(coords=coords, frame=frame)
        cd: PolyCell[PolyData, PointData] = CellData(topo=topo, frames=frame)

        _ = PolyData(pd, cd)

        self.assertTrue(np.isclose(cd.length(), 1.0))

        shp, dshp, shpf, shpmf, dshpf = CellData.Geometry.generate_class_functions(
            return_symbolic=True
        )
        r = symbols("r", real=True)

        x_loc = np.random.rand(1)

        shpA = shpf(x_loc)
        shpB = CellData.Geometry.shape_function_values(x_loc)
        shp_sym = shp.subs({r: x_loc[0]})
        self.assertTrue(np.allclose(shpA, shpB))
        self.assertTrue(np.allclose(shpA, np.array(shp_sym.tolist(), dtype=float).T))

        dshpA = dshpf(x_loc)
        dshpB = CellData.Geometry.shape_function_derivatives(x_loc)
        dshp_sym = dshp.subs({r: x_loc[0]})
        self.assertTrue(np.allclose(dshpA, dshpB))
        self.assertTrue(np.allclose(dshpA, np.array(dshp_sym.tolist(), dtype=float)))

        shpmfA = shpmf(x_loc)
        shpmfB = CellData.Geometry.shape_function_matrix(x_loc)
        self.assertTrue(np.allclose(shpmfA, shpmfB))

        mc = CellData.Geometry.master_coordinates()
        shp = CellData.Geometry.shape_function_values(mc)
        self.assertTrue(np.allclose(np.diag(shp), np.ones((nNE))))

        nX = 2
        shpmf = CellData.Geometry.shape_function_matrix(x_loc, N=nX)
        self.assertEqual(shpmf.shape, (1, nX, nNE * nX))

    def _test_master_cell_1d(self, CellData: PolyCell[PolyData, PointData]):
        nNE = CellData.Geometry.number_of_nodes
        nD = CellData.Geometry.number_of_spatial_dimensions
        self.assertEqual(nD, 1)

        frame = CartesianFrame()
        coords = np.zeros((nNE, 3), dtype=float)
        coords[:, 0] = CellData.Geometry.master_coordinates()
        topo = np.array([list(range(nNE))], dtype=int)
        pd = PointData(coords=coords, frame=frame)
        cd: PolyCell[PolyData, PointData] = CellData(topo=topo, frames=frame)
        _ = PolyData(pd, cd)
        self.assertTrue(np.isclose(cd.length(), 2.0))
        self.assertTrue(np.allclose(cd.jacobian(), np.ones((1, nNE))))

    def test_cells_1d(self):
        cells = filter(
            lambda c: c.Geometry.number_of_spatial_dimensions == 1,
            vtk_to_celltype.values(),
        )
        for cell in cells:
            self._test_polycell_1d_single_evaluation(cell)
            self._test_master_cell_1d(cell)


if __name__ == "__main__":
    unittest.main()
