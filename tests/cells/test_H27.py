# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame
from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.cells import H27
from sigmaepsilon.mesh.utils.cells.h27 import (
    monoms_H27,
    shp_H27,
    shp_H27_multi,
    shape_function_matrix_H27,
    shape_function_matrix_H27_multi,
    dshp_H27,
    dshp_H27_multi,
)


class TestH27(unittest.TestCase):
    def test_H27_utils(self):
        shp = monoms_H27(np.array([0.0, 0.0, 0.0])).shape
        self.assertEqual(shp, (27,))

        shp = monoms_H27(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])).shape
        self.assertEqual(shp, (2, 27))

        shp = monoms_H27(np.zeros((2, 27, 3), dtype=float)).shape
        self.assertEqual(shp, (2, 27, 27))

        shp = shp_H27((0.0, 0.0, 0.0)).shape
        self.assertEqual(shp, (27,))

        shp = shp_H27_multi(np.zeros((2, 3), dtype=float)).shape
        self.assertEqual(shp, (2, 27))

        shp = shape_function_matrix_H27(np.array([0.0, 0.0, 0.0])).shape
        self.assertEqual(shp, (3, 3 * 27))

        shp = shape_function_matrix_H27_multi(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).shape
        self.assertEqual(shp, (2, 3, 3 * 27))

        shp = dshp_H27(np.array([0.0, 0.0, 0.0])).shape
        self.assertEqual(shp, (27, 3))

        shp = dshp_H27_multi(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])).shape
        self.assertEqual(shp, (2, 27, 3))

    def test_H27(self):
        def test_H27_volume(Lx, Ly, Lz, nx, ny, nz):
            try:
                coords, topo = grid(size=(Lx, Ly, Lz), shape=(nx, ny, nz), eshape="H27")
                pd = PointData(coords=coords)
                cd = H27(topo=topo)
                mesh = PolyData(pd, cd)
                assert np.isclose(mesh.volume(), Lx * Ly * Lz)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e

        self.assertTrue(test_H27_volume(1.0, 1.0, 1.0, 2, 2, 2))

        pcoords = H27.Geometry.master_coordinates()
        shpf = H27.Geometry.shape_function_values
        shpmf = H27.Geometry.shape_function_matrix
        dshpf = H27.Geometry.shape_function_derivatives
        shpfH27, shpmfH27, dshpfH27 = H27.Geometry.generate_class_functions(
            return_symbolic=False
        )

        self.assertTrue(np.all(np.isclose(shpfH27(pcoords), shpf(pcoords))))
        self.assertTrue(np.all(np.isclose(dshpfH27(pcoords), dshpf(pcoords))))
        self.assertTrue(np.all(np.isclose(shpmfH27(pcoords), shpmf(pcoords))))

    def test_H27_shape_function_derivatives(self):
        Lx, Ly, Lz = 1, 1, 1
        nx, ny, nz = 2, 3, 4
        xbins = np.linspace(0, Lx, nx + 1)
        ybins = np.linspace(0, Ly, ny + 1)
        zbins = np.linspace(0, Lz, nz + 1)
        bins = xbins, ybins, zbins
        coords, topo = grid(bins=bins, eshape="H27")
        frame = CartesianFrame(dim=3)

        pd = PointData(coords=coords)
        cd = H27(topo=topo, frames=frame)
        mesh = PolyData(pd, cd, frame=frame)

        jac = cd.jacobian_matrix()
        pcoords = H27.Geometry.master_coordinates()
        gdshp = H27.Geometry.shape_function_derivatives(pcoords[:2], jac=jac)

        self.assertTrue(gdshp.shape, (topo.shape[0], 2, topo.shape[1], 3))

        del gdshp, mesh


if __name__ == "__main__":
    unittest.main()
