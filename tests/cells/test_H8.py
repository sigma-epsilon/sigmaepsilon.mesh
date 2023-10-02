# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame
from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.grid import Grid
from sigmaepsilon.mesh.cells import H8


class TestHex(unittest.TestCase):
    def test_H8(self):
        def test_H8_volume(Lx, Ly, Lz, nx, ny, nz):
            try:
                mesh = Grid(size=(Lx, Ly, Lz), shape=(nx, ny, nz), eshape="H8")
                assert np.isclose(mesh.volume(), Lx * Ly * Lz)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e

        self.assertTrue(test_H8_volume(1.0, 1.0, 1.0, 2, 2, 2))

        pcoords = H8.Geometry.master_coordinates()
        shpf = H8.Geometry.shape_function_values
        shpmf = H8.Geometry.shape_function_matrix
        dshpf = H8.Geometry.shape_function_derivatives
        shpfH8, shpmfH8, dshpfH8 = H8.Geometry.generate_class_functions(
            return_symbolic=False
        )

        self.assertTrue(np.all(np.isclose(shpfH8(pcoords), shpf(pcoords))))
        self.assertTrue(np.all(np.isclose(dshpfH8(pcoords), dshpf(pcoords))))
        self.assertTrue(np.all(np.isclose(shpmfH8(pcoords), shpmf(pcoords))))

    def test_H8_shape_function_derivatives(self):
        Lx, Ly, Lz = 800, 600, 100
        nx, ny, nz = 8, 6, 2
        xbins = np.linspace(0, Lx, nx + 1)
        ybins = np.linspace(0, Ly, ny + 1)
        zbins = np.linspace(0, Lz, nz + 1)
        bins = xbins, ybins, zbins
        coords, topo = grid(bins=bins, eshape="H8")
        frame = CartesianFrame(dim=3)

        pd = PointData(coords=coords)
        cd = H8(topo=topo, frames=frame)
        mesh = PolyData(pd, cd, frame=frame)

        jac = cd.jacobian_matrix()
        pcoords = H8.Geometry.master_coordinates()
        gdshp = H8.Geometry.shape_function_derivatives(pcoords[:2], jac=jac)

        self.assertTrue(gdshp.shape, (topo.shape[0], 2, topo.shape[1], 3))

        del gdshp, mesh


if __name__ == "__main__":
    unittest.main()
