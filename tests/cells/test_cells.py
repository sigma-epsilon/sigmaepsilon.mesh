import numpy as np
import unittest

from sigmaepsilon.mesh.cells import H8, TET10
from sigmaepsilon.mesh import grid, PolyData, CartesianFrame, PointData


class TestGeneratedExpressions(unittest.TestCase):
    def test_generated_H8(self):
        pcoords = H8.Geometry.master_coordinates()
        shpf = H8.Geometry.shape_function_values
        shpmf = H8.Geometry.shape_function_matrix
        dshpf = H8.Geometry.shape_function_derivatives
        _shpf, _shpmf, _dshpf = H8.Geometry.generate_class_functions(
            return_symbolic=False
        )
        self.assertTrue(np.all(np.isclose(_shpf(pcoords), shpf(pcoords))))
        self.assertTrue(np.all(np.isclose(_dshpf(pcoords), dshpf(pcoords))))
        self.assertTrue(np.all(np.isclose(_shpmf(pcoords), shpmf(pcoords))))

    def test_generated_TET10(self):
        pcoords = TET10.Geometry.master_coordinates()
        shpf = TET10.Geometry.shape_function_values
        shpmf = TET10.Geometry.shape_function_matrix
        dshpf = TET10.Geometry.shape_function_derivatives
        _shpf, _shpmf, _dshpf = TET10.Geometry.generate_class_functions(
            return_symbolic=False
        )
        self.assertTrue(np.all(np.isclose(_shpf(pcoords), shpf(pcoords))))
        self.assertTrue(np.all(np.isclose(_dshpf(pcoords), dshpf(pcoords))))
        self.assertTrue(np.all(np.isclose(_shpmf(pcoords), shpmf(pcoords))))


class TestPIP(unittest.TestCase):
    def test_pip_H8(self):
        Lx, Ly, Lz = 800, 600, 100
        nx, ny, nz = 2, 2, 2
        xbins = np.linspace(0, Lx, nx + 1)
        ybins = np.linspace(0, Ly, ny + 1)
        zbins = np.linspace(0, Lz, nz + 1)
        bins = xbins, ybins, zbins
        coords, topo = grid(bins=bins, eshape="H8")
        frame = CartesianFrame(dim=3)
        pd = PointData(coords=coords, frame=frame)
        cd = H8(topo=topo, frames=frame)
        mesh = PolyData(pd, cd)
        self.assertTrue(mesh.cd.pip(coords[0, :], tol=1e-12))
        self.assertTrue(mesh.cd.pip(coords[-1, :], tol=1e-12))
        self.assertFalse(mesh.cd.pip(coords[0, :] - 1, tol=1e-12))
        self.assertFalse(mesh.cd.pip(coords[-1, :] + 1, tol=1e-12))


if __name__ == "__main__":
    unittest.main()
