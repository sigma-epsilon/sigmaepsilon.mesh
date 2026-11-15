import unittest

import numpy as np

from sigmaepsilon.mesh.cells import (
    L2,
    L3,
    T3,
    Q4,
    Q8,
    Q9,
    T6,
    H8,
    H27,
    TET4,
    TET10,
    W6,
    W18,
)

CELLS = [L2, L3, T3, Q4, Q8, Q9, T6, H8, H27, TET4, TET10, W6, W18]


def _points_inside(cell, n: int = 7, seed: int = 0) -> np.ndarray:
    """Random convex combinations of the nodes of the master cell."""
    G = cell.Geometry
    mc = G.master_coordinates()
    nn = G.number_of_nodes
    w = np.random.default_rng(seed).random((n, nn))
    w /= w.sum(axis=1, keepdims=True)
    pts = w @ mc.reshape(nn, -1)
    return pts.ravel() if mc.ndim == 1 else pts


class TestShapeFunctions(unittest.TestCase):
    def test_identity_at_nodes(self):
        for cell in CELLS:
            with self.subTest(cell=cell.__name__):
                G = cell.Geometry
                nn = G.number_of_nodes
                N = np.asarray(G.shape_function_values(G.master_coordinates()))
                self.assertTrue(np.allclose(N.reshape(nn, nn), np.eye(nn)))

    def test_partition_of_unity(self):
        for cell in CELLS:
            with self.subTest(cell=cell.__name__):
                G = cell.Geometry
                pts = _points_inside(cell)
                N = np.asarray(G.shape_function_values(pts)).reshape(len(pts), -1)
                self.assertTrue(np.allclose(N.sum(axis=1), 1.0))

    def test_derivatives_sum_to_zero(self):
        for cell in CELLS:
            with self.subTest(cell=cell.__name__):
                G = cell.Geometry
                pts = _points_inside(cell)
                dN = np.asarray(G.shape_function_derivatives(pts))
                dN = dN.reshape(len(pts), G.number_of_nodes, -1)
                self.assertTrue(np.allclose(dN.sum(axis=1), 0.0))

    def test_shape_function_matrix_TET4(self):
        G = TET4.Geometry
        pts = _points_inside(TET4)
        N = G.shape_function_values(pts)
        NM = G.shape_function_matrix(pts, N=3)
        self.assertEqual(NM.shape, (len(pts), 3, 12))
        for i in range(4):
            block = NM[:, :, 3 * i : 3 * (i + 1)]
            expected = N[:, i, None, None] * np.eye(3)
            self.assertTrue(np.allclose(block, expected))

    def test_generated_TET4(self):
        G = TET4.Geometry
        pts = np.vstack([G.master_coordinates(), _points_inside(TET4)])
        _shpf, _shpmf, _dshpf = G.generate_class_functions(
            return_symbolic=False, update=False
        )
        self.assertTrue(np.allclose(_shpf(pts), G.shape_function_values(pts)))
        self.assertTrue(np.allclose(_shpmf(pts, 3), G.shape_function_matrix(pts, N=3)))
        self.assertTrue(np.allclose(_dshpf(pts), G.shape_function_derivatives(pts)))


if __name__ == "__main__":
    unittest.main()
