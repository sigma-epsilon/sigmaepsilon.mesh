import unittest, doctest

from matplotlib.tri import Triangulation as triobj_mpl
from scipy.spatial import Delaunay as triobj_scipy

import sigmaepsilon.mesh
from sigmaepsilon.mesh.triang import (
    triangulate,
    triobj_to_mpl,
    get_triobj_data,
    _is_triobj,
)


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(sigmaepsilon.mesh.triang))
    return tests


class TestTriangulate(unittest.TestCase):
    def test_triangulate_main(self):
        coords, topo, triobj = triangulate(size=(800, 600), shape=(10, 10))

        coords, topo, triobj = triangulate(triobj)

        points, edges, triangles, edgeIDs, triobj = triangulate(
            triobj, return_lines=True
        )

    def test_triangulate_mpl(self):
        coords, topo, triobj = triangulate(
            size=(800, 600), shape=(10, 10), backend="mpl"
        )
        self.assertTrue(_is_triobj(triobj))
        coords, topo, triobj = triangulate(triobj)
        coords, topo = get_triobj_data(triobj)
        self.assertTrue(isinstance(triobj, triobj_mpl))
        triobj = triobj_to_mpl(triobj)
        self.assertTrue(isinstance(triobj, triobj_mpl))

    def test_triangulate_scipy(self):
        coords, topo, triobj = triangulate(
            size=(800, 600), shape=(10, 10), backend="scipy"
        )
        self.assertTrue(_is_triobj(triobj))
        coords, topo, triobj = triangulate(triobj)
        coords, topo = get_triobj_data(triobj)
        self.assertTrue(isinstance(triobj, triobj_scipy))
        triobj = triobj_to_mpl(triobj)
        self.assertTrue(isinstance(triobj, triobj_mpl))

    def test_triangulate_pv(self):
        coords, topo, triobj = triangulate(
            size=(800, 600), shape=(10, 10), backend="pv"
        )
        self.assertTrue(_is_triobj(triobj))
        coords, topo, triobj = triangulate(triobj)
        coords, topo = get_triobj_data(triobj)
        triobj = triobj_to_mpl(triobj)
        self.assertTrue(isinstance(triobj, triobj_mpl))

    def test__is_triobj(self):
        self.assertFalse(_is_triobj(None))
        self.assertFalse(_is_triobj(0))
        self.assertFalse(_is_triobj("0"))

    def test_triangulate_random(self):
        coords, topo, triobj = triangulate(
            size=(800, 600), shape=10, backend="mpl", random=True
        )
        self.assertTrue(_is_triobj(triobj))

    def test_triangulate_origo(self):
        coords, topo, triobj = triangulate(
            size=(800, 600), shape=10, origo=(0, 0), random=True
        )
        self.assertTrue(_is_triobj(triobj))

    def test_triangulate_shape(self):
        coords, topo, triobj = triangulate(size=(800, 600), shape=10, origo=(0, 0))
        self.assertTrue(_is_triobj(triobj))

        coords, topo, triobj = triangulate(size=(800, 600), shape=None, origo=(0, 0))
        self.assertTrue(_is_triobj(triobj))


if __name__ == "__main__":
    unittest.main()
