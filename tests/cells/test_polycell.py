import unittest, doctest

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
import sigmaepsilon.mesh
from sigmaepsilon.mesh import PolyData, PointData, LineData
from sigmaepsilon.mesh.space import CartesianFrame
from sigmaepsilon.mesh.cells import H8, TET4, L2
from sigmaepsilon.mesh.utils.topology import H8_to_TET4, H8_to_L2
from sigmaepsilon.mesh.utils.space import frames_of_lines
from sigmaepsilon.mesh.grid import grid as _grid


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(sigmaepsilon.mesh.cells))
    return tests


class TestPolyCell(SigmaEpsilonTestCase):
    def test_polycell(self):
        size = 10, 10, 5
        shape = 2, 2, 2
        coords, topo = _grid(size=size, shape=shape, eshape="H8")
        pd = PointData(coords=coords)
        cd = H8(topo=topo)
        grid = PolyData(pd, cd)
        grid.centralize()

        coords = grid.coords()
        topo = grid.topology().to_numpy()
        centers = grid.centers()

        b_left = centers[:, 0] < 0
        b_right = centers[:, 0] >= 0
        b_front = centers[:, 1] >= 0
        b_back = centers[:, 1] < 0
        iTET4 = np.where(b_left)[0]
        iH8 = np.where(b_right & b_back)[0]
        iL2 = np.where(b_right & b_front)[0]
        _, tTET4 = H8_to_TET4(coords, topo[iTET4])
        _, tL2 = H8_to_L2(coords, topo[iL2])
        tH8 = topo[iH8]

        # crate supporting pointcloud
        frame = CartesianFrame(dim=3)
        pd = PointData(coords=coords, frame=frame)
        mesh = PolyData(pd, frame=frame)

        # tetrahedra
        cdTET4 = TET4(topo=tTET4)
        mesh["tetra"] = PolyData(cdTET4, frame=frame)
        mesh["tetra"].config["A", "color"] = "green"

        # hexahedra
        cdH8 = H8(topo=tH8)
        mesh["hex"] = PolyData(cdH8, frame=frame)
        mesh["hex"].config["A", "color"] = "blue"

        # lines
        cdL2 = L2(topo=tL2)
        mesh["line"] = LineData(cdL2, frame=frame)
        mesh["line"].config["A", "color"] = "red"
        mesh["line"].config["A", "line_width"] = 3
        mesh["line"].config["A", "render_lines_as_tubes"] = True

        # finalize the mesh and lock the layout
        mesh.to_standard_form()
        mesh.lock(create_mappers=True)

        cdL2.db = cdL2.db
        cdL2._dbkey_id_

        cdL2.flip().flip()
        cdH8.flip().flip()
        cdTET4.flip().flip()

        cdL2._get_points_and_range()
        cdH8._get_points_and_range()
        cdTET4._get_points_and_range()
        
        cdL2.points_of_cells()
        cdL2.points_of_cells(points=[-1.0, 1.0], rng=[-1, 1])
        cdH8.points_of_cells()
        cdTET4.points_of_cells()

        cdL2.local_coordinates()
        cdL2.local_coordinates(target=frame)
        cdH8.local_coordinates()
        cdH8.local_coordinates(target=frame)
        cdTET4.local_coordinates()
        cdTET4.local_coordinates(target=frame)

        cdL2.rewire()
        cdL2.rewire(imap=pd.id)
        cdH8.rewire()
        cdH8.rewire(imap=pd.id)
        cdTET4.rewire()
        cdTET4.rewire(imap=pd.id)

        self.assertTrue(np.allclose(cdL2.centers(), cdL2.centers(target=frame)))
        self.assertTrue(np.allclose(cdH8.centers(), cdH8.centers(target=frame)))
        self.assertTrue(np.allclose(cdTET4.centers(), cdTET4.centers(target=frame)))

        self.assertEqual(cdL2.root(), mesh)
        self.assertEqual(cdH8.root(), mesh)
        self.assertEqual(cdTET4.root(), mesh)

        self.assertTrue(np.allclose(cdL2.lengths(), cdL2.measures()))
        self.assertTrue(np.allclose(cdH8.volumes(), cdH8.measures()))
        self.assertTrue(np.allclose(cdTET4.volumes(), cdTET4.measures()))

        self.assertTrue(np.isclose(cdL2.length(), cdL2.measure()))
        self.assertTrue(np.isclose(cdH8.volume(), cdH8.measure()))
        self.assertTrue(np.isclose(cdTET4.volume(), cdTET4.measure()))

        self.assertEqual(cdL2.container, mesh["line"])
        self.assertEqual(cdH8.container, mesh["hex"])
        self.assertEqual(cdTET4.container, mesh["tetra"])

        self.assertEqual(len(cdL2.jacobian()), len(cdL2))
        self.assertEqual(len(cdH8.jacobian()), len(cdH8))
        self.assertEqual(len(cdTET4.jacobian()), len(cdTET4))

        self.assertEqual(len(cdL2), len(cdL2.frames))

        self.assertRaises(TypeError, setattr, cdL2, "pointdata", 1)
        self.assertRaises(NotImplementedError, cdL2.to_triangles)
        self.assertRaises(NotImplementedError, cdL2.thickness)
        self.assertRaises(NotImplementedError, cdH8.thickness)


if __name__ == "__main__":
    unittest.main()
