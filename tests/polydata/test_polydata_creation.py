import unittest

import numpy as np

from sigmaepsilon.mesh import PolyData, PointData, LineData
from sigmaepsilon.mesh.cells import L2
from sigmaepsilon.mesh.utils.space import frames_of_lines


class TestPolyDataCreation(unittest.TestCase):
    def _gen_L2_mesh_data(self):
        coords = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float
        )
        topology = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
        frames = frames_of_lines(coords, topology)
        return coords, topology, frames

    def test_multi_host_pre_fed(self):
        coordsA, topologyA, framesA = self._gen_L2_mesh_data()
        coordsB, *_ = self._gen_L2_mesh_data()
        coordsB[:, 2] = 1
        topologyB = topologyA + len(coordsA)
        framesB = frames_of_lines(coordsB, topologyB)

        pdA = PointData(coordsA)
        cdA = L2(topo=topologyA, frames=framesA)
        pdB = PointData(coordsB)
        cdB = L2(topo=topologyB, frames=framesB)
        mesh = PolyData(A=LineData(pdA, cdA), B=LineData(pdB, cdB))

        self.assertTrue(mesh["A"].source() is mesh["A"])
        self.assertTrue(mesh["B"].source() is mesh["B"])

        self.assertTrue(mesh.pd is None)
        self.assertTrue(mesh.cd is None)
        self.assertTrue(np.all(mesh["A"].pd.id == np.arange(5)))
        self.assertTrue(np.all(mesh["B"].pd.id == np.arange(5, 10)))
        self.assertTrue(np.all(mesh["A"].cd.db.id == np.arange(4)))
        self.assertTrue(np.all(mesh["B"].cd.db.id == np.arange(4, 8)))

    def test_multi_host_post_fed(self):
        coordsA, topologyA, framesA = self._gen_L2_mesh_data()
        coordsB, *_ = self._gen_L2_mesh_data()
        coordsB[:, 2] = 1
        topologyB = topologyA + len(coordsA)
        framesB = frames_of_lines(coordsB, topologyB)

        pdA = PointData(coordsA)
        cdA = L2(topo=topologyA, frames=framesA)
        pdB = PointData(coordsB)
        cdB = L2(topo=topologyB, frames=framesB)
        mesh = PolyData()
        mesh["A"] = LineData(pdA, cdA)
        mesh["B"] = LineData(pdB, cdB)

        self.assertTrue(mesh["A"].source() is mesh["A"])
        self.assertTrue(mesh["B"].source() is mesh["B"])

        self.assertTrue(mesh.pd is None)
        self.assertTrue(mesh.cd is None)
        self.assertTrue(np.all(mesh["A"].pd.id == np.arange(5)))
        self.assertTrue(np.all(mesh["B"].pd.id == np.arange(5, 10)))
        self.assertTrue(np.all(mesh["A"].cd.db.id == np.arange(4)))
        self.assertTrue(np.all(mesh["B"].cd.db.id == np.arange(4, 8)))

    def test_single_host_post_fed(self):
        coordsA, topologyA, framesA = self._gen_L2_mesh_data()
        coordsB, topologyB, _ = self._gen_L2_mesh_data()
        coordsB[:, 2] = 1
        framesB = frames_of_lines(coordsB, topologyB)
        coords = np.vstack([coordsA, coordsB])
        topologyB += len(coordsA)

        pd = PointData(coords)
        cdA = L2(topo=topologyA, frames=framesA)
        cdB = L2(topo=topologyB, frames=framesB)
        mesh = PolyData(pd)
        mesh["A"] = LineData(cdA)
        mesh["B"] = LineData(cdB)

        self.assertTrue(mesh["A"].source() is mesh)
        self.assertTrue(mesh["B"].source() is mesh)

        self.assertTrue(mesh.pd is not None)
        self.assertTrue(mesh.cd is None)
        self.assertTrue(mesh["A"].pd is None)
        self.assertTrue(mesh["B"].pd is None)
        self.assertTrue(np.all(mesh.pd.id == np.arange(10)))
        self.assertTrue(np.all(mesh["A"].cd.db.id == np.arange(4)))
        self.assertTrue(np.all(mesh["B"].cd.db.id == np.arange(4, 8)))

    def test_single_host_pre_fed(self):
        coordsA, topologyA, framesA = self._gen_L2_mesh_data()
        coordsB, topologyB, _ = self._gen_L2_mesh_data()
        coordsB[:, 2] = 1
        framesB = frames_of_lines(coordsB, topologyB)
        coords = np.vstack([coordsA, coordsB])
        topologyB += len(coordsA)

        pd = PointData(coords)
        cdA = L2(topo=topologyA, frames=framesA)
        cdB = L2(topo=topologyB, frames=framesB)
        mesh = PolyData(pd, A=LineData(cdA), B=LineData(cdB))

        self.assertTrue(mesh["A"].source() is mesh)
        self.assertTrue(mesh["B"].source() is mesh)

        self.assertTrue(mesh.pd is not None)
        self.assertTrue(mesh.cd is None)
        self.assertTrue(mesh["A"].pd is None)
        self.assertTrue(mesh["B"].pd is None)
        self.assertTrue(np.all(mesh.pd.id == np.arange(10)))
        self.assertTrue(np.all(mesh["A"].cd.db.id == np.arange(4)))
        self.assertTrue(np.all(mesh["B"].cd.db.id == np.arange(4, 8)))
