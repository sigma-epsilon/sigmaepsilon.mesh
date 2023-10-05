# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sigmaepsilon.mesh import PointData, PolyData, TriMesh, triangulate
from sigmaepsilon.mesh.recipes import circular_disk
from sigmaepsilon.mesh.voxelize import voxelize_cylinder
from sigmaepsilon.mesh.cells import H8, TET4, W6, T3
from sigmaepsilon.mesh.utils.topology import detach_mesh_bulk
from sigmaepsilon.mesh.extrude import extrude_T3_TET4, extrude_T3_W6
from sigmaepsilon.mesh.space import StandardFrame


class TestMeshing(unittest.TestCase):
    def test_trimesh(self):
        triangulate(size=(800, 600), shape=(10, 10))
        circular_disk(120, 60, 5, 25)

    def test_voxelize(self):
        d, h, a, b = 100.0, 0.8, 1.5, 0.5
        coords, topo = voxelize_cylinder(radius=[b / 2, a / 2], height=h, size=h / 20)
        A = StandardFrame(dim=3)
        pd = PointData(coords=coords, frame=A)
        cd = H8(topo=topo, frames=A)
        PolyData(pd, cd)

    def test_extrude_T3_TET4(self):
        n_angles = 120
        n_radii = 60
        min_radius = 5
        max_radius = 25
        h = 20
        zres = 20
        mesh = circular_disk(n_angles, n_radii, min_radius, max_radius)
        points = mesh.coords()
        triangles = mesh.topology().to_numpy()
        points, triangles = detach_mesh_bulk(points, triangles)
        coords, topo = extrude_T3_TET4(points, triangles, h, zres)

        vol_exact = np.pi * (max_radius ** 2 - min_radius ** 2) * h

        A = StandardFrame(dim=3)
        pd = PointData(coords=coords, frame=A)
        cd = TET4(topo=topo, frames=A)
        tetmesh = PolyData(pd, cd)
        self.assertTrue(np.isclose(vol_exact, tetmesh.volume(), atol=vol_exact / 1000))

        coords, topo, _ = triangulate(size=(800, 600), shape=(10, 10))
        pd = PointData(coords=coords, frame=A)
        cd = T3(topo=topo)
        trimesh = TriMesh(pd, cd)
        trimesh.area()
        tetmesh = trimesh.extrude(h=300, N=5)
        vol_exact = 800 * 600 * 300
        self.assertTrue(np.isclose(vol_exact, tetmesh.volume(), atol=vol_exact / 1000))

    def test_extrude_T3_W6(self):
        n_angles = 120
        n_radii = 60
        min_radius = 5
        max_radius = 25
        h = 20
        zres = 20
        mesh = circular_disk(n_angles, n_radii, min_radius, max_radius)
        points = mesh.coords()
        triangles = mesh.topology().to_numpy()
        points, triangles = detach_mesh_bulk(points, triangles)
        coords, topo = extrude_T3_W6(points, triangles, h, zres)

        vol_exact = np.pi * (max_radius ** 2 - min_radius ** 2) * h

        A = StandardFrame(dim=3)
        pd = PointData(coords=coords, frame=A)
        cd = W6(topo=topo, frames=A)
        mesh = PolyData(pd, cd)
        self.assertTrue(np.isclose(vol_exact, mesh.volume(), atol=vol_exact / 1000))

    def test_tet(self):
        A = StandardFrame(dim=3)
        coords, topo, _ = triangulate(size=(800, 600), shape=(10, 10))
        pd = PointData(coords=coords, frame=A)
        cd = T3(topo=topo)
        trimesh = TriMesh(pd, cd)
        tetmesh = trimesh.extrude(h=300, N=5)
        self.assertTrue(np.isclose(144000000.0, tetmesh.volume()))


if __name__ == "__main__":
    unittest.main()
