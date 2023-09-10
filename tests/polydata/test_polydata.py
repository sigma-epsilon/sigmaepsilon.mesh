import os
import unittest

import numpy as np
import pyvista as pv
from pyvista import examples
import meshio

from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame
from sigmaepsilon.mesh.trimesh import TriMesh
from sigmaepsilon.mesh.grid import Grid
from sigmaepsilon.mesh.space import StandardFrame
from sigmaepsilon.mesh.cells import Q4, H8
from sigmaepsilon.mesh.grid import grid


class TestPolyData(unittest.TestCase):
    def test_compound_mesh(self):
        A = StandardFrame(dim=3)
        tri = TriMesh(size=(100, 100), shape=(4, 4), frame=A)
        grid2d = Grid(size=(100, 100), shape=(4, 4), eshape="Q4", frame=A)
        grid3d = Grid(size=(100, 100, 20), shape=(4, 4, 2), eshape="H8", frame=A)
        mesh = PolyData(frame=A)
        mesh["tri", "T3"] = tri.move(np.array([0.0, 0.0, -50]))
        mesh["grids", "Q4"] = grid2d.move(np.array([0.0, 0.0, 50]))
        mesh["grids", "H8"] = grid3d
        mesh.lock(create_mappers=True)
        mesh.to_standard_form(inplace=True)
        aT3 = mesh["tri"].area()
        aQ4 = mesh["grids", "Q4"].area()
        V0 = aT3 + aQ4 + mesh["grids", "H8"].volume()
        V1 = mesh.volume()
        ndf = mesh.nodal_distribution_factors()
        self.assertTrue(np.isclose(ndf.data.min(), 0.125))
        self.assertTrue(np.isclose(ndf.data.max(), 1.0))
        self.assertTrue(np.isclose(aT3, 10000.0))
        self.assertTrue(np.isclose(aQ4, 10000.0))
        self.assertTrue(np.isclose(V1, 220000.0))
        self.assertTrue(np.isclose(V0, V1))
        self.assertTrue(np.all(np.isclose(mesh["tri"].center(), [50.0, 50.0, -50.0])))
        self.assertTrue(
            np.all(np.isclose(mesh["grids", "Q4"].center(), [50.0, 50.0, 50.0]))
        )
        self.assertTrue(
            np.all(np.isclose(mesh["grids", "H8"].center(), [50.0, 50.0, 10.0]))
        )
        self.assertTrue(mesh.topology().is_jagged())
        mesh.unlock()
        mesh.copy()
        mesh.deepcopy()
        mesh.nummrg()
        mesh["grids", "Q4"].detach()

    def test_copy(self):
        gridparams = {
            "size": (1, 1),
            "shape": (4, 4),
            "eshape": (2, 2),
            "path": [0, 2, 3, 1],
        }
        coords, topo = grid(**gridparams)

        # the `grid` function creates a 2d mesh in the x-y plane,
        # but we want a 3d mesh, with zero values for the z axis.
        coords = np.pad(coords, ((0, 0), (0, 1)), mode="constant")

        frame = CartesianFrame(dim=3)

        pd = PointData(coords=coords, frame=frame)
        cd = Q4(topo=topo, frames=frame)

        mesh = PolyData(pd, cd)
        mesh_copy = mesh.copy()

        # Check if the copied object is not the same object as the original
        self.assertIsNot(mesh, mesh_copy)

        # Check if the data attributes are the same (shallow copy)
        self.assertIs(mesh.pd, mesh_copy.pd)
        self.assertIs(mesh.cd, mesh_copy.cd)

    def test_deepcopy(self):
        gridparams = {
            "size": (1, 1),
            "shape": (4, 4),
            "eshape": (2, 2),
            "path": [0, 2, 3, 1],
        }
        coords, topo = grid(**gridparams)

        # the `grid` function creates a 2d mesh in the x-y plane,
        # but we want a 3d mesh, with zero values for the z axis.
        coords = np.pad(coords, ((0, 0), (0, 1)), mode="constant")

        frame = CartesianFrame(dim=3)

        pd = PointData(coords=coords, frame=frame)
        cd = Q4(topo=topo, frames=frame)

        mesh = PolyData(pd, cd)
        mesh_copy = mesh.deepcopy()

        # Check if the copied object is not the same object as the original
        self.assertIsNot(mesh, mesh_copy)

        # Check if the data attributes are the same (shallow copy)
        self.assertIsNot(mesh.pd, mesh_copy.pd)
        self.assertIsNot(mesh.cd, mesh_copy.cd)

    def test_surface(self):
        gridparams = {
            "size": (1, 1, 1),
            "shape": (4, 4, 4),
            "eshape": "H8",
        }
        coords, topo = grid(**gridparams)

        frame = CartesianFrame(dim=3)

        pd = PointData(coords=coords, frame=frame)
        cd = H8(topo=topo, frames=frame)

        mesh = PolyData(pd, cd)
        surface = mesh.surface()
        surface_area = surface.area()
        self.assertTrue(np.isclose(surface_area, 6.0))


class TestPolyDataRead(unittest.TestCase):
    def test_read_from_pv(self):
        mesh = examples.download_cow_head()
        _ = PolyData.from_pv(mesh)

        filename = examples.planefile
        _ = PolyData.read(filename)

    def test_read_from_meshio(self):
        mesh = pv.read(examples.antfile)
        pv.save_meshio("mymesh.inp", mesh)
        mesh = meshio.read("mymesh.inp")
        mesh = PolyData.from_meshio(mesh)
        os.remove("mymesh.inp")


if __name__ == "__main__":
    unittest.main()
