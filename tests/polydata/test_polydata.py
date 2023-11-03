import os
import unittest
from typing import Iterable

import numpy as np
import pyvista as pv
from pyvista import examples
import meshio

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.core.warning import SigmaEpsilonPerformanceWarning
from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame, triangulate
from sigmaepsilon.mesh.data.trimesh import TriMesh
from sigmaepsilon.mesh.data.celldata import CellData
from sigmaepsilon.mesh.space import StandardFrame
from sigmaepsilon.mesh.cells import H8, Q4, T3
from sigmaepsilon.mesh.grid import grid


class TestPolyDataSingleBlock(SigmaEpsilonTestCase):
    def setUp(self) -> None:
        A = StandardFrame(dim=3)
        coords, topo, _ = triangulate(size=(100, 100), shape=(4, 4))
        pd = PointData(coords=coords, frame=A)
        pd["random_data"] = np.random.rand(coords.shape[0])
        cd = T3(topo=topo, frames=A)
        cd["random_data"] = np.random.rand(topo.shape[0])
        tri = TriMesh(cd, pd)
        self.mesh = tri
        
    def test_basic(self):
        mesh: PolyData = self.mesh
        mesh.parent = mesh.parent
        self.assertFalse(mesh.topology().is_jagged())
        self.assertIsInstance(mesh.cells_at_nodes(), Iterable)
        
    def test_set_pointdata_raises_TypeError(self):
        with self.assertRaises(TypeError) as cm:
            self.mesh.pointdata = "a"
        the_exception = cm.exception
        self.assertEqual(
            the_exception.args[0],
            "Value must be a PointData instance.",
        )
        
    def test_set_celldata_raises_TypeError(self):
        with self.assertRaises(TypeError) as cm:
            self.mesh.pointdata = "a"
        the_exception = cm.exception
        self.assertEqual(
            the_exception.args[0],
            "Value must be a PointData instance.",
        )

    def test_to_lists(self):
        self.mesh.to_lists(
            point_fields=["random_data"],
            cell_fields=["random_data"]
        )
        
    def test_rewire(self):
        self.mesh.rewire()
        self.mesh.rewire(deep=True)
        
    def test_to_standard_form(self):
        self.mesh.to_standard_form()
        self.mesh.to_standard_form(inplace=True)
        
    def test_nodal_distribution_factors(self):
        self.mesh.nodal_distribution_factors()


class TestPolyDataMultiBlock(SigmaEpsilonTestCase):
    def setUp(self) -> None:
        A = StandardFrame(dim=3)

        coords, topo, _ = triangulate(size=(100, 100), shape=(4, 4))
        pd = PointData(coords=coords, frame=A)
        pd["random_data"] = np.random.rand(coords.shape[0])
        cd = T3(topo=topo, frames=A)
        cd["random_data"] = np.random.rand(topo.shape[0])
        tri = TriMesh(pd, cd)

        coords, topo = grid(size=(100, 100), shape=(4, 4), eshape="Q4")
        pd = PointData(coords=coords, frame=A)
        cd = Q4(topo=topo, frames=A)
        cd["random_data"] = np.random.rand(topo.shape[0])
        grid2d = PolyData(pd, cd)

        coords, topo = grid(size=(100, 100, 20), shape=(4, 4, 2), eshape="H8")
        pd = PointData(coords=coords, frame=A)
        cd = H8(topo=topo, frames=A)
        cd["random_data"] = np.random.rand(topo.shape[0])
        grid3d = PolyData(pd, cd)

        mesh = PolyData(frame=A)
        mesh["tri", "T3"] = tri.move(np.array([0.0, 0.0, -50]))
        mesh["grids", "Q4"] = grid2d.move(np.array([0.0, 0.0, 50]))
        mesh["grids", "H8"] = grid3d
        mesh.lock(create_mappers=True)
        mesh.to_standard_form(inplace=True)
        self.mesh = mesh

    def test_misc(self):
        mesh: PolyData = self.mesh
        aT3 = mesh["tri"].area()
        aQ4 = mesh["grids", "Q4"].area()
        V0 = aT3 + aQ4 + mesh["grids", "H8"].volume()
        V1 = mesh.volume()
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

        self.assertIsInstance(mesh.index_of_closest_point([0, 0, 0]), int)
        self.assertIsInstance(mesh.index_of_furthest_point([0, 0, 0]), int)
        self.assertIsInstance(mesh.index_of_closest_cell([0, 0, 0]), int)
        self.assertIsInstance(mesh.index_of_furthest_cell([0, 0, 0]), int)

        self.assertIsInstance(mesh.point_fields, Iterable)
        self.assertIsInstance(mesh.cell_fields, Iterable)

        self.assertIsInstance(mesh["grids", "Q4"].cd.frames, np.ndarray)
        mesh["grids", "Q4"].cd.frames = mesh["grids", "Q4"].cd.frames
        
        mesh._in_all_pointdata_("_")
        mesh._in_all_celldata_("_")
        dbkey = PointData._dbkey_x_
        self.assertTrue(mesh._in_all_pointdata_(dbkey))
        self.assertTrue(mesh._in_all_pointdata_("random_data"))
        dbkey = CellData._dbkey_nodes_
        self.assertTrue(mesh._in_all_celldata_(dbkey))
        self.assertTrue(mesh._in_all_celldata_("random_data"))
        
    def test_root(self):
        self.assertEqual(self.mesh["grids", "Q4"].root, self.mesh)
        self.assertEqual(self.mesh["grids", "H8"].root, self.mesh)
        self.assertEqual(self.mesh["tri", "T3"].root, self.mesh)
        self.assertEqual(self.mesh["tri"].root, self.mesh)
        self.assertEqual(self.mesh["grids"].root, self.mesh)
        
    def blocks_of_cells(self):
        mesh: PolyData = self.mesh
        mesh._cid2bid=None
        self.assertWarns(SigmaEpsilonPerformanceWarning, mesh.blocks_of_cells)
        mesh.lock()
        mesh.blocks_of_cells()

    def test_coordinates(self):
        mesh: PolyData = self.mesh
        self.assertIsInstance(mesh.coords(), np.ndarray)
        coords, _ = mesh.coords(return_inds=True)
        mesh.bounds()
        self.mesh["grids", "Q4"].cells_coords()

        nP = self.mesh.number_of_points()
        self.assertEqual(nP, coords.shape[0])

    def test_topology(self):
        mesh: PolyData = self.mesh
        self.assertTrue(mesh.topology().is_jagged())
        self.assertIsInstance(mesh["grids", "Q4"].topology().to_numpy(), np.ndarray)

        topo, _ = mesh.topology(return_inds=True)
        mesh.cell_indices()
        mesh["grids", "Q4"].topology(return_inds=True)
        mesh["grids", "Q4"].cell_indices()

        nE = self.mesh.number_of_cells()
        self.assertEqual(nE, topo.shape[0])

    def test_copy(self):
        mesh: PolyData = self.mesh
        mesh_copy = self.mesh.copy()

        # Check if the copied object is not the same object as the original
        self.assertIsNot(mesh, mesh_copy)

        # Check if the data attributes are the same (shallow copy)
        self.assertIs(mesh.pd, mesh_copy.pd)
        for block, block_copy in zip(mesh.cellblocks(), mesh_copy.cellblocks()):
            self.assertIs(block.cd, block_copy.cd)

    def test_deepcopy(self):
        mesh: PolyData = self.mesh
        mesh_copy = mesh.deepcopy()

        # Check if the copied object is not the same object as the original
        self.assertIsNot(mesh, mesh_copy)

        # Check if the data attributes are the same (shallow copy)
        self.assertIsNot(mesh.pd, mesh_copy.pd)
        for block, block_copy in zip(mesh.cellblocks(), mesh_copy.cellblocks()):
            self.assertIsNot(block.cd, block_copy.cd)

    def test_nodal_distribution_factors(self):
        ndf = self.mesh.nodal_distribution_factors()
        self.assertTrue(np.isclose(ndf.data.min(), 0.125))
        self.assertTrue(np.isclose(ndf.data.max(), 1.0))
        self.mesh.nodal_distribution_factors(weights="uniform")

    def test_lock(self):
        self.mesh.lock()
        self.mesh.unlock()
        self.mesh.lock(create_mappers=True)

    def test_nummrg(self):
        self.mesh.nummrg()

    def test_detach(self):
        self.mesh["grids", "Q4"].detach()
        self.mesh["grids", "H8"].detach(nummrg=True)
        self.mesh["grids"].detach(nummrg=True)

    def test_to_vtk(self):
        self.mesh.to_vtk()
        self.mesh.to_vtk(multiblock=True)

    def test_to_pv(self):
        self.mesh.to_pv()
        self.mesh.to_pv(multiblock=True)

    def test_delete(self):
        del self.mesh["grids", "Q4"]
        self.mesh.lock()

        def boo():
            self.mesh["grids", "Q4"]

        self.assertFailsProperly(KeyError, boo)
        
    """def test_replace(self):
        A = StandardFrame(dim=3)
        coords, topo, _ = triangulate(size=(100, 100), shape=(4, 4))
        pd = PointData(coords=coords, frame=A)
        cd = T3(topo=topo, frames=A)
        tri = TriMesh(pd, cd)
        self.mesh["tri", "T3"] = tri"""

    def test_centers(self):
        self.mesh.centers()
        target = CartesianFrame(dim=3)
        self.mesh.centers(target=target)

    def test_adjacency(self):
        self.mesh.nodal_adjacency()
        self.mesh.cells_at_nodes()
        self.mesh.cells_around_cells(radius=1.0)
        

class TestSurfaceExtraction(unittest.TestCase):
    def test_surface_extraction(self):
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
