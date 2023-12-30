# -*- coding: utf-8 -*-
import unittest

from sigmaepsilon.mesh.vtkcelltypes import (
    vtkCellTypes,
    CellTypeId,
    celltype_by_name,
    celltype_by_value,
    meshio_to_vtk,
)


class TestVTKCellTypes(unittest.TestCase):
    def test_vtk_celltypes(self):
        self.assertTrue(vtkCellTypes["VTK_LINE"] == celltype_by_name("VTK_LINE"))
        vtkCellTypes.celltype("VTK_LINE")
        vtkCellTypes.celltype(3)
        celltype_by_value(3)
        meshio_to_vtk("triangle")
        meshio_to_vtk("A___$$-8-$___")
        CellTypeId("VTK_LINE")
        CellTypeId("A___$$-8-$___")


if __name__ == "__main__":
    unittest.main()
