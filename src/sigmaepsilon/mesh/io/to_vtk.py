from ..config import __hasvtk__
from ..helpers import exporters

if not __hasvtk__:  # pragma: no cover

    def to_vtk(*_) -> None:
        raise ImportError(
            "You need VTK for this. Install it with 'pip install vtk'. "
            "You may also need to restart your kernel and reload the package."
        )

else:
    import vtk
    from typing import Union

    from ..data import PolyData
    from ..vtkutils import mesh_to_UnstructuredGrid as mesh_to_vtk

    def to_vtk(
        obj: PolyData, deepcopy: bool = False, multiblock: bool = False
    ) -> Union[vtk.vtkUnstructuredGrid, vtk.vtkMultiBlockDataSet]:
        """
        Returns the mesh as a `VTK` object.

        Parameters
        ----------
        deepcopy: bool, Optional
            Default is False.
        multiblock: bool, Optional
            Wether to return the blocks as a `vtkMultiBlockDataSet` or a list
            of `vtkUnstructuredGrid` instances. Default is False.

        Returns
        -------
        vtk.vtkUnstructuredGrid or vtk.vtkMultiBlockDataSet
        """
        ugrids = []
        for block, c, t, _ in obj._detach_block_data_():
            vtk_cell_id = block.celltype.Geometry.vtk_cell_id
            ugrid = mesh_to_vtk(c, t, vtk_cell_id, deepcopy)
            ugrids.append(ugrid)

        if multiblock:
            mb = vtk.vtkMultiBlockDataSet()
            mb.SetNumberOfBlocks(len(ugrids))

            for i, ugrid in enumerate(ugrids):
                mb.SetBlock(i, ugrid)

            return mb
        else:
            if len(ugrids) > 1:
                return ugrids
            else:
                return ugrids[0]


exporters["vtk"] = to_vtk

__all__ = ["to_vtk"]
