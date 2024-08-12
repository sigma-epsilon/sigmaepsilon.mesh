from ..config import __haspyvista__
from ..helpers import exporters

if not __haspyvista__:  # pragma: no cover

    def to_pv(*_) -> None:
        raise ImportError(
            "You need PyVista for this. Install it with 'pip install pyvista'. "
            "You may also need to restart your kernel and reload the package."
        )

else:
    from typing import Union, Optional
    from contextlib import suppress

    import pyvista as pv
    import vtk
    from numpy import ndarray

    from ..data import PolyData
    from ..vtkutils import mesh_to_UnstructuredGrid as mesh_to_vtk

    pyVistaLike = Union[pv.PolyData, pv.PointGrid, pv.UnstructuredGrid]

    def to_pv(
        obj: PolyData,
        deepcopy: Optional[bool] = False,
        multiblock: Optional[bool] = False,
        scalars: Optional[Union[str, ndarray, None]] = None,
    ) -> Union[pv.UnstructuredGrid, pv.MultiBlock]:
        """
        Returns the mesh as a `PyVista` object, optionally set up with data.

        Parameters
        ----------
        deepcopy: bool, Optional
            Default is False.
        multiblock: bool, Optional
            Wether to return the blocks as a `vtkMultiBlockDataSet` or a list
            of `vtkUnstructuredGrid` instances. Default is False.
        scalars: str or numpy.ndarray, Optional
            A string or an array describing scalar data. Default is None.

        Returns
        -------
        pyvista.UnstructuredGrid or pyvista.MultiBlock
        """
        ugrids = []
        data = []
        for block, c, t, d in obj._detach_block_data_(scalars):
            vtk_cell_id = block.celltype.Geometry.vtk_cell_id
            ugrid = mesh_to_vtk(c, t, vtk_cell_id, deepcopy)
            ugrids.append(ugrid)
            data.append(d)

        if multiblock:
            mb = vtk.vtkMultiBlockDataSet()
            mb.SetNumberOfBlocks(len(ugrids))

            for i, ugrid in enumerate(ugrids):
                mb.SetBlock(i, ugrid)

            mb = pv.wrap(mb)

            with suppress(AttributeError):
                mb.wrap_nested()

            return mb
        else:
            if scalars is None:
                return [pv.wrap(ugrid) for ugrid in ugrids]
            else:
                res = []
                for ugrid, d in zip(ugrids, data):
                    pvobj = pv.wrap(ugrid)
                    if isinstance(d, ndarray):
                        if isinstance(scalars, str):
                            pvobj[scalars] = d
                        else:
                            pvobj["scalars"] = d
                    res.append(pvobj)
                return res


exporters["PyVista"] = to_pv

__all__ = ["to_pv"]
