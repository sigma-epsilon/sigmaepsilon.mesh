from ..config import __haspyvista__
from ..helpers import importers

if not __haspyvista__:  # pragma: no cover

    def from_pv(*_) -> None:
        raise ImportError(
            "You need PyVista for this. Install it with 'pip install pyvista'. "
            "You may also need to restart your kernel and reload the package."
        )

else:
    import pyvista as pv
    from typing import Union

    from ..data import PolyData, PointData, PolyCell
    from ..space import CartesianFrame

    from ..helpers import vtk_to_celltype
    from ..utils.space import frames_of_surfaces, frames_of_lines

    pyVistaLike = Union[pv.PolyData, pv.PointGrid, pv.UnstructuredGrid]

    def from_pv(pvobj: pyVistaLike) -> PolyData:
        """
        Returns a :class:`~sigmaepsilon.mesh.polydata.PolyData` instance from
        a :class:`pyvista.PolyData` or a :class:`pyvista.UnstructuredGrid`
        instance.

        .. note::
            See https://github.com/pyvista/pyvista for more examples with
            ``pyvista``. Be sure to install ``pyvista`` with ``pip install
            pyvista`` if you wish to use it.

        Example
        -------
        >>> from pyvista import examples
        >>> from sigmaepsilon.mesh import PolyData
        >>> bunny = examples.download_bunny_coarse()
        >>> mesh = PolyData.from_pv(bunny)
        """
        coords, cells_dict = None, None

        if isinstance(pvobj, pv.UnstructuredGrid):
            coords = pvobj.points.astype(float)
            cells_dict = pvobj.cells_dict
        else:
            try:
                ugrid = pvobj.cast_to_unstructured_grid()
                return from_pv(ugrid)
            except Exception:
                raise TypeError(f"Can't import from type {type(pvobj)}.")

        GlobalFrame = CartesianFrame(dim=3)
        pd = PointData(coords=coords, frame=GlobalFrame)
        polydata = PolyData(pd)

        for vtkid, vtktopo in cells_dict.items():
            if vtkid in vtk_to_celltype:
                celltype: PolyCell = vtk_to_celltype[vtkid]

                NDIM = celltype.Geometry.number_of_spatial_dimensions
                if NDIM == 1:
                    frames = frames_of_lines(coords, vtktopo)
                elif NDIM == 2:
                    frames = frames_of_surfaces(coords, vtktopo)
                elif NDIM == 3:
                    frames = GlobalFrame

                cd = celltype(topo=vtktopo, frames=frames)
                polydata[vtkid] = PolyData(cd)
            else:  # pragma: no cover
                raise NotImplementedError(
                    f"The element type with vtkId <{vtkid}> is not yet supported here."
                )

        return polydata


importers["PyVista"] = from_pv

__all__ = ["from_pv"]
