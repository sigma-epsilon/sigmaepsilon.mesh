import warnings

import numpy as np
from meshio import Mesh as MeshioMesh

from sigmaepsilon.core.warning import SigmaEpsilonWarning

from ..data import PolyData, PointData, PolyCell
from ..space import CartesianFrame

from ..helpers import meshio_to_celltype, importers
from ..utils.space import frames_of_surfaces, frames_of_lines

__all__ = ["from_meshio"]


def from_meshio(mesh: MeshioMesh) -> PolyData:
    """
    Returns a :class:`~sigmaepsilon.mesh.polydata.PolyData` instance from a :class:`meshio.Mesh` instance.

    .. note::
        See https://github.com/nschloe/meshio for formats supported by
        ``meshio``. Be sure to install ``meshio`` with ``pip install
        meshio`` if you wish to use it.
    """
    GlobalFrame = CartesianFrame(dim=3)

    coords = mesh.points
    pd = PointData(coords=coords, frame=GlobalFrame)
    polydata = PolyData(pd)

    for cb in mesh.cells:
        cd = None
        cbtype = cb.type
        celltype: PolyCell = meshio_to_celltype.get(cbtype, None)
        if celltype:
            topo = np.array(cb.data, dtype=int)

            NDIM = celltype.Geometry.number_of_spatial_dimensions
            if NDIM == 1:
                frames = frames_of_lines(coords, topo)
            elif NDIM == 2:
                frames = frames_of_surfaces(coords, topo)
            elif NDIM == 3:
                frames = GlobalFrame

            cd = celltype(topo=topo, frames=frames)
            polydata[cbtype] = PolyData(cd)
        else:
            if cbtype != "vertex":  # pragma: no cover
                warnings.warn(
                    f"Cells of type '{cbtype}' are not supported here.",
                    SigmaEpsilonWarning,
                )

    return polydata


importers["meshio"] = from_meshio
