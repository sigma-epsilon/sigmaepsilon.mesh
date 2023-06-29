import numpy as np

from .polydata import PolyData
from .pointdata import PointData
from .cells import TET4, TET10
from .utils.topology.tr import TET4_to_TET10
from .config import __has_tetgen__

if __has_tetgen__:
    import tetgen


def tetrahedralize(
    mesh: PolyData,
    *args,
    order: int = 1,
    mindihedral: int = 10,
    minratio: float = 1.5,
    quality: bool = True,
    steinerleft: int = -1,
    **kwargs
) -> PolyData:
    """
    Returns a tetrahedralized version of an input PolyData object.
    The tetrahedralization is done using the `tetgen` library. For
    the meaning of the parameters -except the first one which is a
    `PolyData` object- the user is redirected to `tetgen`.

    All extra positional and keyword arguments are forwarded to
    :func:`~tetgen.TetGen.tetrahedralize`.

    Notes
    -----
    The input mesh must contain exactly one block of solid cells.

    See Also
    --------
    :func:`~tetgen.TetGen.tetrahedralize`
    """
    assert (
        len(list(mesh.cellblocks(inclusive=True))) == 1
    ), "The mesh must contain exactly block of solid cells."

    pv_body = mesh.to_pv()[0]
    pv_surf = pv_body.extract_surface().triangulate()

    tet = tetgen.TetGen(pv_surf)
    tet.tetrahedralize(
        *args,
        order=1,
        mindihedral=mindihedral,
        minratio=minratio,
        quality=quality,
        steinerleft=steinerleft,
        **kwargs
    )
    grid = tet.grid

    coords = np.array(grid.points).astype(float)
    topo = grid.cells_dict[10].astype(int)

    if order == 2:
        coords, topo = TET4_to_TET10(coords, topo)
    elif order > 2:
        raise ValueError("'order' must be either 1 or 2")

    frame = mesh.frame
    pd = PointData(coords=coords, frame=frame)

    if topo.shape[1] == 4:
        cd = TET4(topo=topo, frames=frame)
    elif topo.shape[1] == 10:
        cd = TET10(topo=topo, frames=frame)

    return PolyData(pd, cd, frame=frame)
