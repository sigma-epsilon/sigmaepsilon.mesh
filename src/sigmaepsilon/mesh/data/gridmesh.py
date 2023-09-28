from .polydata import PolyData
from ..grid import grid

__all__ = ["Grid"]


class Grid(PolyData):
    """
    A class to generate meshes based on grid-like data. All input arguments are 
    forwarded to :func:`~sigmaepsilon.mesh.grid.grid`. The difference is that
    a :class:`~sigmaepsilon.mesh.data.polydata.PolyData` instance is returned, 
    insted of raw mesh data.

    Examples
    --------
    >>> from sigmaepsilon.mesh import Grid
    >>> size = 80, 60, 20
    >>> shape = 8, 6, 2
    >>> grid = Grid(size=size, shape=shape, eshape='H8')

    See also
    --------
    :class:`~sigmaepsilon.mesh.data.polydata.PolyData`
    :func:`~sigmaepsilon.mesh.grid.grid`
    """

    def __init__(self, *args, celltype=None, frame=None, eshape=None, **kwargs):
        # parent class handles pointdata and celldata creation
        coords, topo = grid(*args, eshape=eshape, **kwargs)
        super().__init__(
            *args, coords=coords, topo=topo, celltype=celltype, frame=frame, **kwargs
        )
