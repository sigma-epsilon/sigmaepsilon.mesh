from typing import Iterable, Any, Tuple

import numpy as np
import scipy.spatial

try:
    from scipy.spatial import Delaunay as spDelaunay
except Exception:  # pragma: no cover
    from scipy.spatial.qhull import Delaunay as spDelaunay

from .utils.topology import unique_topo_data
from .utils.tri import edges_tri

from .config import __hasvtk__, __haspyvista__, __hasmatplotlib__

if __hasvtk__:
    from vtk import vtkIdList
if __haspyvista__:
    import pyvista as pv
if __hasmatplotlib__:
    import matplotlib.tri as tri


__all__ = ["triangulate"]


def triangulate(
    *args,
    points: np.ndarray = None,
    size: tuple = None,
    shape: tuple = None,
    origo: tuple = None,
    backend: str = "mpl",
    random: bool = False,
    triangles: Iterable = None,
    triobj=None,
    return_lines: bool = False,
    **kwargs,
):
    """
    Crates a triangulation using different backends.

    Parameters
    ----------
    points: numpy.ndarray, Optional
        A 2d array of coordinates of a flat surface. This or `shape`
        must be provided.
    size: tuple, Optional
        A 2-tuple, containg side lengths of a rectangular domain.
        Should be provided alongside `shape`. Must be provided if
        points is None.
    shape: tuple or int, Optional
        A 2-tuple, describing subdivisions along coordinate directions,
        or the number of base points used for triangulation.
        Should be provided alongside `size`.
    origo: numpy.ndarray, Optional
        1d float array, specifying the origo of the mesh.
    backend: str, Optional
        The backend to use. It can be 'mpl' (matplotlib), 'pv' (pyvista),
        or 'scipy'. Default is 'mpl'.
    random: bool, Optional
        If `True`, and points are provided by `shape`, it creates random
        points in the region defined by `size`. Default is False.
    triangles: numpy.ndarray
        2d integer array of vertex indices. If both `points` and
        `triangles` are specified, the only thing this function does is to
        create a triangulation object according to the argument `backend`.
    triobj: object
        An object representing a triangulation. It can be
        - a result of a call to matplotlib.tri.Triangulation
        - a Delaunay object from scipy
        - an instance of pyvista.PolyData
        In this case, the function can be used to transform between
        the supported backends.
    return_lines: bool, Optional
        If `True` the function return the unique edges of the
        triangulation and the indices of the edges for each triangle.

    Returns
    -------
    numpy.ndarray
        A 2d float array of coordinates of points.
    numpy.ndarray
        A 2d integer array representing the topology.
    MeshLike
        An object representing the triangulation, according to the specified backend.

    Examples
    --------
    Triangulate a rectangle of size 800x600 with a subdivision of 10x10

    >>> from sigmaepsilon.mesh import triangulate
    >>> coords, topo, triobj = triangulate(size=(800, 600), shape=(10, 10))
    ...

    Triangulate a rectangle of size 800x600 with a number of 100 randomly
    distributed points

    >>> coords, topo, triobj = triangulate(size=(800, 600), shape=100, random=True)
    ...
    """
    if len(args) > 0:
        if _is_triobj(args[0]):
            triobj = args[0]
            
    if triobj is not None:
        points, triangles = _get_triobj_data(triobj, *args, **kwargs)
    else:
        # create points from input
        if points is None:
            assert size is not None, (
                "Either a collection of points, or the size of a "
                "rectangular domain must be provided!"
            )
            
            if origo is None:
                origo = (0, 0, 0)
            else:
                if len(origo) == 2:
                    origo = origo + (0,)
            
            if shape is None:
                shape = (3, 3)
            
            if isinstance(shape, int):
                if random:
                    x = np.hstack(
                        [np.array([0, 1, 1, 0], dtype=float), np.random.rand(shape)]
                    )
                    y = np.hstack(
                        [np.array([0, 0, 1, 1], dtype=float), np.random.rand(shape)]
                    )
                    z = np.zeros(len(x), dtype=float)
                    points = np.c_[
                        x * size[0] - origo[0], y * size[1] - origo[1], z - origo[2]
                    ]
                else:
                    shape = (shape, shape)
                    
            if points is None and isinstance(size, tuple):
                x = np.linspace(-origo[0], size[0] - origo[0], num=shape[0])
                y = np.linspace(-origo[1], size[1] - origo[1], num=shape[1])
                z = np.zeros(len(x), dtype=float) - origo[2]
                xx, yy = np.meshgrid(x, y)
                zz = np.zeros(xx.shape, dtype=xx.dtype)
                # Get the points as a 2D NumPy array (N by 2)
                points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]

        # generate triangles
        if triangles is None:
            if backend == "mpl":
                triobj = tri.Triangulation(points[:, 0], points[:, 1])
                triangles = triobj.triangles
            elif backend == "scipy":
                triobj = scipy.spatial.Delaunay(points[:, 0:2])
                triangles = triobj.simplices
            elif backend == "pv":
                if not __haspyvista__ or not __hasvtk__:
                    raise ImportError("PyVista must be installed for this.")
                cloud = pv.PolyData(points)
                triobj = cloud.delaunay_2d()
                nCell = triobj.n_cells
                triangles = np.zeros((nCell, 3), dtype=int)
                for cellID in range(nCell):
                    idlist = vtkIdList()
                    triobj.GetCellPoints(cellID, idlist)
                    n = idlist.GetNumberOfIds()
                    triangles[cellID] = [idlist.GetId(i) for i in range(n)]
        else:
            assert backend == "mpl", (
                "This feature is not yet supported by "
                "other backends, only matplotlib."
            )
            triobj = tri.Triangulation(points[:, 0], points[:, 1], triangles=triangles)
    
    if return_lines:
        edges, edgeIDs = unique_topo_data(edges_tri(triangles))
        return points, edges, triangles, edgeIDs, triobj
    
    return points, triangles, triobj


def triobj_to_mpl(triobj, *args, **kwargs) -> tri.Triangulation:
    """
    Transforms a triangulation into a matplotlib.tri.Triangulation
    object.
    """
    assert _is_triobj(triobj)
    if isinstance(triobj, tri.Triangulation):
        return triobj
    else:
        points, triangles = _get_triobj_data(triobj, *args, **kwargs)
        kwargs["backend"] = "mpl"
        _, _, triang = triangulate(*args, points=points, triangles=triangles, **kwargs)
        return triang


def _get_triobj_data(obj:Any=None, *_, trim2d:bool=True, **__) -> Tuple[np.ndarray]:
    coords, topo = None, None
    
    if isinstance(obj, spDelaunay):
        coords = obj.points
        topo = obj.simplices
    elif isinstance(obj, tri.Triangulation):
        coords = np.vstack((obj.x, obj.y)).T
        topo = obj.triangles
    else:
        if __haspyvista__ and __hasvtk__:
            if isinstance(obj, pv.PolyData):
                if trim2d:
                    coords = obj.points[:, 0:2]
                else:
                    coords = obj.points
                triang = obj.delaunay_2d()
                nCell = triang.n_cells
                topo = np.zeros((nCell, 3), dtype=np.int32)
                for cellID in range(nCell):
                    idlist = vtkIdList()
                    triang.GetCellPoints(cellID, idlist)
                    n = idlist.GetNumberOfIds()
                    topo[cellID] = [idlist.GetId(i) for i in range(n)]
    
    if coords is None or topo is None:
        raise RuntimeError("Failed to recognize a valid input.")
    
    return coords, topo


def _is_triobj(triobj: Any) -> bool:
    try:
        if isinstance(triobj, spDelaunay) or isinstance(triobj, tri.Triangulation):
            return True
        else:
            if __haspyvista__:
                if isinstance(triobj, pv.PolyData):
                    if hasattr(triobj, "delaunay_2d"):
                        return True
        return False
    except Exception:  # pragma: no cover
        return False
