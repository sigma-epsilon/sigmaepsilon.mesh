import numpy as np

from sigmaepsilon.math import ascont

from .polydata import PolyData
from .cells import T3, T6, TET4
from .utils.space import frames_of_surfaces, is_planar_surface as is_planar
from .extrude import extrude_T3_TET4
from .triang import triangulate
from .utils.tri import edges_tri
from .utils.topology import unique_topo_data, T3_to_T6, T6_to_T3


class TriMesh(PolyData):
    """
    A class to handle triangular meshes.

    Parameters
    ----------
    points: numpy.ndarray, Optional
        2d numpy array of floats, describing a pointcloud. Default is None.
    triangles: numpy.ndarray, Optional
        2d numpy array of integers, describing the topology of a polygonal mesh.
        Default is None.

    Notes
    -----
    See the PolyData class for the rest of the possible arguments to the
    creator of this class. Note that, `points` and `triangles` are aliases
    to `coords` and `topo`, but the original terminology is still available.

    Examples
    --------
    Triangulate a rectangle of size 800x600 with a subdivision of 10x10
    and calculate the area

    >>> from sigmaepsilon.mesh import TriMesh, CartesianFrame
    >>> A = CartesianFrame(dim=3)
    >>> trimesh = TriMesh(size=(800, 600), shape=(10, 10), frame=A)
    >>> trimesh.area()
    480000.0

    Extrude to create a tetrahedral mesh

    >>> tetmesh = trimesh.extrude(h=300, N=5)
    >>> tetmesh.volume()
    144000000.0

    Calculate normals and tell if the triangles form
    a planar surface or not

    >>> trimesh.normals()
    >>> trimesh.is_planar()
    True

    See Also
    --------
    :class:`~sigmaepsilon.mesh.polydata.PolyData`
    :class:`~sigmaepsilon.mesh.space.frame.CartesianFrame`
    """

    _cell_classes_ = {
        3: T3,
        6: T6,
    }

    def __init__(self, *args, points=None, triangles=None, celltype=None, **kwargs):
        # parent class handles pointdata and celldata creation
        points = points if points is not None else kwargs.get("coords", None)
        triangles = triangles if triangles is not None else kwargs.get("topo", None)
        if triangles is None:
            try:
                points, triangles, _ = triangulate(*args, points=points, **kwargs)
            except Exception:
                raise RuntimeError
        if celltype is None and triangles is not None:
            if isinstance(triangles, np.ndarray):
                nNode = triangles.shape[1]
                if nNode == 3:
                    celltype = T3
                elif nNode == 6:
                    celltype = T6
            else:
                raise NotImplementedError
        if triangles.shape[1] == 3 and celltype.NNODE == 6:
            points, triangles = T3_to_T6(points, triangles)
        assert triangles.shape[1] == celltype.NNODE
        super().__init__(*args, coords=points, topo=triangles, **kwargs)

    def axes(self) -> np.ndarray:
        """
        Returns the normalized coordinate frames of triangles as a 3d numpy array.
        """
        x = self.coords()
        assert x.shape[-1] == 3, "This is only available for 3d datasets."
        return frames_of_surfaces(x, self.topology()[:, :3])

    def normals(self) -> np.ndarray:
        """
        Retuns the surface normals as a 2d numpy array.
        """
        return ascont(self.axes()[:, 2, :])

    def is_planar(self) -> bool:
        """
        Returns `True` if the triangles form a planar surface.
        """
        return is_planar(self.normals())

    def extrude(
        self, *args, celltype=None, h: float = None, N: int = None, **kwargs
    ) -> PolyData:
        """
        Exctrude mesh perpendicular to the plane of the triangulation.
        The target element type can be specified with the `celltype` argument.

        Parameters
        ----------
        h: float, Optional
            Size perpendicular to the plane of the surface to be extruded.
        N: int, Optional
            Number of subdivisions along the perpendicular direction.

        Returns
        -------
        :class:`~sigmaepsilon.mesh.tetmesh.TetMesh`
            A tetrahedral mesh.
        """
        from .tetmesh import TetMesh

        if not self.is_planar():
            raise RuntimeError("Only planar surfaces can be extruded!")
        assert celltype is None, "Currently only TET4 element is supported!"
        ct = TET4 if celltype == None else celltype
        inds = list(range(3))
        inds.pop(self._newaxis)
        x = self.coords()[:, inds]
        x, topo = extrude_T3_TET4(x, self.topology()[:, :3], h, N)
        c = np.zeros((x.shape[0], 3))
        c[:, inds] = x[:, :2]
        c[:, self._newaxis] = x[:, -1]
        return TetMesh(coords=c, topo=topo, celltype=ct, frame=self.frame)

    def edges(self, return_cells: bool = False):
        """
        Returns point indices of the unique edges in the model.
        If `return_cells` is `True`, it also returns the edge
        indices of the triangles, referencing the edges.

        Parameters
        ----------
        return_cells: bool, Optional
            If True, returns the edge indices of the triangles,
            that can be used to reconstruct the topology.
            Default is False.

        Returns
        -------
        numpy.ndarray
            Integer array of indices, representing point indices of edges.

        numpy.ndarray, Optional
            Integer array of indices, that together with the edge data
            reconstructs the topology.
        """
        edges, IDs = unique_topo_data(edges_tri(self.topology()))
        if return_cells:
            return edges, IDs
        else:
            return edges

    def to_triobj(self, *args, **kwargs):
        """
        Returns a triangulation object of a specified backend.
        See :func:`~sigmaepsilon.mesh.tri.triang.triangulate` for the details.

        Note
        ----
        During the process, the 6-noded triangles of the section are converted
        into 3-noded ones.

        See also
        --------
        :class:`~matplotlib.tri.Triangulation`
        :func:`~sigmaepsilon.mesh.triang.triangulate`
        """
        coords, topo = self.coords(), self.topology()
        if topo.shape[-1] == 6:
            path = np.array([[0, 5, 4], [5, 1, 3], [3, 2, 4], [5, 3, 4]], dtype=int)
            coords, topo = T6_to_T3(coords, topo, path=path)
        return triangulate(points=coords, triangles=topo)[-1]
