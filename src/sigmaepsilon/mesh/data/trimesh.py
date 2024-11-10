from typing import Tuple, Optional, Any

import numpy as np
from numpy import ndarray

from sigmaepsilon.math import ascont
from sigmaepsilon.math.linalg import ReferenceFrame

from .polydata import PolyData
from .pointdata import PointData
from ..cells import TET4
from ..utils.space import frames_of_surfaces, is_planar_surface as is_planar
from ..extrude import extrude_T3_TET4
from ..triang import triangulate
from ..utils.tri import edges_tri
from ..utils.topology import unique_topo_data, T6_to_T3


__all__ = ["TriMesh"]


class TriMesh(PolyData):
    """
    A class to handle triangular meshes.

    All positional and keyword arguments not listed here are forwarded to
    :class:`~sigmaepsilon.mesh.data.polydata.PolyData`.

    Notes
    -----
    See the PolyData class for the rest of the possible arguments to the
    creator of this class.

    Examples
    --------
    Triangulate a rectangle of size 800x600 with a subdivision of 10x10
    and calculate the area

    >>> from sigmaepsilon.mesh import TriMesh, CartesianFrame, PointData, triangulate
    >>> from sigmaepsilon.mesh.cells import T3
    >>> import numpy as np
    >>> frame = CartesianFrame(dim=3)
    >>> coords, topo, _ = triangulate(size=(800, 600), shape=(10, 10))
    >>> pd = PointData(coords=coords, frame=frame)
    >>> cd = T3(topo=topo)
    >>> trimesh = TriMesh(pd, cd)
    >>> np.isclose(trimesh.area(), 480000.0)
    True

    Extrude to create a tetrahedral mesh

    >>> tetmesh = trimesh.extrude(h=300, N=5)
    >>> np.isclose(tetmesh.volume(), 144000000.0)
    True

    Calculate normals and tell if the triangles form
    a planar surface or not

    >>> normals = trimesh.normals()
    >>> trimesh.is_planar()
    True

    Create a circular disk

    >>> from sigmaepsilon.mesh.recipes import circular_disk
    >>> trimesh = circular_disk(120, 60, 5, 25)

    See Also
    --------
    :class:`~sigmaepsilon.mesh.data.polydata.PolyData`
    :class:`~sigmaepsilon.mesh.space.frame.CartesianFrame`
    """

    def axes(self) -> np.ndarray:
        """
        Returns the normalized coordinate frames of triangles as a 3d numpy array.
        """
        x = self.coords()
        assert x.shape[-1] == 3, "This is only available for 3d datasets."
        return frames_of_surfaces(x, self.topology().to_numpy()[:, :3])

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

    def extrude(self, *, h: float, N: int) -> PolyData:
        """
        Exctrude mesh perpendicular to the plane of the triangulation.
        The target element type can be specified with the `celltype` argument.

        Parameters
        ----------
        h: float
            Size perpendicular to the plane of the surface to be extruded.
        N: int
            Number of subdivisions along the perpendicular direction.

        Returns
        -------
        :class:`~sigmaepsilon.mesh.tetmesh.TetMesh`
            A tetrahedral mesh.
        """
        if not self.is_planar():
            raise RuntimeError("Only planar surfaces can be extruded!")

        frame = ReferenceFrame(self.cd.frames[0])
        x = self.coords(target=frame)[:, :2]
        x, topo = extrude_T3_TET4(x, self.topology().to_numpy()[:, :3], h, N)
        pd = PointData(coords=x, frame=frame)
        cd = TET4(topo=topo, frames=frame)
        return PolyData(pd, cd)

    def edges(self, return_cells: bool = False) -> Tuple[ndarray, Optional[ndarray]]:
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
        edges, IDs = unique_topo_data(edges_tri(self.topology().to_numpy()))
        if return_cells:
            return edges, IDs
        else:
            return edges

    def to_triobj(self) -> Any:
        """
        Returns a triangulation object of a specified backend.
        See :func:`~sigmaepsilon.mesh.triang.triangulate` for the details.

        Note
        ----
        During the process, the 6-noded triangles of the section are converted
        into 3-noded ones.

        See also
        --------
        :class:`~matplotlib.tri.Triangulation`
        :func:`~sigmaepsilon.mesh.triang.triangulate`
        """
        coords, topo = self.coords(), self.topology().to_numpy()
        if topo.shape[-1] == 6:
            path = np.array([[0, 5, 4], [5, 1, 3], [3, 2, 4], [5, 3, 4]], dtype=int)
            coords, topo = T6_to_T3(coords, topo, path=path)
        return triangulate(points=coords, triangles=topo)[-1]
