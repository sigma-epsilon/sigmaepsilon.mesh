# -*- coding: utf-8 -*-
from .data.polydata import PolyData
from .cells import Tetra
import numpy as np

__all__ = ["TetMesh"]


class TetMesh(PolyData):
    """
    A class to handle tetrahedral meshes.

    Examples
    --------
    >>> from sigmaepsilon.mesh import TriMesh
    >>> trimesh = TriMesh(size=(800, 600), shape=(10, 10))
    >>> tetmesh = trimesh.extrude(h=300, N=5)
    >>> tetmesh.volume()
    144000000.0

    """

    def __init__(self, *args, celltype=None, topo=None, **kwargs):
        if celltype is None and topo is not None:
            if isinstance(topo, np.ndarray):
                nNode = topo.shape[1]
                if nNode == 4:
                    celltype = Tetra
            else:
                raise NotImplementedError
        assert celltype is not None
        super().__init__(*args, celltype=celltype, topo=topo, **kwargs)
