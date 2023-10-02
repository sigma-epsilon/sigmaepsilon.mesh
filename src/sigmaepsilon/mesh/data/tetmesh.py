# -*- coding: utf-8 -*-
from typing import Iterable

import numpy as np

from ..typing import PolyCellProtocol
from .polydata import PolyData
from ..cells import Tetra, TET10

__all__ = ["TetMesh"]


class TetMesh(PolyData):
    """
    A class to handle tetrahedral meshes.
    
    All positional and keyword arguments are forwarded to
    :class:`~sigmaepsilon.mesh.data.polydata.PolyData`.
    
    Parameters
    ----------

    See also
    --------
    :class:`~sigmaepsilon.mesh.data.polydata.PolyData`.
    :func:`~sigmaepsilon.mesh.tetrahedralize.tetrahedralize`

    Examples
    --------
    >>> from sigmaepsilon.mesh import TriMesh
    >>> trimesh = TriMesh(size=(800, 600), shape=(10, 10))
    >>> tetmesh = trimesh.extrude(h=300, N=5)
    >>> tetmesh.volume()
    144000000.0
    """

    def __init__(
        self,
        *args,
        celltype: PolyCellProtocol = None,
        topo: Iterable[int] = None,
        **kwargs,
    ):
        if celltype is None and topo is not None:
            if isinstance(topo, np.ndarray):
                nNode = topo.shape[1]
                if nNode == 4:
                    celltype = Tetra
                elif nNode == 10:
                    celltype = TET10
            elif isinstance(topo, Iterable):
                topo = np.array(topo, dtype=int)
            else:
                raise TypeError(
                    (
                        f"Invalid type {type(topo)} for topology."
                        "It must be a list of integers or a numpy integer array."
                        )
                )
                
            nNode = topo.shape[1]
            if nNode == 4:
                celltype = Tetra
            elif nNode == 10:
                celltype = TET10
            else:
                raise ValueError("Tetrahedra must have 4 or 10 nodes.")
            
        assert celltype is not None
        super().__init__(*args, celltype=celltype, topo=topo, **kwargs)
