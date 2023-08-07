from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from .base.polyhedron import QuadraticTetraHedron
from ..utils.cells.tet10 import (
    monoms_TET10,
)
from ..utils.cells.numint import Gauss_Legendre_Tet_4
from ..utils.cells.utils import volumes
from ..utils.utils import cells_coords


class TET10(QuadraticTetraHedron):
    """
    10-node isoparametric hexahedron.

    See Also
    --------
    :class:`~sigmaepsilon.mesh.polyhedron.QuadraticTetraHedron`
    """

    monomsfnc = monoms_TET10

    quadrature = {
        "full": Gauss_Legendre_Tet_4(),
    }

    @classmethod
    def polybase(cls) -> Tuple[List]:
        """
        Retruns the polynomial base of the master element.

        Returns
        -------
        list
            A list of SymPy symbols.
        list
            A list of monomials.
        """
        locvars = r, s, t = symbols("r s t", real=True)
        monoms = [1, r, s, t, r * s, r * t, s * t, r ** 2, s ** 2, t ** 2]
        return locvars, monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ]
        )

    @classmethod
    def lcenter(cls) -> ndarray:
        return np.array([[1 / 3, 1 / 3, 1 / 3]])

    def volumes(self, coords: ndarray = None, topo: ndarray = None) -> ndarray:
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        topo = self.topology().to_numpy() if topo is None else topo
        ecoords = cells_coords(coords, topo)
        qpos, qweight = self.quadrature["full"]
        dshp = self.shape_function_derivatives(qpos)
        return volumes(ecoords, dshp, qweight)
