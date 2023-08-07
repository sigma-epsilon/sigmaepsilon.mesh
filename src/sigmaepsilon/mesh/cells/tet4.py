# -*- coding: utf-8 -*-
from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from .base.polyhedron import TetraHedron
from ..utils.cells.tet4 import (
    shp_TET4_multi,
    dshp_TET4_multi,
    shape_function_matrix_TET4_multi,
    monoms_TET4,
)
from ..utils.cells.numint import Gauss_Legendre_Tet_1


class TET4(TetraHedron):
    """
    4-node isoparametric hexahedron.

    See Also
    --------
    :class:`~sigmaepsilon.mesh.polyhedron.TetraHedron`
    """

    shpfnc = shp_TET4_multi
    shpmfnc = shape_function_matrix_TET4_multi
    dshpfnc = dshp_TET4_multi
    monomsfnc = monoms_TET4

    quadrature = {
        "full": Gauss_Legendre_Tet_1(),
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
        monoms = [1, r, s, t]
        return locvars, monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        return np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

    @classmethod
    def lcenter(cls) -> ndarray:
        return np.array([[1 / 3, 1 / 3, 1 / 3]])
