# -*- coding: utf-8 -*-
from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..polygon import Quadrilateral
from ..utils.cells.q4 import (
    shp_Q4_multi,
    dshp_Q4_multi,
    shape_function_matrix_Q4_multi,
    monoms_Q4,
)
from ..utils.cells.gauss import Gauss_Legendre_Quad_4


class Q4(Quadrilateral):
    """
    Polygon class for 4-noded bilinear quadrilaterals.

    See Also
    --------
    :class:`~polymesh.polygon.Quadrilateral`
    """

    shpfnc = shp_Q4_multi
    shpmfnc = shape_function_matrix_Q4_multi
    dshpfnc = dshp_Q4_multi
    monomsfnc = monoms_Q4

    quadrature = {
        "full": Gauss_Legendre_Quad_4(),
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
        locvars = r, s = symbols("r, s", real=True)
        monoms = [1, r, s, r * s]
        return locvars, monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([0.0, 0.0])
