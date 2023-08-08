# -*- coding: utf-8 -*-
from typing import Tuple, List

import numpy as np
import numpy as np
from numpy import ndarray
from sympy import symbols

from .base.line import Line
from ..utils.cells.l2 import (
    shp_L2_multi,
    dshp_L2_multi,
    shape_function_matrix_L2_multi,
    monoms_L2,
)
from ..utils.cells.numint import Gauss_Legendre_Line_Grid

__all__ = ["L2"]


class L2(Line):
    """
    2-Node line element.

    See Also
    --------
    :class:`~sigmaepsilon.mesh.line.Line`
    """

    shpfnc = shp_L2_multi
    shpmfnc = shape_function_matrix_L2_multi
    dshpfnc = dshp_L2_multi
    monomsfnc = monoms_L2

    quadrature = {
        "full": Gauss_Legendre_Line_Grid(2),
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
        locvars = r = symbols("r", real=True)
        monoms = [1, r]
        return [locvars], monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([-1.0, 1.0])

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([0.0])
