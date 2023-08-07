# -*- coding: utf-8 -*-
from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from .base.polygon import Triangle
from ..utils.cells.numint import Gauss_Legendre_Tri_1
from ..utils.cells.t3 import (
    shp_T3_multi,
    dshp_T3_multi,
    shape_function_matrix_T3_multi,
    monoms_T3,
)


class T3(Triangle):
    """
    A class to handle 3-noded triangles.

    See Also
    --------
    :class:`~sigmaepsilon.mesh.polygon.Triangle`
    """

    shpfnc = shp_T3_multi
    shpmfnc = shape_function_matrix_T3_multi
    dshpfnc = dshp_T3_multi
    monomsfnc = monoms_T3

    quadrature = {
        "full": Gauss_Legendre_Tri_1(),
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
        locvars = r, s = symbols("r s", real=True)
        monoms = [1, r, s]
        return locvars, monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([[1 / 3, 1 / 3]])
