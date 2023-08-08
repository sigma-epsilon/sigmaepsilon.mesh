# -*- coding: utf-8 -*-
from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from .base.line import QuadraticLine
from ..utils.cells.numint import Gauss_Legendre_Line_Grid
from ..utils.cells.l3 import monoms_L3


__all__ = ["L3"]


class L3(QuadraticLine):
    """
    3-Node line element.

    See Also
    --------
    :class:`~sigmaepsilon.mesh.line.QuadraticLine`
    """

    monomsfnc = monoms_L3

    quadrature = {
        "full": Gauss_Legendre_Line_Grid(3),
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
        monoms = [1, r, r ** 2]
        return [locvars], monoms

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([-1.0, 0.0, 1.0])

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([0.0])
