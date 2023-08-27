# -*- coding: utf-8 -*-
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
