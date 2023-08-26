# -*- coding: utf-8 -*-
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
