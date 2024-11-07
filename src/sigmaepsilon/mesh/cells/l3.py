# -*- coding: utf-8 -*-
from functools import partial

from ..geometry import PolyCellGeometry1d
from ..data.polycell import PolyCell
from ..utils.numint import Gauss_Legendre_Line_Grid
from ..utils.cells.l3 import monoms_L3


__all__ = ["L3"]


class L3(PolyCell):
    """
    Class for 3-node line segments.
    """

    class Geometry(PolyCellGeometry1d):
        number_of_nodes = 3
        vtk_cell_id = 21
        monomial_evaluator = monoms_L3
        quadrature = {
            "full": partial(Gauss_Legendre_Line_Grid, 3),
            "geometry": "full",
        }
