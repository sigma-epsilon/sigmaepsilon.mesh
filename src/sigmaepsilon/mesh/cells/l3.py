# -*- coding: utf-8 -*-
from .cell import PolyCell
from ..typing.geometry import PolyCellGeometryMixin1d
from ..utils.cells.numint import Gauss_Legendre_Line_Grid
from ..utils.cells.l3 import monoms_L3


__all__ = ["L3"]


class L3(PolyCell):
    """
    3-Node line element.
    """

    class Geometry(PolyCellGeometryMixin1d):
        number_of_nodes = 3
        vtk_cell_id = 21
        monomial_evaluator: monoms_L3
        quadrature = {
            "full": Gauss_Legendre_Line_Grid(3),
        }
