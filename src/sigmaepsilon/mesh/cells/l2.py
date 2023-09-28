# -*- coding: utf-8 -*-
from ..geometry import PolyCellGeometry1d
from ..data.polycell import PolyCell
from ..utils.cells.l2 import (
    shp_L2_multi,
    dshp_L2_multi,
    shape_function_matrix_L2_multi,
    monoms_L2,
)
from ..utils.cells.numint import Gauss_Legendre_Line_Grid

__all__ = ["L2"]


class L2(PolyCell):
    """
    2-Node line element.
    """

    class Geometry(PolyCellGeometry1d):
        number_of_nodes = 2
        vtk_cell_id = 3
        shape_function_evaluator: shp_L2_multi
        shape_function_matrix_evaluator: shape_function_matrix_L2_multi
        shape_function_derivative_evaluator: dshp_L2_multi
        monomial_evaluator: monoms_L2
        quadrature = {
            "full": Gauss_Legendre_Line_Grid(2),
        }
