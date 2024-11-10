# -*- coding: utf-8 -*-
from typing import Tuple, List

import numpy as np
from numpy import ndarray
from sympy import symbols

from ..geometry import PolyCellGeometry2d
from ..data.polycell import PolyCell
from ..utils.cells.q4 import (
    shp_Q4_multi,
    dshp_Q4_multi,
    shape_function_matrix_Q4_multi,
    monoms_Q4,
)
from ..utils.numint import Gauss_Legendre_Quad_4
from ..utils.topology import Q4_to_T3


class Q4(PolyCell):
    """
    Class for 4-noded bilinear quadrilaterals.
    """

    label = "Q4"

    class Geometry(PolyCellGeometry2d):
        number_of_nodes = 4
        vtk_cell_id = 9
        shape_function_evaluator = shp_Q4_multi
        shape_function_matrix_evaluator = shape_function_matrix_Q4_multi
        shape_function_derivative_evaluator = dshp_Q4_multi
        monomial_evaluator = monoms_Q4
        quadrature = {
            "full": Gauss_Legendre_Quad_4,
            "geometry": "full",
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
        def master_coordinates(cls) -> ndarray:
            """
            Returns local coordinates of the cell.

            Returns
            -------
            numpy.ndarray
            """
            return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

        @classmethod
        def master_center(cls) -> ndarray:
            """
            Returns the local coordinates of the center of the cell.

            Returns
            -------
            numpy.ndarray
            """
            return np.array([0.0, 0.0])

        @classmethod
        def trimap(cls) -> ndarray:
            return np.array([[0, 1, 2], [0, 2, 3]], dtype=int)

    def to_triangles(self) -> ndarray:
        return Q4_to_T3(None, self.topology().to_numpy())[1]
