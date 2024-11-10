from typing import Tuple, List

import numpy as np
from numpy import ndarray
from sympy import symbols

from ..geometry import PolyCellGeometry2d
from ..data.polycell import PolyCell
from ..utils.cells.q9 import (
    shp_Q9_multi,
    dshp_Q9_multi,
    shape_function_matrix_Q9_multi,
    monoms_Q9,
)
from ..utils.numint import Gauss_Legendre_Quad_9
from ..utils.topology import Q4_to_T3, Q9_to_Q4


class Q9(PolyCell):
    """
    Class for 9-noded biquadratic quadrilaterals.
    """

    label = "Q9"

    class Geometry(PolyCellGeometry2d):
        number_of_nodes = 9
        vtk_cell_id = 28
        shape_function_evaluator = shp_Q9_multi
        shape_function_matrix_evaluator = shape_function_matrix_Q9_multi
        shape_function_derivative_evaluator = dshp_Q9_multi
        monomial_evaluator = monoms_Q9
        quadrature = {
            "full": Gauss_Legendre_Quad_9,
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
            monoms = [
                1,
                r,
                s,
                r * s,
                r**2,
                s**2,
                r * s**2,
                s * r**2,
                s**2 * r**2,
            ]
            return locvars, monoms

        @classmethod
        def master_coordinates(cls) -> ndarray:
            """
            Returns local coordinates of the cell.

            Returns
            -------
            numpy.ndarray
            """
            return np.array(
                [
                    [-1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                    [-1.0, 1.0],
                    [0.0, -1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, 0.0],
                ]
            )

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
            return np.array(
                [
                    [0, 4, 8],
                    [0, 8, 7],
                    [4, 1, 5],
                    [4, 5, 8],
                    [8, 5, 2],
                    [8, 2, 6],
                    [7, 8, 6],
                    [7, 6, 3],
                ],
                dtype=int,
            )

    def to_triangles(self) -> ndarray:
        return Q4_to_T3(*Q9_to_Q4(None, self.topology().to_numpy()))[1]
