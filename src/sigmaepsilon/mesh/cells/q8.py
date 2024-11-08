import numpy as np
from numpy import ndarray
from sympy import symbols, Symbol

from ..geometry import PolyCellGeometry2d
from ..data.polycell import PolyCell
from ..utils.cells.q8 import (
    shp_Q8_multi,
    dshp_Q8_multi,
    shape_function_matrix_Q8_multi,
    monoms_Q8,
)
from ..utils.numint import Gauss_Legendre_Quad_9
from ..utils.topology import Q8_to_T3, trimap_Q8


class Q8(PolyCell):
    """
    Class for 8-noded quadratic quadrilaterals.
    """

    label = "Q8"

    class Geometry(PolyCellGeometry2d):
        number_of_nodes = 8
        vtk_cell_id = 23
        shape_function_evaluator = shp_Q8_multi
        shape_function_matrix_evaluator = shape_function_matrix_Q8_multi
        shape_function_derivative_evaluator = dshp_Q8_multi
        monomial_evaluator = monoms_Q8
        quadrature = {
            "full": Gauss_Legendre_Quad_9,
            "geometry": "full",
        }

        @classmethod
        def polybase(cls) -> tuple[list[Symbol], list[int | Symbol]]:
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
            ]
            return locvars, monoms

        @classmethod
        def master_coordinates(cls) -> ndarray[float]:
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
                ]
            )

        @classmethod
        def master_center(cls) -> ndarray[float]:
            """
            Returns the local coordinates of the center of the cell.

            Returns
            -------
            numpy.ndarray
            """
            return np.array([0.0, 0.0])

        @classmethod
        def trimap(cls) -> ndarray[int]:
            return trimap_Q8()

    def to_triangles(self) -> ndarray[int]:
        return Q8_to_T3(None, self.topology().to_numpy())[1]
