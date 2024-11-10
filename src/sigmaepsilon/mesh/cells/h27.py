from typing import Tuple, List
from functools import partial

import numpy as np
from numpy import ndarray
import sympy as sy

from ..geometry import PolyCellGeometry3d
from ..data.polycell import PolyCell
from ..utils.cells.h27 import (
    shp_H27_multi,
    dshp_H27_multi,
    shape_function_matrix_H27_multi,
    monoms_H27,
)
from ..utils.numint import Gauss_Legendre_Hex_Grid


class H27(PolyCell):
    """
    Class for 27-node triquadratic hexahedra.

    ::

        top
        7---14---6
        |    |   |
        15--25--13
        |    |   |
        4---12---5

        middle
        19--23--18
        |    |   |
        20--26--21
        |    |   |
        16--22--17

        bottom
        3---10---2
        |    |   |
        11--24---9
        |    |   |
        0----8---1
    """

    label = "H27"

    class Geometry(PolyCellGeometry3d):
        number_of_nodes = 27
        vtk_cell_id = 29
        shape_function_evaluator = shp_H27_multi
        shape_function_matrix_evaluator = shape_function_matrix_H27_multi
        shape_function_derivative_evaluator = dshp_H27_multi
        monomial_evaluator = monoms_H27
        quadrature = {
            "full": partial(Gauss_Legendre_Hex_Grid, 3, 3, 3),
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
            locvars = r, s, t = sy.symbols("r s t", real=True)
            monoms = [
                1,
                r,
                s,
                t,
                s * t,
                r * t,
                r * s,
                r * s * t,
                r**2,
                s**2,
                t**2,
                r**2 * s,
                r * s**2,
                r * t**2,
                r**2 * t,
                s**2 * t,
                s * t**2,
                r**2 * s * t,
                r * s**2 * t,
                r * s * t**2,
                r**2 * s**2,
                s**2 * t**2,
                r**2 * t**2,
                r**2 * s**2 * t**2,
                r**2 * s**2 * t,
                r**2 * s * t**2,
                r * s**2 * t**2,
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
                    [-1.0, -1.0, -1],
                    [1.0, -1.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [0.0, -1.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [0.0, 1.0, -1.0],
                    [-1.0, 0.0, -1.0],
                    [0.0, -1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
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
            return np.array([0.0, 0.0, 0.0])
