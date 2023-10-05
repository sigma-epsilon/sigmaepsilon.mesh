from typing import Tuple, List

from sympy import symbols
import numpy as np
from numpy import ndarray

from sigmaepsilon.math.numint import gauss_points as gp

from ..geometry import PolyCellGeometry3d
from ..data.polycell import PolyCell
from ..utils.utils import cells_coords
from ..utils.cells.h8 import (
    shp_H8_multi,
    dshp_H8_multi,
    volumes_H8,
    shape_function_matrix_H8_multi,
    monoms_H8,
)
from ..utils.cells.numint import Gauss_Legendre_Hex_Grid


class H8(PolyCell):
    """
    8-node hexahedron.

    ::

        top
        7--6
        |  |
        4--5

        bottom
        3--2
        |  |
        0--1
    """

    label = "H8"

    class Geometry(PolyCellGeometry3d):
        number_of_nodes = 8
        vtk_cell_id = 12
        shape_function_evaluator: shp_H8_multi
        shape_function_matrix_evaluator: shape_function_matrix_H8_multi
        shape_function_derivative_evaluator: dshp_H8_multi
        monomial_evaluator: monoms_H8
        quadrature = {
            "full": Gauss_Legendre_Hex_Grid(2, 2, 2),
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
            locvars = r, s, t = symbols("r s t", real=True)
            monoms = [1, r, s, t, r * s, r * t, s * t, r * s * t]
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

        @classmethod
        def tetmap(cls) -> np.ndarray:
            return np.array(
                [[1, 2, 0, 5], [3, 0, 2, 7], [5, 4, 7, 0], [6, 5, 7, 2], [0, 2, 7, 5]],
                dtype=int,
            )

    def volumes(self) -> ndarray:
        """
        Returns the volumes of the cells.

        Returns
        -------
        numpy.ndarray
        """
        coords = self.source_coords()
        topo = self.topology().to_numpy()
        ecoords = cells_coords(coords, topo)
        qpos, qweight = gp(2, 2, 2)
        return volumes_H8(ecoords, qpos, qweight)
