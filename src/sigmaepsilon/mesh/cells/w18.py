import numpy as np
from numpy import ndarray
from sympy import symbols

from ..geometry import PolyCellGeometry3d
from ..data.polycell import PolyCell
from ..utils.numint import Gauss_Legendre_Wedge_3x3
from ..utils.cells.w18 import monoms_W18
from ..utils.topology import compose_trmap
from .w6 import W6


class W18(PolyCell):
    """
    Class for 18-noded biquadratic wedges.
    """

    label = "W18"

    class Geometry(PolyCellGeometry3d):
        number_of_nodes = 18
        vtk_cell_id = 32
        monomial_evaluator = monoms_W18
        quadrature = {
            "full": Gauss_Legendre_Wedge_3x3,
            "geometry": "full",
        }

        @classmethod
        def polybase(cls) -> tuple[list, list]:
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
            monoms = [
                1,
                r,
                s,
                r**2,
                s**2,
                r * s,
                t,
                t * r,
                t * s,
                t * r**2,
                t * s**2,
                t * r * s,
                t**2,
                t**2 * r,
                t**2 * s,
                t**2 * r**2,
                t**2 * s**2,
                t**2 * r * s,
            ]
            return locvars, monoms

        @classmethod
        def master_coordinates(cls) -> ndarray[float]:
            """Returns local coordinates of the master cell as a NumPy array."""
            return np.array(
                [
                    [-1 / 3, -1 / 3, -1.0],
                    [2 / 3, -1 / 3, -1.0],
                    [-1 / 3, 2 / 3, -1.0],
                    [-1 / 3, -1 / 3, 1.0],
                    [2 / 3, -1 / 3, 1.0],
                    [-1 / 3, 2 / 3, 1.0],
                    [1 / 6, -1 / 3, -1.0],
                    [1 / 6, 1 / 6, -1.0],
                    [-1 / 3, 1 / 6, -1.0],
                    [1 / 6, -1 / 3, 1.0],
                    [1 / 6, 1 / 6, 1.0],
                    [-1 / 3, 1 / 6, 1.0],
                    [-1 / 3, -1 / 3, 0.0],
                    [2 / 3, -1 / 3, 0.0],
                    [-1 / 3, 2 / 3, 0.0],
                    [1 / 6, -1 / 3, 0.0],
                    [1 / 6, 1 / 6, 0.0],
                    [-1 / 3, 1 / 6, 0.0],
                ]
            )

        @classmethod
        def master_center(cls) -> ndarray[float]:
            """Returns the coordinates of the center of the master cell as a NumPy array."""
            return np.array([[0.0, 0.0, 0.0]], dtype=float)

        @classmethod
        def tetmap(cls) -> ndarray[int]:
            """Returns a mapping in the form of a NumPy array to convert a
            single cell to 4-node tetrahedra."""
            w18_to_w6 = np.array(
                [
                    [15, 13, 16, 9, 4, 10],
                    [17, 16, 14, 11, 10, 5],
                    [17, 15, 16, 11, 9, 10],
                    [12, 15, 17, 3, 9, 11],
                    [6, 1, 7, 15, 13, 16],
                    [8, 6, 7, 17, 15, 16],
                    [8, 7, 2, 17, 16, 14],
                    [8, 0, 6, 17, 12, 15],
                ],
                dtype=int,
            )
            w6_to_tet4 = W6.Geometry.tetmap()
            return compose_trmap(w18_to_w6, w6_to_tet4)
