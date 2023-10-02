from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..geometry import PolyCellGeometry3d
from ..data.polycell import PolyCell
from ..utils.cells.numint import Gauss_Legendre_Wedge_3x3
from ..utils.cells.utils import volumes
from ..utils.utils import cells_coords
from ..utils.cells.w18 import monoms_W18
from ..utils.topology import compose_trmap
from .w6 import W6


class W18(PolyCell):
    """
    Polyhedra class for 18-noded biquadratic wedges.
    """

    label = "W18"

    class Geometry(PolyCellGeometry3d):
        number_of_nodes = 18
        vtk_cell_id = 32
        monomial_evaluator: monoms_W18
        quadrature = {
            "full": Gauss_Legendre_Wedge_3x3(),
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
            monoms = [
                1,
                r,
                s,
                r ** 2,
                s ** 2,
                r * s,
                t,
                t * r,
                t * s,
                t * r ** 2,
                t * s ** 2,
                t * r * s,
                t ** 2,
                t ** 2 * r,
                t ** 2 * s,
                t ** 2 * r ** 2,
                t ** 2 * s ** 2,
                t ** 2 * r * s,
            ]
            return locvars, monoms

        @classmethod
        def master_coordinates(cls) -> ndarray:
            return np.array(
                [
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [0.0, 1.0, -1.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.5, 0.0, -1.0],
                    [0.5, 0.5, -1.0],
                    [0.0, 0.5, -1.0],
                    [0.5, 0.0, 1.0],
                    [0.5, 0.5, 1.0],
                    [0.0, 0.5, 1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                ]
            )

        @classmethod
        def master_center(cls) -> ndarray:
            return np.array([[1 / 3, 1 / 3, 0]])

        @classmethod
        def tetmap(cls) -> np.ndarray:
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

    def volumes(self) -> ndarray:
        coords = self.source_coords()
        topo = self.topology().to_numpy()
        ecoords = cells_coords(coords, topo)
        qpos, qweight = self.Geometry.quadrature["full"]
        dshp = self.Geometry.shape_function_derivatives(qpos)
        return volumes(ecoords, dshp, qweight)
