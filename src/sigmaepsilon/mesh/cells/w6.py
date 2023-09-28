from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..geometry import PolyCellGeometry3d
from ..data.polycell import PolyCell
from ..utils.cells.numint import Gauss_Legendre_Wedge_3x2
from ..utils.cells.utils import volumes
from ..utils.utils import cells_coords
from ..utils.cells.w6 import monoms_W6


class W6(PolyCell):
    """
    Polyhedra class for 6-noded trilinear wedges.
    """

    label = "W6"

    class Geometry(PolyCellGeometry3d):
        number_of_nodes = 6
        vtk_cell_id = 13
        monomial_evaluator: monoms_W6
        quadrature = {
            "full": Gauss_Legendre_Wedge_3x2(),
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
            monoms = [1, r, s, t, r * t, s * t]
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
                ]
            )

        @classmethod
        def master_center(cls) -> ndarray:
            return np.array([[1 / 3, 1 / 3, 0]])

        @classmethod
        def tetmap(cls) -> np.ndarray:
            return np.array(
                [[0, 1, 2, 4], [3, 5, 4, 2], [2, 5, 0, 4]],
                dtype=int,
            )

    def volumes(self) -> ndarray:
        coords = self.source_coords()
        topo = self.topology().to_numpy()
        ecoords = cells_coords(coords, topo)
        qpos, qweight = self.Geometry.quadrature["full"]
        dshp = self.Geometry.shape_function_derivatives(qpos)
        return volumes(ecoords, dshp, qweight)
