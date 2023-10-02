from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..geometry import PolyCellGeometry3d
from ..data.polycell import PolyCell
from ..utils.cells.tet10 import (
    monoms_TET10,
)
from ..utils.cells.numint import Gauss_Legendre_Tet_4
from ..utils.cells.utils import volumes
from ..utils.utils import cells_coords


class TET10(PolyCell):
    """
    10-node isoparametric hexahedron.
    """

    label = "TET10"

    class Geometry(PolyCellGeometry3d):
        number_of_nodes = 10
        vtk_cell_id = 24
        monomial_evaluator: monoms_TET10
        quadrature = {
            "full": Gauss_Legendre_Tet_4(),
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
            monoms = [1, r, s, t, r * s, r * t, s * t, r ** 2, s ** 2, t ** 2]
            return locvars, monoms

        @classmethod
        def master_coordinates(cls) -> ndarray:
            return np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ]
            )

        @classmethod
        def master_center(cls) -> ndarray:
            return np.array([[1 / 3, 1 / 3, 1 / 3]])

        @classmethod
        def tetmap(cls, subdivide: bool = True) -> np.ndarray:
            if subdivide:
                raise NotImplementedError
            else:
                return np.array([[0, 1, 2, 3]], dtype=int)

    def volumes(self) -> ndarray:
        coords = self.source_coords()
        topo = self.topology().to_numpy()
        ecoords = cells_coords(coords, topo)
        qpos, qweight = self.Geometry.quadrature["full"]
        dshp = self.Geometry.shape_function_derivatives(qpos)
        return volumes(ecoords, dshp, qweight)
