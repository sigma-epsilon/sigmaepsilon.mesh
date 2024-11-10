# -*- coding: utf-8 -*-
from typing import Tuple, List

import numpy as np
from numpy import ndarray
from sympy import symbols

from ..geometry import PolyCellGeometry3d
from ..data.polycell import PolyCell
from ..utils.cells.tet4 import (
    shp_TET4_multi,
    dshp_TET4_multi,
    shape_function_matrix_TET4_multi,
    monoms_TET4,
)
from ..utils.numint import Gauss_Legendre_Tet_1
from ..utils.tet import vol_tet_bulk
from ..utils.utils import cells_coords


class TET4(PolyCell):
    """
    Class for 4-node tetrahedra.
    """

    label = "TET4"

    class Geometry(PolyCellGeometry3d):
        number_of_nodes = 4
        vtk_cell_id = 10
        shape_function_evaluator = shp_TET4_multi
        shape_function_matrix_evaluator = shape_function_matrix_TET4_multi
        shape_function_derivative_evaluator = dshp_TET4_multi
        monomial_evaluator = monoms_TET4
        quadrature = {
            "full": Gauss_Legendre_Tet_1,
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
            locvars = r, s, t = symbols("r s t", real=True)
            monoms = [1, r, s, t]
            return locvars, monoms

        @classmethod
        def master_coordinates(cls) -> ndarray:
            return np.array(
                [
                    [-1 / 3, -1 / 3, -1 / 3],
                    [2 / 3, -1 / 3, -1 / 3],
                    [-1 / 3, 2 / 3, -1 / 3],
                    [-1 / 3, -1 / 3, 2 / 3],
                ]
            )

        @classmethod
        def master_center(cls) -> ndarray:
            return np.array([[0.0, 0.0, 0.0]], dtype=float)

        @classmethod
        def tetmap(cls) -> ndarray:
            return np.array([[0, 1, 2, 3]], dtype=int)

    def to_tetrahedra(self, flatten: bool = True) -> ndarray:
        tetra = self.topology().to_numpy()
        if flatten:
            return tetra
        else:
            return tetra.reshape(len(tetra), 1, 4)

    def volumes(self) -> ndarray:
        coords = self.source_coords()
        topo = self.topology().to_numpy()
        volumes = vol_tet_bulk(cells_coords(coords, topo))
        res = np.sum(
            volumes.reshape(topo.shape[0], int(len(volumes) / topo.shape[0])),
            axis=1,
        )
        return res
