# -*- coding: utf-8 -*-
from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..geometry import PolyCellGeometry2d
from ..data.polycell import PolyCell
from ..utils.utils import cells_coords
from ..utils.cells.t6 import (
    shp_T6_multi,
    dshp_T6_multi,
    areas_T6,
    shape_function_matrix_T6_multi,
    monoms_T6,
)
from ..utils.cells.numint import Gauss_Legendre_Tri_3a
from ..utils.topology import T6_to_T3, T3_to_T6


class T6(PolyCell):
    """
    A class to handle 6-noded triangles.
    """

    label = "T6"

    class Geometry(PolyCellGeometry2d):
        number_of_nodes = 6
        vtk_cell_id = 22
        shape_function_evaluator: shp_T6_multi
        shape_function_matrix_evaluator: shape_function_matrix_T6_multi
        shape_function_derivative_evaluator: dshp_T6_multi
        monomial_evaluator: monoms_T6
        quadrature = {
            "full": Gauss_Legendre_Tri_3a(),
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
            locvars = r, s = symbols("r s", real=True)
            monoms = [1, r, s, r ** 2, s ** 2, r * s]
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
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
            )

        @classmethod
        def master_center(cls) -> ndarray:
            """
            Returns the local coordinates of the center of the cell.

            Returns
            -------
            numpy.ndarray
            """
            return np.array([[1 / 3, 1 / 3]])

        @classmethod
        def trimap(cls, subdivide: bool = True):
            if subdivide:
                return np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [5, 3, 4]], dtype=int)
            else:
                return np.array([[0, 1, 2]], dtype=int)

    def to_triangles(self) -> ndarray:
        return T6_to_T3(None, self.topology().to_numpy())[1]

    def areas(self) -> ndarray:
        """
        Returns the areas of the triangles of the block.

        Returns
        -------
        numpy.ndarray
        """
        coords = self.source_coords()
        topo = self.topology().to_numpy()
        ecoords = cells_coords(coords[:, :2], topo)
        qpos, qweight = self.Geometry.quadrature["full"]
        return areas_T6(ecoords, qpos, qweight)

    @classmethod
    def from_TriMesh(cls, *args, coords=None, topo=None, **kwargs):
        from sigmaepsilon.mesh.data.trimesh import TriMesh

        if len(args) > 0 and isinstance(args[0], TriMesh):
            return T3_to_T6(TriMesh.coords(), TriMesh.topology())
        elif coords is not None and topo is not None:
            return T3_to_T6(coords, topo)
        else:
            raise NotImplementedError
