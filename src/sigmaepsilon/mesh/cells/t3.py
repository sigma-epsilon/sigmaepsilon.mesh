# -*- coding: utf-8 -*-
from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..geometry import PolyCellGeometry2d
from ..data.polycell import PolyCell
from ..utils.cells.numint import Gauss_Legendre_Tri_1
from ..utils.cells.t3 import (
    shp_T3_multi,
    dshp_T3_multi,
    shape_function_matrix_T3_multi,
    monoms_T3,
)
from ..utils.utils import points_of_cells
from ..utils.tri import area_tri_bulk


class T3(PolyCell):
    """
    A class to handle 3-noded triangles.
    """

    label = "T3"

    class Geometry(PolyCellGeometry2d):
        number_of_nodes = 3
        vtk_cell_id = 5
        shape_function_evaluator: shp_T3_multi
        shape_function_matrix_evaluator: shape_function_matrix_T3_multi
        shape_function_derivative_evaluator: dshp_T3_multi
        monomial_evaluator: monoms_T3
        quadrature = {
            "full": Gauss_Legendre_Tri_1(),
        }

        @classmethod
        def trimap(cls) -> ndarray:
            return np.array([[0, 1, 2]], dtype=int)

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
            monoms = [1, r, s]
            return locvars, monoms

        @classmethod
        def master_coordinates(cls) -> ndarray:
            """
            Returns local coordinates of the cell.

            Returns
            -------
            numpy.ndarray
            """
            return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        @classmethod
        def master_center(cls) -> ndarray:
            """
            Returns the local coordinates of the center of the cell.

            Returns
            -------
            numpy.ndarray
            """
            return np.array([[1 / 3, 1 / 3]])

    def to_triangles(self) -> ndarray:
        return self.topology().to_numpy()

    def areas(self, *args, **kwargs) -> ndarray:
        coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        ec = points_of_cells(coords, topo, local_axes=self.frames)
        return area_tri_bulk(ec)

    @classmethod
    def from_TriMesh(cls, *args, coords=None, topo=None, **kwargs):
        from sigmaepsilon.mesh.data.trimesh import TriMesh

        if len(args) > 0 and isinstance(args[0], TriMesh):
            mesh = args[0]
            return mesh.coords(), mesh.topology().to_numpy()
        elif coords is not None and topo is not None:
            return coords, topo
        else:
            raise NotImplementedError
