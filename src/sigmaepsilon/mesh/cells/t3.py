# -*- coding: utf-8 -*-
from typing import Tuple, List

import numpy as np
from numpy import ndarray
from sympy import symbols

from ..geometry import PolyCellGeometry2d
from ..data.polycell import PolyCell
from ..utils.numint import Gauss_Legendre_Tri_1
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

    Example
    -------
    >>> from sigmaepsilon.mesh import TriMesh, CartesianFrame, PointData, triangulate
    >>> from sigmaepsilon.mesh.cells import T3 as CellData
    >>> A = CartesianFrame(dim=3)
    >>> coords, topo = triangulate(size=(800, 600), shape=(10, 10))
    >>> pd = PointData(coords=coords, frame=A)
    >>> cd = CellData(topo=topo)
    >>> trimesh = TriMesh(pd, cd)
    >>> trimesh.area()
    480000.0
    """

    label = "T3"

    class Geometry(PolyCellGeometry2d):
        number_of_nodes = 3
        vtk_cell_id = 5
        shape_function_evaluator = shp_T3_multi
        shape_function_matrix_evaluator = shape_function_matrix_T3_multi
        shape_function_derivative_evaluator = dshp_T3_multi
        monomial_evaluator = monoms_T3
        quadrature = {
            "full": Gauss_Legendre_Tri_1,
            "geometry": "full",
        }

        @classmethod
        def trimap(cls) -> ndarray:
            """
            Returns a mapping used to transform the topology to triangles.
            This is only implemented here for standardization.
            """
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
            Returns local coordinates of the master cell relative to the origo
            of the master cell.

            Returns
            -------
            numpy.ndarray
            """
            return np.array([[-1 / 3, -1 / 3], [2 / 3, -1 / 3], [-1 / 3, 2 / 3]])

        @classmethod
        def master_center(cls) -> ndarray:
            """
            Returns the center of the master cell relative to the origo
            of the master cell.

            Returns
            -------
            numpy.ndarray
            """
            return np.array([[0.0, 0.0]], dtype=float)

    def to_triangles(self) -> ndarray:
        """
        Returns the topology as triangles.
        """
        return self.topology().to_numpy()

    def areas(self, *_, **__) -> ndarray:
        """
        Returns the areas of the cells as an 1d NumPy array.
        """
        coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        ec = points_of_cells(coords, topo, local_axes=self.frames)
        return area_tri_bulk(ec)

    @classmethod
    def from_TriMesh(
        cls, *args, coords: ndarray = None, topo: ndarray = None, **__
    ) -> Tuple[ndarray, ndarray]:
        from sigmaepsilon.mesh.data.trimesh import TriMesh

        if len(args) > 0 and isinstance(args[0], TriMesh):
            mesh = args[0]
            return mesh.coords(), mesh.topology().to_numpy()
        elif coords is not None and topo is not None:
            return coords, topo
        else:
            raise NotImplementedError
