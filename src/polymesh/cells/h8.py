from typing import Tuple, List

from sympy import symbols
import numpy as np
from numpy import ndarray

from neumann.numint import gauss_points as gp

from ..polyhedron import HexaHedron
from ..utils.utils import cells_coords
from ..utils.cells.h8 import (
    shp_H8_multi,
    dshp_H8_multi,
    volumes_H8,
    shape_function_matrix_H8_multi,
    monoms_H8,
)
from ..utils.cells.gauss import Gauss_Legendre_Hex_Grid


class H8(HexaHedron):
    """
    8-node isoparametric hexahedron.

    ::

        top
        7--6
        |  |
        4--5

        bottom
        3--2
        |  |
        0--1

    See Also
    --------
    :class:`~polymesh.polyhedron.HexaHedron`
    """

    shpfnc = shp_H8_multi
    shpmfnc = shape_function_matrix_H8_multi
    dshpfnc = dshp_H8_multi
    monomsfnc = monoms_H8

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
    def lcoords(cls) -> ndarray:
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
    def lcenter(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([0.0, 0.0, 0.0])

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
