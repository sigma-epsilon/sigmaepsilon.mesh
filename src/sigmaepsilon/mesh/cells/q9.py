from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from .base.polygon import BiQuadraticQuadrilateral
from ..utils.cells.q9 import (
    shp_Q9_multi,
    dshp_Q9_multi,
    shape_function_matrix_Q9_multi,
    monoms_Q9,
)
from ..utils.cells.numint import Gauss_Legendre_Quad_9


class Q9(BiQuadraticQuadrilateral):
    """
    Polygon class for 9-noded biquadratic quadrilaterals.

    See Also
    --------
    :class:`~sigmaepsilon.mesh.polygon.BiQuadraticQuadrilateral`
    """

    shpfnc = shp_Q9_multi
    shpmfnc = shape_function_matrix_Q9_multi
    dshpfnc = dshp_Q9_multi
    monomsfnc = monoms_Q9

    quadrature = {
        "full": Gauss_Legendre_Quad_9(),
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
        locvars = r, s = symbols("r, s", real=True)
        monoms = [
            1,
            r,
            s,
            r * s,
            r ** 2,
            s ** 2,
            r * s ** 2,
            s * r ** 2,
            s ** 2 * r ** 2,
        ]
        return locvars, monoms

    @classmethod
    def lcoords(cls):
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [0.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, 0.0],
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
        return np.array([0.0, 0.0])
