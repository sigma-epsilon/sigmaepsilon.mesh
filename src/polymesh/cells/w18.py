from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..polyhedron import BiquadraticWedge
from ..utils.cells.gauss import Gauss_Legendre_Wedge_3x3
from ..utils.cells.utils import volumes
from ..utils.utils import cells_coords
from ..utils.cells.w18 import monoms_W18


class W18(BiquadraticWedge):
    """
    Polyhedra class for 18-noded biquadratic wedges.

    See Also
    --------
    :class:`~polymesh.polyhedron.Wedge`
    """

    monomsfnc = monoms_W18

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
    def lcoords(cls):
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
    def lcenter(cls):
        return np.array([[1 / 3, 1 / 3, 0]])

    def volumes(self) -> ndarray:
        coords = self.source_coords()
        topo = self.topology().to_numpy()
        ecoords = cells_coords(coords, topo)
        qpos, qweight = self.quadrature["full"]
        dshp = self.shape_function_derivatives(qpos)
        return volumes(ecoords, dshp, qweight)
