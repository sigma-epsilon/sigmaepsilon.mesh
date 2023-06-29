from typing import Tuple, List
import numpy as np
from numpy import ndarray
from sympy import symbols

from ..polyhedron import Wedge
from ..utils.cells.gauss import Gauss_Legendre_Wedge_3x2
from ..utils.cells.utils import volumes
from ..utils.utils import cells_coords
from ..utils.cells.w6 import monoms_W6


class W6(Wedge):
    """
    Polyhedra class for 6-noded trilinear wedges.

    See Also
    --------
    :class:`~polymesh.polyhedron.Wedge`
    """

    monomsfnc = monoms_W6

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
    def lcoords(cls):
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
    def lcenter(cls):
        return np.array([[1 / 3, 1 / 3, 0]])

    def volumes(self) -> ndarray:
        coords = self.source_coords()
        topo = self.topology().to_numpy()
        ecoords = cells_coords(coords, topo)
        qpos, qweight = self.quadrature["full"]
        dshp = self.shape_function_derivatives(qpos)
        return volumes(ecoords, dshp, qweight)
