# -*- coding: utf-8 -*-
import numpy as np

from .polygon import Triangle
from .cell import PolyCell3d
from .utils.topology import compose_trmap


class PolyHedron(PolyCell3d):
    """Base class to handle polyhedra."""

    _face_cls_ = Triangle

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TetraHedron(PolyHedron):
    """Class for 4-noded tetrahedra."""

    NNODE = 4
    vtkCellType = 10
    __label__ = "TET4"

    @classmethod
    def tetmap(cls) -> np.ndarray:
        return np.array([[0, 1, 2, 3]], dtype=int)

    def to_tetrahedra(self, flatten: bool = True) -> np.ndarray:
        tetra = self.topology().to_numpy()
        if flatten:
            return tetra
        else:
            return tetra.reshape(len(tetra), 1, 4)


class QuadraticTetraHedron(PolyHedron):
    """Class for 10-noded quadratic tetrahedra."""

    NNODE = 10
    vtkCellType = 24
    __label__ = "TET10"

    @classmethod
    def tetmap(cls, subdivide: bool = True) -> np.ndarray:
        if subdivide:
            raise NotImplementedError
        else:
            return np.array([[0, 1, 2, 3]], dtype=int)


class HexaHedron(PolyHedron):
    """Class for 8-noded hexahedra."""

    NNODE = 8
    vtkCellType = 12
    __label__ = "H8"

    @classmethod
    def tetmap(cls) -> np.ndarray:
        return np.array(
            [[1, 2, 0, 5], [3, 0, 2, 7], [5, 4, 7, 0], [6, 5, 7, 2], [0, 2, 7, 5]],
            dtype=int,
        )


class TriquadraticHexaHedron(PolyHedron):
    """Class for 27-noded triquadratic hexahedra."""

    NNODE = 27
    vtkCellType = 29
    __label__ = "H27"


class Wedge(PolyHedron):
    """Class for 6-noded trilinear wedges."""

    NNODE = 6
    vtkCellType = 13
    __label__ = "W6"

    @classmethod
    def tetmap(cls) -> np.ndarray:
        return np.array(
            [[0, 1, 2, 4], [3, 5, 4, 2], [2, 5, 0, 4]],
            dtype=int,
        )


class BiquadraticWedge(PolyHedron):
    """Class for 6-noded biquadratic wedges."""

    NNODE = 18
    vtkCellType = 32
    __label__ = "W18"

    @classmethod
    def tetmap(cls) -> np.ndarray:
        w18_to_w6 = np.array(
            [
                [15, 13, 16, 9, 4, 10],
                [17, 16, 14, 11, 10, 5],
                [17, 15, 16, 11, 9, 10],
                [12, 15, 17, 3, 9, 11],
                [6, 1, 7, 15, 13, 16],
                [8, 6, 7, 17, 15, 16],
                [8, 7, 2, 17, 16, 14],
                [8, 0, 6, 17, 12, 15],
            ],
            dtype=int,
        )
        w6_to_tet4 = Wedge.tetmap()
        return compose_trmap(w18_to_w6, w6_to_tet4)
