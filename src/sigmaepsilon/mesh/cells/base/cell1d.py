from typing import Union, Iterable

import numpy as np
from numpy import ndarray

from sigmaepsilon.math import atleast1d
from sigmaepsilon.math.utils import to_range_1d

from sigmaepsilon.mesh.space import CartesianFrame
from .cell import PolyCell
from ...utils.utils import (
    points_of_cells,
    pcoords_to_coords_1d,
    lengths_of_lines,
)
from ...space import CartesianFrame


class PolyCell1d(PolyCell):
    """Base class for 1d cells"""

    NDIM = 1

    def lenth(self) -> float:
        """Returns the total length of the cells in the block."""
        return np.sum(self.lengths())

    def lengths(self) -> ndarray:
        """
        Returns the lengths as a NumPy array.
        """
        coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        return lengths_of_lines(coords, topo)

    def area(self) -> float:
        # should return area of the surface of the volume
        raise NotImplementedError

    def areas(self) -> ndarray:
        """
        Returns the areas as a NumPy array.
        """
        areakey = self._dbkey_areas_
        if areakey in self.fields:
            return self[areakey].to_numpy()
        else:
            return np.ones((len(self)))

    def volumes(self) -> ndarray:
        """
        Returns the volumes as a NumPy array.
        """
        return self.lengths() * self.areas()

    def measures(self) -> ndarray:
        return self.lengths()

    def points_of_cells(
        self,
        *,
        points: Union[float, Iterable] = None,
        cells: Union[int, Iterable] = None,
        flatten: bool = False,
        target: Union[str, CartesianFrame] = "global",
        rng: Iterable = None,
        **kwargs,
    ) -> ndarray:
        if isinstance(target, str):
            assert target.lower() in ["global", "g"]
        else:
            raise NotImplementedError
        topo = kwargs.get("topo", self.topology().to_numpy())
        coords = kwargs.get("coords", None)
        if coords is None:
            if self.pointdata is not None:
                coords = self.pointdata.x
            else:
                coords = self.container.source().coords()
        ecoords = points_of_cells(coords, topo, centralize=False)
        if points is None and cells is None:
            return ecoords

        # points or cells is not None
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {}
            ecoords = ecoords[cells]
            topo = topo[cells]
        else:
            cells = np.s_[:]

        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            points = np.array(points)
            rng = np.array([0, 1]) if rng is None else np.array(rng)

        points, rng = to_range_1d(points, source=rng, target=[0, 1]).flatten(), [0, 1]
        res = pcoords_to_coords_1d(points, ecoords)  # (nE * nP, nD)

        if not flatten:
            nE = ecoords.shape[0]
            nP = points.shape[0]
            res = res.reshape(nE, nP, res.shape[-1])  # (nE, nP, nD)

        return res

    @classmethod
    def shape_function_values(
        cls, pcoords: ndarray, *, rng: Iterable = None
    ) -> ndarray:
        """
        Evaluates the shape functions at the specified locations.

        Parameters
        ----------
        pcoords: float or Iterable[float]
            Locations of the evaluation points.
        rng: Iterable, Optional
            The range in which the locations ought to be understood,
            typically [0, 1] or [-1, 1]. Default is [0, 1].

        Returns
        -------
        numpy.ndarray
            An array of shape (nP, nNE) where nP and nNE are the number of
            evaluation points and shape functions. If there is only one
            evaluation point, the returned array is one dimensional.
        """
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords))
        pcoords = to_range_1d(pcoords, source=rng, target=[-1, 1])
        return super().shape_function_values(pcoords)

    @classmethod
    def shape_function_matrix(
        cls, pcoords: ndarray, *, rng: Iterable = None, N: int = None
    ) -> ndarray:
        """
        Evaluates the shape function matrix at the specified locations.

        Parameters
        ----------
        pcoords: float or Iterable[float]
            Locations of the evaluation points.
        rng: Iterable, Optional
            The range in which the locations ought to be understood,
            typically [0, 1] or [-1, 1]. Default is [0, 1].
        N: int, Optional
            Number of unknowns per node.

        Returns
        -------
        numpy.ndarray
            An array of shape (nP, nDOF, nDOF * nNE) where nP, nDOF and nNE
            are the number of evaluation points, degrees of freedom per node
            and nodes per cell.
        """
        rng = np.array([-1, 1]) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords))
        pcoords = to_range_1d(pcoords, source=rng, target=[-1, 1])
        return super().shape_function_matrix(pcoords, N=N)

    @classmethod
    def shape_function_derivatives(
        cls,
        pcoords: Union[float, Iterable[float]] = None,
        *,
        rng: Iterable = None,
        jac: ndarray = None,
        dshp: ndarray = None
    ) -> ndarray:
        """
        Evaluates shape function derivatives wrt. the master element or the local
        coordinate frames of some cells. To control the behaviour, either 'jac' or 'wrt'
        can be provided.

        Parameters
        ----------
        pcoords: float or Iterable[float], Optional
            Locations of the evaluation points.
        rng: Iterable, Optional
            The range in which the locations ought to be understood,
            typically [0, 1] or [-1, 1]. Default is [0, 1].
        jac: Iterable, Optional
            The jacobian matrix as a float array of shape (nE, nP, nD=1, nD=1), evaluated for
            each point in each cell. Default is None.
        dshp: numpy.ndarray, Optional
            Shape function derivatives wrt. the master element. Only relevant if 'jac' is
            provided. The purpose of this argument is to avoid repeated evaluation in situations
            where 'dshp' is required on its own and is already at hand when calling this function.
            Default is None, in which case it is calculated automatically.

        Returns
        -------
        numpy.ndarray
            An array of shape (nP, nNE, nD=1), where nP, nNE and nD are
            the number of evaluation points, nodes and spatial dimensions.
        """
        rng = np.array([-1, 1], dtype=float) if rng is None else np.array(rng)
        pcoords = atleast1d(np.array(pcoords, dtype=float))
        pcoords = to_range_1d(pcoords, source=rng, target=[-1, 1])
        return super().shape_function_derivatives(pcoords, jac=jac, dshp=dshp)