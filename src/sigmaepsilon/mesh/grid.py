from typing import Tuple, Union, Iterable

import numpy as np
from numpy import ndarray
from numba import njit, prange

from .utils.topology import unique_topo_data, detach_mesh_bulk, transform_topology
from .utils import (
    center_of_points,
    k_nearest_neighbours as knn,
    knn_to_lines,
    coords_to_3d,
)

__cache = True


__all__ = ["grid", "gridQ4", "gridQ8", "gridQ9", "gridH8", "gridH27", "knngridL2"]


def grid(
    size: Tuple[float] = None,
    shape: Union[int, Tuple[int]] = None,
    eshape: Union[str, Tuple[int]] = None,
    shift: Iterable = None,
    start: int = 0,
    bins: Iterable = None,
    centralize: bool = False,
    path: Iterable = None,
    **kwargs,
) -> Tuple[ndarray, ndarray]:
    """
    Crates a 1d, 2d or 3d grid for different patterns and returnes the raw data.
    If you want a more high level mesh object, consider using the :class:`~sigmaepsilon.mesh.grid.Grid`
    class, which calls this method to generate a :class:`~sigmaepsilon.mesh.data.polydata.PolyData` instance.

    Parameters
    ----------
    size: Tuple[float], Optional
        A 2-tuple, containg side lengths of a rectangular domain.
        Should be provided alongside `shape`. Default is None.
    shape: Union[int, Tuple[int]], Optional
        An integer or a tuple of integers, describing number of cells along
        coordinate directions. Should be provided alongside `size`. Default is None.
    eshape: str or Tuple, Optional
        This specifies element shape.
        Supported strings are thw following:
          - 'Q4' : 4-noded quadrilateral
          - 'Q9' : 9-noded quadrilateral
          - 'H8' : 8-noded hexagon
          - 'H27' : 27-noded hexagon
    shift: numpy.ndarray, Optional
        1d float array, specifying a translation.
    start: int, Optional
        Starting value for node numbering. Default is 0.
    bins: numpy.ndarray, Optional
        Numpy array or an iterable of numy arrays.
    centralize: bool, Optional
        If True, the returned coordinates are centralized.
    path: Iterable, Optional
        A 1d iterable used to rewire the nodes of the resulting grid. If provided, the
        j-th node of the i-th cell becomes ``topo[i, path[j]]``. Default is None.

    Notes
    -----
    1) The returned topology may not be compliant with vtk. If you want to use
    the results of this call to build a vtk model, you have to account for this
    (for instance through the 'path' parameter). Optionally, you can use the dedicated
    grid generation routines of this module.
    2) If you'd rather get the result as a `PolyData`, use the `Grid` class.

    Returns
    -------
    numpy.ndarray
        A numpy float array of coordinates.
    numpy.ndarray
        A numpy integer array describing the topology.

    Examples
    --------
    Create a simple hexagonal mesh of 8 x 6 x 2 cells

    >>> from sigmaepsilon.mesh import grid
    >>> size = 80, 60, 20
    >>> shape = 8, 6, 2
    >>> mesh = (coords, topo) = grid(size=size, shape=shape, eshape='H8')

    Create a mesh of 4-noded quadrilaterals

    >>> gridparams = {
    ...     'size' : (1200, 600),
    ...     'shape' : (30, 15),
    ...     'eshape' : (2, 2),
    ...     'start' : 0
    ... }
    >>> coordsQ4, topoQ4 = grid(**gridparams)

    The same mesh with 6-noded quadrialterals, 2 in x direction, 3 in y direction

    >>> gridparams['eshape'] = (2, 3)
    >>> coordsQ4, topoQ4 = grid(**gridparams)

    See also
    --------
    :class:`~sigmaepsilon.mesh.grid.Grid`
    :func:`~sigmaepsilon.mesh.grid.gridQ4`
    :func:`~sigmaepsilon.mesh.grid.gridQ9`
    :func:`~sigmaepsilon.mesh.grid.gridH8`
    :func:`~sigmaepsilon.mesh.grid.gridH27`
    """
    if size is not None:
        nDime = len(size)
    elif bins is not None:
        nDime = len(bins)
    elif shape is not None:
        nDime = len(shape)

    if shift is None:
        shift = np.zeros(nDime)

    if eshape is None:
        eshape = np.full(nDime, 2, dtype=int)
    elif isinstance(eshape, str):
        if eshape == "Q4":
            return gridQ4(
                size=size,
                shape=shape,
                shift=shift,
                start=start,
                centralize=centralize,
                bins=bins,
                **kwargs,
            )
        elif eshape == "Q8":
            return gridQ8(
                size=size,
                shape=shape,
                shift=shift,
                start=start,
                centralize=centralize,
                bins=bins,
                **kwargs,
            )
        elif eshape == "Q9":
            return gridQ9(
                size=size,
                shape=shape,
                shift=shift,
                start=start,
                centralize=centralize,
                bins=bins,
                **kwargs,
            )
        elif eshape == "H8":
            return gridH8(
                size=size,
                shape=shape,
                shift=shift,
                start=start,
                centralize=centralize,
                bins=bins,
                **kwargs,
            )
        elif eshape == "H27":
            return gridH27(
                size=size,
                shape=shape,
                shift=shift,
                start=start,
                centralize=centralize,
                bins=bins,
                **kwargs,
            )
        else:
            raise NotImplementedError

    if bins is not None:
        if len(bins) == 2:
            coords, topo = grid_2d_bins(bins[0], bins[1], eshape, shift, start)
        elif len(bins) == 3:
            coords, topo = grid_3d_bins(bins[0], bins[1], bins[2], eshape, shift, start)
        else:
            raise NotImplementedError
    else:
        assert isinstance(eshape, tuple)
        if shape is None:
            shape = np.full(nDime, 1, dtype=int)
        coords, topo = rgridMT(size, shape, eshape, shift, start)

    if centralize:
        center = center_of_points(coords)
        coords[:, 0] -= center[0]
        coords[:, 1] -= center[1]
        if center.shape[0] > 2:
            coords[:, 2] -= center[2]

    if path is not None:
        if not isinstance(path, Iterable):
            raise TypeError("Argument 'path' must be an Iterable!")
        path = np.array(path, dtype=int)
        topo = transform_topology(topo, path)

    return coords_to_3d(coords), topo


def gridQ4(*args, **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Customized version of :func:`grid` dedicated to Q4 elements.
    It returns a topology with vtk-compliant node numbering.

    In terms of parameters, this function have to be called the
    same way `grid` would be called, except the parameter
    `eshape` being obsolete.

    Example
    -------
    Creating a mesh of 4-noded quadrilaterals

    >>> from sigmaepsilon.mesh.grid import gridQ4
    >>> gridparams = {
    ...     'size' : (1200, 600),
    ...     'shape' : (30, 15),
    ...     'origo' : (0, 0),
    ...     'start' : 0
    ... }
    >>> coordsQ4, topoQ4 = gridQ4(**gridparams)
    """
    coords, topo = grid(*args, eshape=(2, 2), **kwargs)
    path = np.array([0, 2, 3, 1], dtype=int)
    return coords_to_3d(coords), transform_topology(topo, path)


def gridQ9(*args, **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Customized version of `grid` dedicated to Q9 elements.
    It returns a topology with vtk-compliant node numbering.

    In terms of parameters, this function have to be called the
    same way `grid` would be called, except the parameter
    `eshape` being obsolete.
    """
    coords, topo = grid(*args, eshape=(3, 3), **kwargs)
    path = np.array([0, 6, 8, 2, 3, 7, 5, 1, 4], dtype=int)
    return coords_to_3d(coords), transform_topology(topo, path)


def gridQ8(*args, **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Customized version of `grid` dedicated to Q8 elements.
    It returns a topology with vtk-compliant node numbering.

    In terms of parameters, this function have to be called the
    same way `grid` would be called, except the parameter
    `eshape` being obsolete.
    """
    coords, topo = gridQ9(*args, **kwargs)
    return detach_mesh_bulk(coords, topo[:, :-1])


def gridH8(*args, **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Customized version of `grid` dedicated to H8 elements.
    It returns a topology with vtk-compliant node numbering.

    In terms of parameters, this function have to be called the
    same way `grid` would be called, except the parameter
    `eshape` being obsolete.
    """
    coords, topo = grid(*args, eshape=(2, 2, 2), **kwargs)
    path = np.array([0, 4, 6, 2, 1, 5, 7, 3], dtype=int)
    return coords_to_3d(coords), transform_topology(topo, path)


# fmt: off
def gridH27(*args, **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Customized version of `grid` dedicated to H27 elements.
    It returns a topology with vtk-compliant node numbering.

    In terms of parameters, this function have to be called the
    same way `grid` would be called, except the parameter
    `eshape` being obsolete.
    """
    coords, topo = grid(*args, eshape=(3, 3, 3), **kwargs)
    path = np.array(
        [
            0, 18, 24, 6, 2, 20, 26, 8, 9, 21,
            15, 3, 11, 23, 17, 5, 1, 19, 25, 7,
            4, 22, 10, 16, 12, 14, 13,
        ],
        dtype=int,
    )
    return coords, transform_topology(topo, path)
# fmt: on


@njit(nogil=True, cache=__cache)
def _rgridST(size, shape, eshape, shift, start=0):
    """
    Legacy code for single-thread implementation, for educational purpose only.
    Use rgridMT.
    """
    nDime = len(size)

    if nDime == 1:
        lX = size
        ndivX = shape
        nNodeX = eshape
        nX = ndivX * (nNodeX - 1) + 1
        dX = lX / ndivX
        ddX = dX / (nNodeX - 1)
        numCell = ndivX
        numPoin = nX
        numNode = nNodeX
    elif nDime == 2:
        lX, lY = size
        ndivX, ndivY = shape
        nNodeX, nNodeY = eshape
        nX = ndivX * (nNodeX - 1) + 1
        nY = ndivY * (nNodeY - 1) + 1
        dX = lX / ndivX
        dY = lY / ndivY
        ddX = dX / (nNodeX - 1)
        ddY = dY / (nNodeY - 1)
        numCell = ndivX * ndivY
        numPoin = nX * nY
        numNode = nNodeX * nNodeY
    elif nDime == 3:
        lX, lY, lZ = size
        ndivX, ndivY, ndivZ = shape
        nNodeX, nNodeY, nNodeZ = eshape
        nX = ndivX * (nNodeX - 1) + 1
        nY = ndivY * (nNodeY - 1) + 1
        nZ = ndivZ * (nNodeZ - 1) + 1
        dX = lX / ndivX
        dY = lY / ndivY
        dZ = lZ / ndivZ
        ddX = dX / (nNodeX - 1)
        ddY = dY / (nNodeY - 1)
        ddZ = dZ / (nNodeZ - 1)
        numCell = ndivY * ndivX * ndivZ
        numPoin = nX * nY * nZ
        numNode = nNodeX * nNodeY * nNodeZ

    # set up nodal coordinates
    coords = np.zeros((numPoin, nDime))
    topo = np.zeros((numCell, numNode), dtype=np.int64)
    elem = -1
    if nDime == 1:
        for i in range(1, ndivX + 1):
            elem += 1
            ne = -1
            n = (nNodeX - 1) * (i - 1)
            for j in range(1, nNodeX + 1):
                n += 1
                coords[n - 1, 0] = shift + dX * (i - 1) + ddX * (j - 1)
                ne += 1
                topo[elem, ne] = n
    elif nDime == 2:
        for i in range(1, ndivX + 1):
            for j in range(1, ndivY + 1):
                elem += 1
                ne = -1
                for k in range(1, nNodeX + 1):
                    for m in range(1, nNodeY + 1):
                        n = (
                            ((nNodeX - 1) * (i - 1) + k - 1) * nY
                            + (nNodeY - 1) * (j - 1)
                            + m
                        )
                        coords[n - 1, 0] = shift[0] + dX * (i - 1) + ddX * (k - 1)
                        coords[n - 1, 1] = shift[1] + dY * (j - 1) + ddY * (m - 1)
                        ne += 1
                        topo[elem, ne] = n
    elif nDime == 3:
        for i in range(1, ndivX + 1):
            for j in range(1, ndivY + 1):
                for k in range(1, ndivZ + 1):
                    elem += 1
                    ne = -1
                    for m in range(1, nNodeX + 1):
                        for q in range(1, nNodeY + 1):
                            for p in range(1, nNodeZ + 1):
                                n = (
                                    (
                                        ((nNodeX - 1) * (i - 1) + m - 1) * nY
                                        + (nNodeY - 1) * (j - 1)
                                        + q
                                        - 1
                                    )
                                    * nZ
                                    + (nNodeZ - 1) * (k - 1)
                                    + p
                                )
                                coords[n - 1, 0] = (
                                    shift[0] + dX * (i - 1) + ddX * (m - 1)
                                )
                                coords[n - 1, 1] = (
                                    shift[1] + dY * (j - 1) + ddY * (q - 1)
                                )
                                coords[n - 1, 2] = (
                                    shift[2] + dZ * (k - 1) + ddZ * (p - 1)
                                )
                                ne += 1
                                topo[elem, ne] = n
    start -= 1
    return coords, topo + start


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def rgridMT(size, shape, eshape, shift, start=0):
    nDime = len(size)
    if nDime == 1:
        lX = size[0]
        ndivX = shape[0]
        nNodeX = eshape[0]
        nX = ndivX * (nNodeX - 1) + 1
        dX = lX / ndivX
        ddX = dX / (nNodeX - 1)
        numCell = ndivX
        numPoin = nX
        numNode = nNodeX
    elif nDime == 2:
        lX, lY = size
        ndivX, ndivY = shape
        nNodeX, nNodeY = eshape
        nX = ndivX * (nNodeX - 1) + 1
        nY = ndivY * (nNodeY - 1) + 1
        dX = lX / ndivX
        dY = lY / ndivY
        ddX = dX / (nNodeX - 1)
        ddY = dY / (nNodeY - 1)
        numCell = ndivX * ndivY
        numPoin = nX * nY
        numNode = nNodeX * nNodeY
    elif nDime == 3:
        lX, lY, lZ = size
        ndivX, ndivY, ndivZ = shape
        nNodeX, nNodeY, nNodeZ = eshape
        nX = ndivX * (nNodeX - 1) + 1
        nY = ndivY * (nNodeY - 1) + 1
        nZ = ndivZ * (nNodeZ - 1) + 1
        dX = lX / ndivX
        dY = lY / ndivY
        dZ = lZ / ndivZ
        ddX = dX / (nNodeX - 1)
        ddY = dY / (nNodeY - 1)
        ddZ = dZ / (nNodeZ - 1)
        numCell = ndivY * ndivX * ndivZ
        numPoin = nX * nY * nZ
        numNode = nNodeX * nNodeY * nNodeZ
    # set up nodal coordinates
    coords = np.zeros((numPoin, nDime))
    topo = np.zeros((numCell, numNode), dtype=np.int64)
    elem = -1
    if nDime == 1:
        for i in prange(1, ndivX + 1):
            elem = i - 1
            for j in prange(1, nNodeX + 1):
                n = (nNodeX - 1) * (i - 1) + j
                ne = j - 1
                coords[n - 1, 0] = shift[0] + dX * (i - 1) + ddX * (j - 1)
                topo[elem, ne] = n
    elif nDime == 2:
        for i in prange(1, ndivX + 1):
            for j in prange(1, ndivY + 1):
                elem = (i - 1) * ndivY + j - 1
                for k in prange(1, nNodeX + 1):
                    for m in prange(1, nNodeY + 1):
                        n = (
                            ((nNodeX - 1) * (i - 1) + k - 1) * nY
                            + (nNodeY - 1) * (j - 1)
                            + m
                        )
                        ne = (k - 1) * nNodeY + m - 1
                        coords[n - 1, 0] = shift[0] + dX * (i - 1) + ddX * (k - 1)
                        coords[n - 1, 1] = shift[1] + dY * (j - 1) + ddY * (m - 1)
                        topo[elem, ne] = n
    elif nDime == 3:
        for i in prange(1, ndivX + 1):
            for j in prange(1, ndivY + 1):
                for k in prange(1, ndivZ + 1):
                    elem = (i - 1) * ndivY * ndivZ + (j - 1) * ndivZ + k - 1
                    for m in prange(1, nNodeX + 1):
                        for q in prange(1, nNodeY + 1):
                            for p in prange(1, nNodeZ + 1):
                                n = (
                                    (
                                        ((nNodeX - 1) * (i - 1) + m - 1) * nY
                                        + (nNodeY - 1) * (j - 1)
                                        + q
                                        - 1
                                    )
                                    * nZ
                                    + (nNodeZ - 1) * (k - 1)
                                    + p
                                )
                                ne = (
                                    (m - 1) * nNodeY * nNodeZ + (q - 1) * nNodeZ + p - 1
                                )
                                coords[n - 1, 0] = (
                                    shift[0] + dX * (i - 1) + ddX * (m - 1)
                                )
                                coords[n - 1, 1] = (
                                    shift[1] + dY * (j - 1) + ddY * (q - 1)
                                )
                                coords[n - 1, 2] = (
                                    shift[2] + dZ * (k - 1) + ddZ * (p - 1)
                                )
                                topo[elem, ne] = n
    start -= 1
    return coords, topo + start


@njit(nogil=True, parallel=True, cache=__cache)
def grid_2d_bins(xbins, ybins, eshape, shift, start=0):
    # shape of cells
    nEX = len(xbins) - 1
    nEY = len(ybins) - 1
    # number of cells
    nC = nEX * nEY
    # number of nodes
    nNEX, nNEY = eshape
    nDIVX = nNEX - 1
    nDIVY = nNEY - 1
    nNE = nNEX * nNEY
    nNX = nEX * nDIVX + 1
    nNY = nEY * nDIVY + 1
    nN = nNX * nNY
    # create grid
    coords = np.zeros((nN, 2))
    topo = np.zeros((nC, nNE), dtype=np.int64)
    for i in prange(nEX):
        ddX = (xbins[i + 1] - xbins[i]) / nDIVX
        for j in prange(nEY):
            ddY = (ybins[j + 1] - ybins[j]) / nDIVY
            iE = i * nEY + j
            for p in prange(nNEX):
                for q in prange(nNEY):
                    n = (nDIVX * i + p) * nNY + nDIVY * j + q
                    coords[n, 0] = shift[0] + xbins[i] + ddX * p
                    coords[n, 1] = shift[1] + ybins[j] + ddY * q
                    iNE = p * nNEY + q
                    topo[iE, iNE] = n
    return coords, topo + start


@njit(nogil=True, parallel=True, cache=__cache)
def grid_3d_bins(xbins, ybins, zbins, eshape, shift, start=0):
    # size
    lX = xbins.max() - xbins.min()
    lY = ybins.max() - ybins.min()
    lZ = zbins.max() - zbins.min()
    # shape of cells
    nEX = len(xbins) - 1
    nEY = len(ybins) - 1
    nEZ = len(zbins) - 1
    # number of cells
    nC = nEX * nEY * nEZ
    # number of nodes
    nNEX, nNEY, nNEZ = eshape
    nDIVX = nNEX - 1
    nDIVY = nNEY - 1
    nDIVZ = nNEZ - 1
    nNE = nNEX * nNEY * nNEZ
    nNX = nEX * nDIVX + 1
    nNY = nEY * nDIVY + 1
    nNZ = nEZ * nDIVZ + 1
    nN = nNX * nNY * nNZ
    # create grid
    coords = np.zeros((nN, 3), dtype=xbins.dtype)
    topo = np.zeros((nC, nNE), dtype=np.int64)
    for i in prange(nEX):
        ddX = (xbins[i + 1] - xbins[i]) / nDIVX
        for j in prange(nEY):
            ddY = (ybins[j + 1] - ybins[j]) / nDIVY
            for k in prange(nEZ):
                ddZ = (zbins[k + 1] - zbins[k]) / nDIVZ
                iE = i * nEZ * nEY + j * nEZ + k
                for p in prange(nNEX):
                    for q in prange(nNEY):
                        for r in prange(nNEZ):
                            n = (
                                ((nDIVX * i + p) * nNY + nDIVY * j + q) * nNZ
                                + nDIVZ * k
                                + r
                            )
                            coords[n, 0] = shift[0] + xbins[i] + ddX * p
                            coords[n, 1] = shift[1] + ybins[j] + ddY * q
                            coords[n, 2] = shift[2] + zbins[k] + ddZ * r
                            iNE = p * nNEZ * nNEY + q * nNEZ + r
                            topo[iE, iNE] = n
    return coords, topo + start


def knngridL2(
    *args, max_distance: float = None, k: int = 3, X: ndarray = None, **kwargs
) -> Tuple[ndarray, ndarray]:
    """
    Returns a KNN grid of L2 lines. First a grid of points is created
    using :func:``grid``, then points are connected based on a KNN-tree.

    Parameters
    ----------
    *args: tuple, Optional
        Positional arguments forwarded to :func:``grid``.
    max_distance: float, Optional
        Maximum distance allowed. Default is None.
    k: int, Optional
        Number of neighbours for a given point.
    X: numpy.ndarray, Optional
        Coordinates of a pointcloud. If provided, `args` and `kwargs`
        are ignored. Default is None.
    **kwargs: dict, Optional
        Keyword arguments forwarded to :func:``grid``.

    See also
    --------
    :func:`~sigmaepsilon.mesh.utils.knn.k_nearest_neighbours`
    :func:`~sigmaepsilon.mesh.utils.knn.knn_to_lines`
    """
    if X is None:
        X, _ = grid(*args, **kwargs)
    i = knn(X, X, k=k, max_distance=max_distance)
    T, _ = unique_topo_data(knn_to_lines(i))
    di = T[:, 0] - T[:, -1]
    inds = np.where(di != 0)[0]
    return detach_mesh_bulk(X, T[inds, :])
