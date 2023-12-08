from typing import Union, Tuple, Iterable

import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from numba import njit, prange
from numba.typed import Dict as nbDict
from numba import types as nbtypes

from sigmaepsilon.math import matrixform, atleast2d
from sigmaepsilon.math.linalg.sparse import JaggedArray, csr_matrix

__cache = True
nbint64 = nbtypes.int64
nbint64A = nbint64[:]
nbfloat64A = nbtypes.float64[:]


def cells_around(*args, **kwargs) -> Union[JaggedArray, csr_matrix, dict]:
    """
    Alias for :func:`points_around`.
    """
    return points_around(*args, **kwargs)


def points_around(
    points: ndarray,
    r_max: float,
    *,
    frmt: str = "dict",
    MT: bool = True,
    n_max: int = 10,
) -> Union[JaggedArray, csr_matrix, dict]:
    """
    Returns neighbouring points for each entry in `points` that are
    closer than the distance `r_max`. The results are returned in
    diffent formats, depending on the format specifier argument `frmt`.

    Parameters
    ----------
    points: numpy.ndarray
        Coordinates of several points as a 2d float numpy array.
    r_max: float
        Maximum distance.
    n_max: int, Optional
        Maximum number of neighbours. Default is 10.
    frmt: str
        A string specifying the output format. Valid options are
        'jagged', 'csr' and 'dict'.
        See below for the details on the returned object.

    Returns
    -------
    if frmt = 'csr' : sigmaepsilon.math.linalg.sparse.csr.csr_matrix
        A numba-jittable sparse matrix format.

    frmt = 'dict' : numba Dict(int : int[:])

    frmt = 'jagged' : sigmaepsilon.math.linalg.sparse.JaggedArray
        A subclass of `awkward.Array`
    """
    if MT:
        data, widths = _cells_around_MT_(points, r_max, n_max)
    else:
        raise NotImplementedError

    if frmt == "dict":
        return _cells_data_to_dict(data, widths)
    elif frmt == "jagged":
        return _cells_data_to_jagged(data, widths)
    elif frmt == "csr":
        d = _cells_data_to_dict(data, widths)
        data, inds, indptr, shp = _dict_to_spdata(d, widths)
        return csr_matrix(data=data, indices=inds, indptr=indptr, shape=shp)

    raise RuntimeError("Unhandled case!")


@njit(nogil=True, cache=__cache)
def _dict_to_spdata(d: dict, widths: np.ndarray):
    N = int(np.sum(widths))
    nE = len(widths)
    data = np.zeros(N, dtype=np.int64)
    inds = np.zeros_like(data)
    indptr = np.zeros(nE + 1, dtype=np.int64)
    _c = 0
    wmax = 0
    for i in range(len(d)):
        w = widths[i]
        if w > wmax:
            wmax = w
        c_ = _c + w
        data[_c:c_] = d[i]
        inds[_c:c_] = np.arange(w)
        indptr[i + 1] = c_
        _c = c_
    return data, inds, indptr, (nE, wmax)


@njit(nogil=True, cache=__cache)
def _jagged_to_spdata(ja: JaggedArray):
    widths = ja.widths()
    N = int(np.sum(widths))
    nE = len(widths)
    data = np.zeros(N, dtype=np.int64)
    inds = np.zeros_like(data)
    indptr = np.zeros(nE + 1, dtype=np.int64)
    _c = 0
    wmax = 0
    for i in range(len(ja)):
        w = widths[i]
        if w > wmax:
            wmax = w
        c_ = _c + w
        data[_c:c_] = ja[i]
        inds[_c:c_] = np.arange(w)
        indptr[i + 1] = c_
        _c = c_
    return data, inds, indptr, (nE, wmax)


@njit(nogil=True, fastmath=True, cache=__cache)
def _cells_data_to_dict(data: np.ndarray, widths: np.ndarray) -> nbDict:
    dres = dict()
    nE = len(widths)
    for iE in range(nE):
        dres[iE] = data[iE, : widths[iE]]
    return dres


@njit(nogil=True, parallel=True, cache=__cache)
def _flatten_jagged_data(data, widths) -> ndarray:
    nE = len(widths)
    inds = np.zeros(nE + 1, dtype=widths.dtype)
    inds[1:] = np.cumsum(widths)
    res = np.zeros(np.sum(widths))
    for i in prange(nE):
        res[inds[i] : inds[i + 1]] = data[i, : widths[i]]
    return res


def _cells_data_to_jagged(data, widths):
    data = _flatten_jagged_data(data, widths)
    return JaggedArray(data, cuts=widths)


@njit(nogil=True, parallel=True, cache=__cache)
def _cells_around_MT_(
    centers: ndarray, r_max: float, n_max: int = 10
) -> Tuple[ndarray, ndarray]:
    nE = len(centers)
    res = np.zeros((nE, n_max), dtype=np.int64)
    widths = np.zeros(nE, dtype=np.int64)
    for iE in prange(nE):
        inds = np.where(norms(centers - centers[iE]) <= r_max)[0]
        if inds.shape[0] <= n_max:
            res[iE, : inds.shape[0]] = inds
            widths[iE] = len(inds)
        else:
            res[iE, :] = inds[:n_max]
            widths[iE] = n_max
    return res, widths


def points_of_cells(
    coords: ndarray,
    topo: ndarray,
    *,
    local_axes: ndarray = None,
    centralize: bool = True,
    centers: ndarray = None,
) -> ndarray:
    """
    Returns an explicit representation of coordinates of the cells from a
    pointset and a topology. If coordinate frames are provided, the
    coorindates are returned with  respect to those frames.

    Parameters
    ----------
    coords: numpy.ndarray
        2d float array of shape (nP, nD) of vertex coordinates.
        nP : number of points
        nD : number of dimensions of the model space
    topo: numpy.ndarray
        A 2D array of shape (nE, nNE) of vertex indices. The i-th row
        contains the vertex indices of the i-th element.
        nE : number of elements
        nNE : number of nodes per element
    local_axes: numpy.ndarray
        Reference frames as a 3d array of shape (..., 3, 3). A single
        3x3 numpy array or matrices for all elements in 'topo' must be
        provided.
    centralize: bool, Optional
        If True, and 'local_axes' is not None, the local coordinates are
        returned with respect to the geometric center of each element.
    centers: numpy.ndarray
        Centers for all cells. This is to account for different master cells
        with different centers (usually for triangles). Default is None.

    Returns
    -------
    numpy.ndarray
        3d float array of coordinates.

    Notes
    -----
    It is assumed that all entries in 'coords' are coordinates of
    points in the same frame.
    """
    if local_axes is not None:
        if centralize:
            if centers is not None:
                ec = _centralize_cells_coords_2(cells_coords(coords, topo), centers)
            else:
                ec = _centralize_cells_coords(cells_coords(coords, topo))
        else:
            ec = cells_coords(coords, topo)
        return _cells_coords_tr_(ec, local_axes)
    else:
        if centralize:
            if centers is not None:
                return _centralize_cells_coords_2(cells_coords(coords, topo), centers)
            else:
                return _centralize_cells_coords(cells_coords(coords, topo))
        else:
            return cells_coords(coords, topo)


@njit(nogil=True, parallel=True, cache=__cache)
def _cells_coords_tr_(ecoords: ndarray, local_axes: ndarray) -> ndarray:
    nE, nNE, _ = ecoords.shape
    res = np.zeros_like(ecoords)
    for i in prange(nE):
        dcm = local_axes[i]
        for j in prange(nNE):
            res[i, j, :] = dcm @ ecoords[i, j, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _centralize_cells_coords(ecoords: ndarray) -> ndarray:
    nE, nNE, _ = ecoords.shape
    res = np.zeros_like(ecoords)
    for i in prange(nE):
        cc = cell_center(ecoords[i])
        for j in prange(nNE):
            res[i, j, :] = ecoords[i, j, :] - cc
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _centralize_cells_coords_2(ecoords: ndarray, centers: ndarray) -> ndarray:
    nE, nNE, _ = ecoords.shape
    res = np.zeros_like(ecoords)
    for i in prange(nE):
        cc = centers[i]
        for j in prange(nNE):
            res[i, j, :] = ecoords[i, j, :] - cc
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def cells_coords(coords: ndarray, topo: ndarray) -> ndarray:
    """
    Returns coordinates of cells from a coordinate base array and
    a topology array.

    Parameters
    ----------
    coords: numpy.ndarray
        2d float array of shape (nP, nD) of vertex coordinates.
        nP : number of points
        nD : number of dimensions of the model space
    topo: numpy.ndarray
        A 2D array of shape (nE, nNE) of vertex indices. The i-th
        row contains the vertex indices of the i-th element.
        nE : number of elements
        nNE : number of nodes per element

    Returns
    -------
    numpy.ndarray
        A 3d array of shape (nE, nNE, nD) that contains coordinates
        for all nodes of all cells according to the argument 'topo'.

    Notes
    -----
    The array 'coords' must be fully populated up to the maximum
    index in 'topo'. (len(coords) >= (topo.max() + 1))
    """
    nE, nNE = topo.shape
    res = np.zeros((nE, nNE, coords.shape[1]), dtype=coords.dtype)
    for iE in prange(nE):
        for iNE in prange(nNE):
            res[iE, iNE] = coords[topo[iE, iNE]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def cell_coords(coords: ndarray, topo: ndarray) -> ndarray:
    """
    Returns coordinates of a single cell from a coordinate
    array and a topology array.

    Parameters
    ----------
    coords: numpy.ndarray
        2d array of shape (nP, nD) of vertex coordinates.
        nP : number of points
        nD : number of dimensions of the model space
    topo: numpy.ndarray
        1d array of vertex indices of shape (nNE,).
        nNE : number of nodes per element

    Returns
    -------
    numpy.ndarray
        2d NumPy array of (nNE, nD) of coordinates for all nodes of all cells
        according to the argument 'topo'.

    Notes
    -----
    The array 'coords' must be fully populated up to the maximum index
    in 'topo'. (len(coords) >= (topo.max() + 1))
    """
    nNE = len(topo)
    res = np.zeros((nNE, coords.shape[1]), dtype=coords.dtype)
    for iNE in prange(nNE):
        res[iNE] = coords[topo[iNE]]
    return res


@njit(nogil=True, cache=__cache)
def cell_center_2d(ecoords: ndarray) -> ndarray:
    """
    Returns the center of a 2d cell.

    Parameters
    ----------
    ecoords: numpy.ndarray
        2d coordinate array of the element. The array has as many rows,
        as the number of nodes of the cell, and two columns.

    Returns
    -------
    numpy.ndarray
        1d coordinate array.
    """
    return np.array(
        [np.mean(ecoords[:, 0]), np.mean(ecoords[:, 1])], dtype=ecoords.dtype
    )


@njit(nogil=True, cache=__cache)
def cell_center(coords: ndarray) -> ndarray:
    """
    Returns the center of a single cell.

    Parameters
    ----------
    ecoords: numpy.ndarray
        2d coordinate array of the element. The array has as many rows,
        as the number of nodes of the cell, and three columns.

    Returns
    -------
    numpy.ndarray
        1d coordinate array.
    """
    return np.array(
        [np.mean(coords[:, i]) for i in range(coords.shape[1])],
        dtype=coords.dtype,
    )


def cell_centers_bulk(coords: ndarray, topo: ndarray) -> ndarray:
    """
    Returns coordinates of the centers of cells of the same kind.

    Parameters
    ----------
    coords: numpy.ndarray
        2d coordinate array.
    topo: numpy.ndarray
        2d point-based topology array.

    Returns
    -------
    numpy.ndarray
        2d coordinate array.
    """
    return np.mean(cells_coords(coords, topo), axis=1)


def cell_centers_bulk2(ecoords: ndarray) -> ndarray:
    """
    Returns coordinates of the centers of cells of the same kind.

    Parameters
    ----------
    ecoords: numpy.ndarray
        3d coordinate array of element coordinates.

    Returns
    -------
    numpy.ndarray
        2d coordinate array.
    """
    return np.mean(ecoords, axis=1)


@njit(nogil=True, parallel=True, cache=__cache)
def _nodal_distribution_factors_csr_(topo: csr_matrix, w: ndarray) -> ndarray:
    """
    The j-th factor of the i-th row is the contribution of
    element i to the j-th node. Assumes zeroed and tight indexing.

    Parameters
    ----------
    topo: :class:`~sigmaepsilon.math.linalg.sparse.csr.csr_matrix`
        2d integer topology array as a CSR matrix.
    w: numpy.ndarray
        The weights of the cells.
    """
    nE = topo.shape[0]
    indptr = topo.indptr
    data = topo.data.astype(np.int32)
    factors = np.zeros(len(data), dtype=w.dtype)
    nodal_w = np.zeros(data.max() + 1, dtype=w.dtype)
    for iE in range(nE):
        nodal_w[data[indptr[iE] : indptr[iE + 1]]] += w[iE]
    for iE in prange(nE):
        _i = indptr[iE]
        i_ = indptr[iE + 1]
        n = i_ - _i
        for j in prange(n):
            i = _i + j
            factors[i] = w[iE] / nodal_w[data[i]]
    return factors


@njit(nogil=True, parallel=True, cache=__cache)
def _nodal_distribution_factors_dense_(topo: ndarray, w: ndarray) -> ndarray:
    """
    The j-th factor of the i-th row is the contribution of
    element i to the j-th node. Assumes zeroed and tight indexing.

    Parameters
    ----------
    topo: numpy.ndarray
        2d integer topology array.
    w: numpy.ndarray
        The weights of the cells.
    """
    factors = np.zeros(topo.shape, dtype=w.dtype)
    nodal_w = np.zeros(topo.max() + 1, dtype=w.dtype)
    for iE in range(topo.shape[0]):
        nodal_w[topo[iE]] += w[iE]
    for iE in prange(topo.shape[0]):
        for jNE in prange(topo.shape[1]):
            factors[iE, jNE] = w[iE] / nodal_w[topo[iE, jNE]]
    return factors


def nodal_distribution_factors(
    topo: Union[csr_matrix, ndarray], weights: ndarray
) -> Union[csr_matrix, ndarray]:
    """
    The j-th factor of the i-th row is the contribution of
    element i to the j-th node. Assumes zeroed and tight indexing.

    Parameters
    ----------
    topo: numpy.ndarray or :class:`~sigmaepsilon.math.linalg.sparse.csr.csr_matrix`
        2d integer topology array.
    w: numpy.ndarray
        The weights of the cells.

    Returns
    -------
    numpy.ndarray or :class:`~sigmaepsilon.math.linalg.sparse.csr.csr_matrix`
        A 2d matrix with a matching shape to 'topo'.

    See also
    --------
    :func:`~sigmaepsilon.mesh.data.polydata.PolyData.nodal_distribution_factors`
    """
    if isinstance(topo, ndarray):
        return _nodal_distribution_factors_dense_(topo, weights)
    elif isinstance(topo, csr_matrix):
        data = _nodal_distribution_factors_csr_(topo, weights)
        indptr = topo.indptr
        indices = topo.indices
        shape = topo.shape
        return csr_matrix(data, indices, indptr, shape)
    else:
        t = type(topo)
        raise TypeError(f"Type {t} is not recognized as a topology.")


@njit(nogil=True, parallel=True, cache=__cache)
def distribute_nodal_data_bulk(data: ndarray, topo: ndarray, ndf: ndarray) -> ndarray:
    """
    Distributes nodal data to the cells for the case when the topology
    of the mesh is dense. The parameter 'ndf' controls the behaviour
    of the distribution.

    Parameters
    ----------
    data: numpy.ndarray
        2d array of shape (nP, nX), the data defined on points.
    topo: numpy.ndarray
        2d integer array of shape (nE, nNE), describing the topology.
    ndf: numpy.ndarray
        2d float array of shape (nE, nNE), describing the distribution
        of cells to the nodes.

    Returns
    -------
    numpy.ndarray
        A 3d float array of shape (nE, nNE, nX).
    """
    nE, nNE = topo.shape
    res = np.zeros((nE, nNE, data.shape[1]))
    for iE in prange(nE):
        for jNE in prange(nNE):
            res[iE, jNE] = data[topo[iE, jNE]] * ndf[iE, jNE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def distribute_nodal_data_sparse(
    data: ndarray, topo: ndarray, cids: ndarray, ndf: csr_matrix
) -> ndarray:
    """
    Distributes nodal data to the cells for the case when the topology of the
    mesh is sparse. The parameter 'ndf' controls the behaviour of the distribution.

    Parameters
    ----------
    data: numpy.ndarray
        2d array of shape (nP, nX), the data defined on points.
    topo: numpy.ndarray
        2d integer array of shape (nE, nNE), describing the topology.
    cids: numpy.ndarray
        A 1d integer array describing the indices of the cells.
    ndf: csr_matrix
        2d float array of shape (nE, nNE), describing the distribution
        of cells to all nodes in the mesh.

    Returns
    -------
    numpy.ndarray
        A 3d float array of shape (nE, nNE, nX).
    """
    nE, nNE = topo.shape
    indptr = ndf.indptr
    ndfdata = ndf.data
    res = np.zeros((nE, nNE, data.shape[1]))
    for iE in prange(nE):
        ndf_e = ndfdata[indptr[cids[iE]] : indptr[cids[iE] + 1]]
        for jNE in prange(nNE):
            res[iE, jNE] = data[topo[iE, jNE]] * ndf_e[jNE]
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def collect_nodal_data(
    celldata: ndarray, topo: ndarray, cids: ndarray, ndf: csr_matrix, res: ndarray
) -> ndarray:
    """
    Collects nodal data from data defined on nodes of cells.

    Parameters
    ----------
    celldata: numpy.ndarray
        Data defined on nodes of cells. It can be any array with at
        least 2 dimensions with a shape (nE, nNE, ...), where nE and
        nNE are the number of cells and nodes per cell.
    topo: numpy.ndarray
        A 2d integer array describing the topology of several cells of
        the same kind.
    cids: numpy.ndarray
        A 1d integer array describing the indices of the cells.
    ndf: csr_matrix
        Nodal distribution factors for each node of each cell in 'topo'.
        This must contain values for all cells in a mesh, not just the
        ones for which cell data and topology is provided by 'celldata'
        and 'topo'.
    res: numpy.ndarray
        An array for the output. It must have a proper size, at lest up
        to the maximum node index in 'topo'.
    """
    nE, nNE = topo.shape
    indptr = ndf.indptr
    ndfdata = ndf.data
    for iE in range(nE):
        ndf_e = ndfdata[indptr[cids[iE]] : indptr[cids[iE] + 1]]
        for jNE in prange(nNE):
            res[topo[iE, jNE]] += celldata[iE, jNE] * ndf_e[jNE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def explode_mesh_bulk(coords: ndarray, topo: ndarray) -> Tuple[ndarray]:
    """
    Turns an implicit representation of a mesh into an explicit one.

    .. note:
        This function is Numba-jittable in 'nopython' mode.

    Parameters
    ----------
    coords: numpy.ndarray
        A 2d coordinate array.
    topo: numpy.ndarray
        A 2d integer array describing the topology of several cells of
        the same kind.

    Returns
    -------
    numpy.ndarray
        A new coordinate array.
    numpy.ndarray
        A new topology array.
    """
    nE, nNE = topo.shape
    nD = coords.shape[1]
    coords_ = np.zeros((nE * nNE, nD), dtype=coords.dtype)
    topo_ = np.zeros_like(topo)
    for i in prange(nE):
        ii = i * nNE
        for j in prange(nNE):
            coords_[ii + j] = coords[topo[i, j]]
            topo_[i, j] = ii + j
    return coords_, topo_


@njit(nogil=True, parallel=True, cache=__cache)
def explode_mesh_data_bulk(
    coords: ndarray, topo: ndarray, data: ndarray
) -> Tuple[ndarray]:
    """
    Turns an implicit representation of a mesh into an explicit one
    and also data defined on the nodes of the cells to an 1d data
    array defined on the points of the new mesh.

    .. note:
        This function is Numba-jittable in 'nopython' mode.

    Parameters
    ----------
    coords: numpy.ndarray
        A 2d coordinate array.
    topo: numpy.ndarray
        A 2d integer array describing the topology of several cells of
        the same kind.
    data: numpy.ndarray
        A 2d array describing data on all nodes of the cells.

    Returns
    -------
    numpy.ndarray
        A new coordinate array.
    numpy.ndarray
        A new topology array.
    numpy.ndarray
        A new 1d data array.
    """
    nE, nNE = topo.shape
    nD = coords.shape[1]
    coords_ = np.zeros((nE * nNE, nD), dtype=coords.dtype)
    topo_ = np.zeros_like(topo)
    data_ = np.zeros(nE * nNE, dtype=coords.dtype)
    for i in prange(nE):
        ii = i * nNE
        for j in prange(nNE):
            coords_[ii + j] = coords[topo[i, j]]
            data_[ii + j] = data[i, j]
            topo_[i, j] = ii + j
    return coords_, topo_, data_


def explode_mesh(coords: ndarray, topo: ndarray, *, data=None):
    if data is None:
        return explode_mesh_bulk(coords, topo)
    elif isinstance(data, ndarray):
        return explode_mesh_data_bulk(coords, topo, data)
    else:
        raise NotImplementedError


@njit(nogil=True, parallel=True, cache=__cache)
def decompose(ecoords: ndarray, topo: ndarray, coords_out: ndarray) -> None:
    """
    Performes the inverse operation to coordinate explosion.
    Example usage at AxisVM domains. Works for all kinds of arrays.
    """
    for iE in prange(len(topo)):
        for jNE in prange(len(topo[iE])):
            coords_out[topo[iE][jNE]] = np.array(ecoords[iE][jNE])


@njit(nogil=True, parallel=True, cache=__cache)
def _avg_cell_data_1d_bulk_(data: np.ndarray, topo: np.ndarray):
    nE, nNE = topo.shape
    nD = data.shape[1]
    res = np.zeros((nE, nD), dtype=data.dtype)
    for iE in prange(nE):
        for jNE in prange(nNE):
            ind = topo[iE, jNE]
            for kD in prange(nD):
                res[iE, kD] += data[ind, kD]
        res[iE, :] /= nNE
    return res


def avg_cell_data(data: np.ndarray, topo: np.ndarray) -> ndarray:
    nR = len(data.shape)
    if nR == 2:
        res = _avg_cell_data_1d_bulk_(matrixform(data), topo)
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_matrix_single(dshp: ndarray, ecoords: ndarray) -> ndarray:
    """
    Returns Jacobian matrix of local to global transformation for for one cell and
    multiple evaluation pointsd.

    Parameters
    ----------
    dshp: numpy.ndarray
        A 3d numpy array of shape (nP, nNE, nD), where nP, nNE and nD
        are the number of evaluation points, nodes and spatial dimensions.
    ecoords: numpy.ndarray
        A 2d numpy array of shape (nNE, nD), where nNE and nD
        are the number of nodes and spatial dimensions.

    Returns
    -------
    numpy.ndarray
        A 3d array of shape (nP, nD, nD).
    """
    nG, _, nD = dshp.shape
    jac = np.zeros((nG, nD, nD), dtype=dshp.dtype)
    for iG in prange(nG):
        jac[iG] = dshp[iG].T @ ecoords
    return jac


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_matrix_bulk(dshp: ndarray, ecoords: ndarray) -> ndarray:
    """
    Returns Jacobian matrices of local to global transformation
    for several cells and evaluation points.

    Parameters
    ----------
    dshp: numpy.ndarray
        A 3d numpy array of shape (nG, nNE, nD), where nG, nNE and nD
        are the number of integration points, nodes and spatial dimensions.
    ecoords: numpy.ndarray
        A 3d numpy array of shape (nE, nNE, nD), where nE, nNE and nD
        are the number of elements, nodes and spatial dimensions.

    Returns
    -------
    numpy.ndarray
        A 4d array of shape (nE, nP, nD, nD).
    """
    nE = ecoords.shape[0]
    nG, _, nD = dshp.shape
    jac = np.zeros((nE, nG, nD, nD), dtype=dshp.dtype)
    for iG in prange(nG):
        d = dshp[iG].T
        for iE in prange(nE):
            jac[iE, iG] = d @ ecoords[iE]
    return jac


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_matrix_bulk_1d(dshp: ndarray, ecoords: ndarray) -> ndarray:
    """
    Returns the Jacobian matrix for multiple cells (nE), evaluated at
    multiple (nP) points.

    Returns
    -------
    numpy.ndarray
        A 4d NumPy array of shape (nE, nP, 1, 1).

    Notes
    -----
    As long as the line is straight, it is a constant metric element,
    and 'dshp' is only required here to provide an output with a correct
    shape.
    """
    lengths = lengths_of_lines2(ecoords)
    nE = ecoords.shape[0]
    if len(dshp.shape) > 4:
        # variable metric element -> dshp (nE, nP, nNE, nDOF, ...)
        nP = dshp.shape[1]
    else:
        # constant metric element -> dshp (nP, nNE, nDOF, ...)
        nP = dshp.shape[0]
    res = np.zeros((nE, nP, 1, 1), dtype=dshp.dtype)
    for iE in prange(nE):
        res[iE, :, 0, 0] = lengths[iE] / 2
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_det_bulk_1d(jac: ndarray) -> ndarray:
    """
    Calculates Jacobian determinants for 1d cells.

    Parameters
    ----------
    jac: numpy.ndarray
        4d float array of shape (nE, nG, 1, 1) for an nE number of
        elements and nG number of evaluation points.

    Returns
    -------
    numpy.ndarray
        A 2d array of shape (nE, nG) of jacobian determinants calculated
        for each element and evaluation points.
    """
    nE, nG = jac.shape[:2]
    res = np.zeros((nE, nG), dtype=jac.dtype)
    for iE in prange(nE):
        res[iE, :] = jac[iE, :, 0, 0]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def center_of_points(coords: ndarray) -> ndarray:
    """
    Returns the center of several points.

    Parameters
    ----------
    coords: numpy.ndarray
        A 2d coordinate array.
    """
    res = np.zeros(coords.shape[1], dtype=coords.dtype)
    for i in prange(res.shape[0]):
        res[i] = np.mean(coords[:, i])
    return res


@njit(nogil=True, cache=__cache)
def centralize(coords: ndarray) -> ndarray:
    """
    Centralizes coordinates of a point cloud.

    Parameters
    ----------
    coords: numpy.ndarray
        A 2d coordinate array.
    """
    nD = coords.shape[1]
    center = center_of_points(coords)
    coords[:, 0] -= center[0]
    coords[:, 1] -= center[1]
    if nD > 2:
        coords[:, 2] -= center[2]
    return coords


@njit(nogil=True, parallel=True, cache=__cache)
def lengths_of_lines(coords: ndarray, topo: ndarray) -> ndarray:
    """
    Returns lengths of several lines, where the geometry is
    defined implicitly.

    Parameters
    ----------
    coords: numpy.ndarray
        A 2d coordinate array.
    topo: numpy.ndarray
        A 2d topology array.
    """
    nE, nNE = topo.shape
    res = np.zeros(nE, dtype=coords.dtype)
    for i in prange(nE):
        res[i] = norm(coords[topo[i, nNE - 1]] - coords[topo[i, 0]])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def lengths_of_lines2(ecoords: ndarray) -> ndarray:
    """
    Returns lengths of several lines, where line cooridnates
    are specified explicitly.

    Parameters
    ----------
    ecoords: numpy.ndarray
        A 3d numpy array of shape (nE, nNE, nD), where nE, nNE and nD
        are the number of elements, nodes and spatial dimensions.
    """
    nE, nNE = ecoords.shape[:2]
    res = np.zeros(nE, dtype=ecoords.dtype)
    _nNE = nNE - 1
    for i in prange(nE):
        res[i] = norm(ecoords[i, _nNE] - ecoords[i, 0])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def distances_of_points(coords: ndarray) -> ndarray:
    """
    Calculates distances between a series of points.

    Parameters
    ----------
    coords: numpy.ndarray
        2d float array of shape (N, ...).

    Returns
    -------
    numpy.ndarray
        1d float array of shape (nP,).
    """
    nP = coords.shape[0]
    res = np.zeros(nP, dtype=coords.dtype)
    for i in prange(1, nP):
        res[i] = norm(coords[i] - coords[i - 1])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def pcoords_to_coords(pcoords: ndarray, ecoords: ndarray, shp: ndarray) -> ndarray:
    nP = pcoords.shape[0]
    nE, nNE, nD = ecoords.shape
    res = np.zeros((nE, nP, nD), dtype=ecoords.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            for iNE in range(nNE):
                res[iE, iP, :] += ecoords[iE, iNE] * shp[iP, iNE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def pcoords_to_coords_1d(pcoords: ndarray, ecoords: ndarray) -> ndarray:
    """
    Returns a flattened array of points, evaluated at multiple
    points and cells.

    Only for 1d cells.

    Parameters
    ----------
    pcoords: numpy.ndarray
        1d float array of length nP, coordinates in the range [0 , 1].
    ecoords: numpy.ndarray
        3d float array of shape (nE, 2+, nD) of cell coordinates.

    Notes
    -----
    It works for arbitrary topologies, but handles every cell as a line
    going from the firts to the last node of the cell.

    Returns
    -------
    numpy.ndarray
        2d float array of shape (nE * nP, nD).
    """
    nP = pcoords.shape[0]
    nE = ecoords.shape[0]
    nX = nE * nP
    res = np.zeros((nX, ecoords.shape[2]), dtype=ecoords.dtype)
    for iE in prange(nE):
        for jP in prange(nP):
            res[iE * nP + jP] = (
                ecoords[iE, 0] * (1 - pcoords[jP]) + ecoords[iE, -1] * pcoords[jP]
            )
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def norms(a: ndarray) -> ndarray:
    """
    Returns the Euclidean norms for the input data, calculated
    along axis 1.

    Parameters
    ----------
    a: numpy.ndarray
        2d array of data of shape (N, ...).

    Returns
    -------
    numpy.ndarray
        1d float array of shape (N, ).
    """
    nI = len(a)
    res = np.zeros(nI)
    for iI in prange(len(a)):
        res[iI] = np.dot(a[iI], a[iI])
    return np.sqrt(res)


@njit(nogil=True, parallel=True, cache=__cache)
def homogenize_nodal_values(data: ndarray, measure: ndarray) -> ndarray:
    """
    Calculates constant values for cells from existing data defined for
    each node, according to some measure.
    """
    nE, _, nDATA = data.shape  # nE, nNE, nDATA
    res = np.zeros((nE, nDATA), dtype=data.dtype)
    for i in prange(nE):
        for j in prange(nDATA):
            res[i, j] = np.sum(data[i, :, j]) / measure[i]
    return res


def is_cw(points: Iterable) -> bool:
    """
    Returns True, if the polygon described by the points is clockwise,
    False if it is counterclockwise.
    """
    area = 0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
    return area < 0


def is_ccw(points: Iterable) -> bool:
    """
    Returns True, if the polygon described by the points is counterclockwise,
    False if it is clockwise.
    """
    return not is_cw(points)


@njit(nogil=True, parallel=True, cache=__cache)
def global_shape_function_derivatives(dshp: ndarray, jac: ndarray) -> ndarray:
    """
    Calculates derivatives of shape functions wrt. the local axes of cells
    using derivatives along master axes evaulated at some points in
    the interval [-1, 1], and jacobian matrices of local-to-global mappings.

    Parameters
    ----------
    dshp: numpy.ndarray
        Derivatives of an nNE number of shape functions evaluated at an nP number
        of points as a float array of shape (nP, nNE, nD).
    jac: numpy.ndarray
        Jacobian matrices, evaluated for a a nP number of points in nE number of cells
        and for nD spatial dimensions as a float array of shape (nE, nP, nD, nD).

    Returns
    -------
    numpy.ndarray
        4d float array of shape (nE, nP, nNE, nD).
    """
    nP, nNE, nD = dshp.shape
    nE = jac.shape[0]
    res = np.zeros((nE, nP, nNE, nD), dtype=dshp.dtype)
    for iE in prange(nE):
        for iP in prange(nP):
            invJ = np.linalg.inv(jac[iE, iP])
            res[iE, iP] = dshp[iP] @ invJ.T
    return res


def coords_to_3d(x: ndarray) -> ndarray:
    x = atleast2d(x, back=True)
    if (N := x.shape[-1]) == 3:
        return x
    else:
        res = np.zeros((x.shape[0], 3), dtype=x.dtype)
        if N == 2:
            res[:, :2] = x
        elif N == 1:
            res[:, 0] = x.flatten()
        return res
