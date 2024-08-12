from typing import MutableMapping, Union, Dict, List, Tuple, Iterable, Any

import numpy as np
from numpy import ndarray
import awkward as ak
from awkward import Array as akarray
from scipy.sparse import csr_matrix as csr_scipy

from numba import njit, prange, types as nbtypes
from numba.typed import Dict as nbDict

from sigmaepsilon.math.linalg.sparse import csr_matrix, JaggedArray
from sigmaepsilon.math.arraysetops import unique2d
from sigmaepsilon.math import count_cols

from ...space import PointCloud
from ...topoarray import TopologyArray
from ...config import __hasnx__

if __hasnx__:
    import networkx as nx


__all__ = [
    "is_regular",
    "regularize",
    "count_cells_at_nodes",
    "cells_at_nodes",
    "nodal_adjacency",
    "unique_topo_data",
    "remap_topo",
    "detach_mesh_bulk",
    "rewire",
    "detach",
    "detach_mesh_data_bulk",
]


__cache = True
CoordsLike = Union[ndarray, PointCloud]
TopoLike = Union[ndarray, akarray, csr_matrix]
MappingLike = Union[ndarray, MutableMapping]
DoL = Dict[int, List[int]]

nbint32 = nbtypes.int32
nbint32A = nbint32[:]
nbint64 = nbtypes.int64
nbint64A = nbint64[:]


def rewire(topo: TopoLike, imap: MappingLike, invert: bool = False) -> Iterable:
    """
    Returns a new topology array. The argument 'imap' may be
    a dictionary or an array, that contains new indices for
    the indices in the old topology array.

    Parameters
    ----------
    topo: numpy.ndarray or sigmaepsilon.math.linalg.sparse.JaggedArray
        1d or 2d integer array representing topological data of a mesh.
    imap: MappingLike
        Inverse mapping on the index sets from global to local.
    invert: bool, Optional
        If `True` the argument `imap` describes a local to global
        mapping and an inversion takes place. In this case,
        `imap` must be a `numpy` array. Default is False.

    Returns
    -------
    TopoLike
        The same topology with the new numbering.
    """
    if invert:
        assert isinstance(imap, ndarray)
        imap = inds_to_invmap_as_dict(imap)
    if isinstance(topo, ndarray):
        if len(topo.shape) == 2:
            return remap_topo(topo, imap)
        elif len(topo.shape) == 1:
            return remap_topo_1d(topo, imap)
    elif isinstance(topo, TopologyArray):
        cuts, topo1d = topo.flatten(return_cuts=True)
        topo1d = remap_topo_1d(topo1d, imap)
        return TopologyArray(JaggedArray(topo1d, cuts=cuts))
    elif isinstance(topo, akarray):
        cuts = count_cols(topo)
        topo1d = ak.flatten(topo).to_numpy()
        topo1d = remap_topo_1d(topo1d, imap)
        return JaggedArray(topo1d, cuts=cuts)
    else:
        raise TypeError("Invalid topology with type <{}>".format(type(topo)))


@njit(nogil=True, parallel=True, cache=__cache)
def remap_topo(topo: ndarray, imap) -> ndarray:
    """
    Returns a new topology array. The argument 'imap' may be
    a dictionary or an array, that contains new indices for
    the indices in the old topology array.
    """
    nE, nNE = topo.shape
    res = np.zeros_like(topo)
    for iE in prange(nE):
        for jNE in prange(nNE):
            res[iE, jNE] = imap[topo[iE, jNE]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def remap_topo_1d(topo1d: ndarray, imap) -> ndarray:
    """
    Returns a new topology array. The argument 'imap' may be
    a dictionary or an array, that contains new indices for
    the indices in the old topology array.
    """
    N = topo1d.shape[0]
    res = np.zeros_like(topo1d)
    for i in prange(N):
        res[i] = imap[topo1d[i]]
    return res


def is_regular(topo: TopoLike) -> bool:
    """
    Returns True if the topology is regular, in the meaning
    that the smallest node index is zero, and every integer
    is represented up to the maximum index.
    """
    if isinstance(topo, ndarray):
        return topo.min() == 0 and len(np.unique(topo)) == topo.max() + 1
    elif isinstance(topo, akarray):
        return np.min(topo) == 0 and len(unique2d(topo)) == np.max(topo) + 1
    elif isinstance(topo, csr_matrix):
        t = topo.data.astype(np.int32)
        return t.min() == 0 and len(np.unique(t)) == t.max() + 1
    elif isinstance(topo, csr_scipy):
        return np.min(t) == 0 and len(np.unique(t)) == np.max(t) + 1
    else:
        raise NotImplementedError


def regularize(topo: TopoLike) -> Tuple[TopoLike, ndarray]:
    """
    Returns a regularized topology and the unique indices.
    The returned topology array contains indices of the unique
    array.

    Parameters
    ----------
    topo : numpy.array or awkward.Array
        A topology array.

    Returns
    -------
    numpy.array or awkward.Array
        An array with a similar type as the input array.
    """
    if isinstance(topo, ndarray):
        unique, regular = np.unique(topo, return_inverse=True)
        regular = regular.reshape(topo.shape)
        return regular, unique
    elif isinstance(topo, akarray):
        unique, regular = unique2d(topo, return_inverse=True)
        return regular, unique
    elif isinstance(topo, csr_matrix):
        t = topo.data.astype(np.int32)
        unique, regular = np.unique(t, return_inverse=True)
        topo.data[:] = regular
        return topo, unique
    else:
        t = type(topo)
        raise NotImplementedError(f"Unknown type {t}")


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _count_cells_at_nodes_reg_np_(topo: ndarray) -> ndarray:
    """
    Assumes a regular topology. Returns an array.
    """
    nE, nNE = topo.shape
    nN = topo.max() + 1
    count = np.zeros((nN), dtype=topo.dtype)
    for iE in prange(nE):
        for jNE in prange(nNE):
            count[topo[iE, jNE]] += 1
    return count


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _count_cells_at_nodes_np_(topo: ndarray, nodeIDs: ndarray) -> Dict[int, int]:
    """
    Returns a dict{int : int} for the nodes in `nideIDs`.
    Assumes an irregular topology. The array `topo` must contain
    indices relative to `nodeIDs`. If the topology is regular,
    `nodeIDs == np.arange(topo.max() + 1)` is `True`.
    """
    nE, nNE = topo.shape
    count = dict()
    for i in range(len(nodeIDs)):
        count[nodeIDs[i]] = 0
    for iE in prange(nE):
        for jNE in prange(nNE):
            count[nodeIDs[topo[iE, jNE]]] += 1
    return count


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _count_cells_at_nodes_reg_ak_(topo: akarray, nN: int) -> ndarray:
    """
    Assumes a regular topology. Returns an array.
    """
    ncols = count_cols(topo)
    nE = len(ncols)
    count = np.zeros((nN), dtype=np.int64)
    for iE in prange(nE):
        for jNE in prange(ncols[iE]):
            count[topo[iE][jNE]] += 1
    return count


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _count_cells_at_nodes_ak_(topo: akarray, nodeIDs: ndarray) -> Dict[int, int]:
    """
    Returns a dict{int : int} for the nodes in `nideIDs`.
    Assumes an irregular topology. The array `topo` must contain
    indices relative to `nodeIDs`. If the topology is regular,
    `nodeIDs == np.arange(topo.max() + 1)` is `True`.
    """
    ncols = count_cols(topo)
    nE = len(ncols)
    count = dict()
    for i in range(len(nodeIDs)):
        count[nodeIDs[i]] = 0
    for iE in prange(nE):
        for jNE in prange(ncols[iE]):
            count[nodeIDs[topo[iE, jNE]]] += 1
    return count


@njit(nogil=True, fastmath=True, cache=__cache)
def _count_cells_at_nodes_reg_csr_(topo: csr_matrix) -> ndarray:
    """
    Assumes a regular topology. Returns an array.
    """
    indptr = topo.indptr
    data = topo.data.astype(np.int32)
    nE = indptr.shape[0] - 1
    nN = np.max(data) + 1
    count = np.zeros((nN), dtype=indptr.dtype)
    for iE in range(nE):
        _i = indptr[iE]
        i_ = indptr[iE + 1]
        n = i_ - _i
        for j in range(n):
            i = _i + j
            count[data[i]] += 1
    return count


@njit(nogil=True, fastmath=True, cache=__cache)
def _count_cells_at_nodes_csr_(topo: csr_matrix, nodeIDs: ndarray) -> Dict[int, int]:
    """
    Returns a dict{int : int} for the nodes in `nideIDs`.
    Assumes an irregular topology. The array `topo` must contain
    indices relative to `nodeIDs`. If the topology is regular,
    `nodeIDs == np.arange(topo.max() + 1)` is `True`.
    """
    indptr = topo.indptr
    data = topo.data.astype(np.int32)
    nE = indptr.shape[0] - 1
    count = dict()
    for i in range(len(nodeIDs)):
        count[nodeIDs[i]] = 0
    for iE in range(nE):
        _i = indptr[iE]
        i_ = indptr[iE + 1]
        n = i_ - _i
        for j in range(n):
            i = _i + j
            count[nodeIDs[data[i]]] += 1
    return count


def count_cells_at_nodes(topo: TopoLike, regular: bool = False) -> Union[ndarray, dict]:
    """
    Returns an array or a dictionary, that counts connecting
    elements at the nodes of a mesh.

    Parameters
    ----------
    topo: TopoLike
        2d numpy array describing the topoogy of a mesh.
    regular: bool, Optional
        A True value means that 'topo' has tight and zeroed indexing.
        In this case, the output is a NumPy array. If False, the output
        a dictionary.

    Returns
    -------
    count: numpy.ndarray or dict
        Number of connecting elements for each node in a mesh.
    """
    if not regular:
        if isinstance(topo, ndarray):
            topo, nodeIDs = regularize(topo)
            return _count_cells_at_nodes_np_(topo, nodeIDs)
        elif isinstance(topo, akarray):
            topo, nodeIDs = regularize(topo)
            return _count_cells_at_nodes_ak_(topo, nodeIDs)
        elif isinstance(topo, csr_matrix):
            topo, nodeIDs = regularize(topo)
            return _count_cells_at_nodes_csr_(topo, nodeIDs)
        else:
            raise NotImplementedError
    else:
        if isinstance(topo, ndarray):
            return _count_cells_at_nodes_reg_np_(topo)
        elif isinstance(topo, akarray):
            nN = np.max(topo) + 1
            return _count_cells_at_nodes_reg_ak_(topo, nN)
        elif isinstance(topo, csr_matrix):
            return _count_cells_at_nodes_reg_csr_(topo)
        else:
            raise NotImplementedError


@njit(nogil=True, cache=__cache)
def _cells_at_nodes_reg_np_(topo: ndarray):
    """Assumes a regular topology."""
    nE, nNE = topo.shape
    nN = topo.max() + 1
    count = _count_cells_at_nodes_reg_np_(topo)
    cmax = count.max()
    ereg = np.zeros((nN, cmax), dtype=topo.dtype)
    nreg = np.zeros((nN, cmax), dtype=topo.dtype)
    count[:] = 0
    for iE in range(nE):
        for jNE in range(nNE):
            ereg[topo[iE, jNE], count[topo[iE, jNE]]] = iE
            nreg[topo[iE, jNE], count[topo[iE, jNE]]] = jNE
            count[topo[iE, jNE]] += 1
    return count, ereg, nreg


@njit(nogil=True, cache=__cache)
def _cells_at_nodes_reg_ak_(topo: akarray, nN: int):
    """Assumes a regular topology."""
    ncols = count_cols(topo)
    nE = len(ncols)
    count = _count_cells_at_nodes_reg_ak_(topo, nN)
    cmax = count.max()
    ereg = np.zeros((nN, cmax), dtype=np.int64)
    nreg = np.zeros((nN, cmax), dtype=np.int64)
    count[:] = 0
    for iE in range(nE):
        for jNE in range(ncols[iE]):
            ereg[topo[iE][jNE], count[topo[iE][jNE]]] = iE
            nreg[topo[iE][jNE], count[topo[iE][jNE]]] = jNE
            count[topo[iE][jNE]] += 1
    return count, ereg, nreg


@njit(nogil=True, cache=__cache)
def _cells_at_nodes_reg_csr_(topo: csr_matrix):
    """Assumes a regular topology."""
    indptr = topo.indptr
    data = topo.data.astype(np.int64)
    nE = len(indptr) - 1
    count = _count_cells_at_nodes_reg_csr_(topo)
    nN = np.max(data) + 1
    cmax = count.max()
    ereg = np.zeros((nN, cmax), dtype=data.dtype)
    nreg = np.zeros((nN, cmax), dtype=data.dtype)
    count[:] = 0
    for iE in range(nE):
        _i = indptr[iE]
        i_ = indptr[iE + 1]
        n = i_ - _i
        for j in prange(n):
            i = _i + j
            ereg[data[i], count[data[i]]] = iE
            nreg[data[i], count[data[i]]] = j
            count[data[i]] += 1
    return count, ereg, nreg


@njit(nogil=True, cache=__cache)
def _nodal_cell_data_to_dicts_(
    count: ndarray, ereg: ndarray, nreg: ndarray, cellIDs: ndarray, nodeIDs: ndarray
) -> Tuple[Dict, Dict]:
    ereg_d = nbDict.empty(key_type=nbint64, value_type=nbint64A)
    nreg_d = nbDict.empty(key_type=nbint64, value_type=nbint64A)
    for i in range(len(count)):
        ereg_d[nodeIDs[i]] = cellIDs[ereg[i, : count[i]]]
        nreg_d[nodeIDs[i]] = nreg[i, : count[i]]
    return ereg_d, nreg_d


@njit(nogil=True, parallel=True, cache=__cache)
def _nodal_cell_data_to_spdata_(
    count: np.ndarray, ereg: np.ndarray, nreg: np.ndarray
) -> tuple:
    nE = ereg.max() + 1
    nN = len(count)
    N = np.sum(count)
    indices = np.zeros(N, dtype=count.dtype)
    data = np.zeros(N, dtype=count.dtype)
    indptr = np.zeros(nN + 1, dtype=count.dtype)
    indptr[1:] = np.cumsum(count)
    for i in prange(nN):
        indices[indptr[i] : indptr[i + 1]] = ereg[i, : count[i]]
        data[indptr[i] : indptr[i + 1]] = nreg[i, : count[i]]
    shape = (nN, nE)
    return data, indices, indptr, shape


def cells_at_nodes(
    topo: TopoLike,
    *_,
    frmt: str = None,
    assume_regular: bool = False,
    cellIDs: Iterable = None,
    return_counts: bool = False,
    **__,
) -> Any:
    """
    Returns data about element connectivity at the nodes of a mesh.

    Parameters
    ----------
    topo: numpy.ndarray or sigmaepsilon.math.linalg.sparse.JaggedArray
        A 2d array (either jagged or not) representing topological data of a mesh.
    frmt: str
        A string specifying the output format. Valid options are
        'jagged', 'csr', 'scipy-csr' and 'dicts'.
        See below for the details on the returned object.
    return_counts: bool
        Wether to return the numbers of connecting elements at the nodes
        as a numpy array. If format is 'raw', the
        counts are always returned irrelevant to this option.
    assume_regular: bool
        If the topology is regular, you can gain some speed with providing
        it as `True`. Default is `False`.
    cellIDs: numpy.ndarray
        Indices of the cells in `topo`. If nor `None`, format must be 'dicts'.
        Default is `None`.

    Returns
    -------
    If `return_counts` is `True`, the number of connecting elements for
    each node is returned as either a numpy array (if `cellIDs` is `None`)
    or a dictionary (if `cellIDs` is not `None`). If format is 'raw', the
    counts are always returned.

    `frmt` = None

        counts : np.ndarray(nN) - numbers of connecting elements, only returned
            if 'return_counts' is `True`

        ereg : np.ndarray(nN, nmax) - indices of connecting elements

        nreg : np.ndarray(nN, nmax) - node indices with respect to the
            connecting elements
    where

        nN - is the number of nodes,

        nmax - is the maximum number of elements that meet at a node, that is
            count.max()

    `frmt` = 'csr'

        counts(optionally) : np.ndarray(nN) - number of connecting elements, only returned
            if 'return_counts' is `True`

        csr : csr_matrix - sparse matrix in a numba-jittable csr format.
                            Column indices denote element indices, values
                            have the meaning of element node locations.

    `frmt` = 'scipy-csr' or 'csr-scipy'

        counts(optionally) : np.ndarray(nN) - number of connecting elements, only returned
            if 'return_counts' is `True`

        csr : csr_matrix - An instance of scipy.linalg.sparse.csr_matrix.
                            Column indices denote element indices, values
                            have the meaning of element node locations.
    `frmt` = 'dicts'

        counts(optionally) : np.ndarray(nN) - number of connecting elements, only returned
            if 'return_counts' is `True`

        ereg : numba Dict(int : int[:]) - indices of elements for each node
                                            index

        nreg : numba Dict(int : int[:]) - local indices of nodes in the
                                            connecting elements

    `frmt` = 'jagged'

        counts(optionally) : np.ndarray(nN) - number of connecting elements, only returned
            if 'return_counts' is `True`

        ereg : JaggedArray - indices of elements for each node index

        nreg : JaggedArray - local indices of nodes in the connecting elements

    """
    if cellIDs is not None:
        assert frmt == "dicts", (
            "If `cellIDs` is not None," + " output format must be 'dicts'."
        )

    if not assume_regular:
        if is_regular(topo):
            nodeIDs = None
        else:
            topo, nodeIDs = regularize(topo)
    else:
        nodeIDs = None

    if nodeIDs is not None:
        assert frmt == "dicts", "Only the format 'dicts' supports an irregular input!"

    if isinstance(topo, ndarray):
        counts, ereg, nreg = _cells_at_nodes_reg_np_(topo)
    elif isinstance(topo, akarray):
        nN = np.max(topo) + 1
        counts, ereg, nreg = _cells_at_nodes_reg_ak_(topo, nN)
    elif isinstance(topo, csr_matrix):
        counts, ereg, nreg = _cells_at_nodes_reg_csr_(topo)
    else:
        raise NotImplementedError

    frmt = "" if frmt is None else frmt

    if frmt in ["csr", "csr-scipy", "scipy-csr"]:
        data, indices, indptr, shape = _nodal_cell_data_to_spdata_(counts, ereg, nreg)
        csr = csr_matrix(data=data, indices=indices, indptr=indptr, shape=shape)
        if frmt == "csr":
            csr = csr_matrix(data=data, indices=indices, indptr=indptr, shape=shape)
        else:
            csr = csr_scipy((data, indices, indptr), shape=shape)
        if return_counts:
            return counts, csr
        return csr
    elif frmt == "dicts":
        if isinstance(topo, csr_matrix):
            if cellIDs is None:
                cellIDs = np.arange(len(topo.indptr) - 1).astype(int)
        cellIDs = np.arange(len(topo)).astype(int) if cellIDs is None else cellIDs
        nodeIDs = np.arange(len(counts)).astype(int) if nodeIDs is None else nodeIDs
        cellIDs = cellIDs.astype(np.int64)
        nodeIDs = nodeIDs.astype(np.int64)
        ereg, nreg = _nodal_cell_data_to_dicts_(counts, ereg, nreg, cellIDs, nodeIDs)
        if return_counts:
            return counts, ereg, nreg
        return ereg, nreg
    elif frmt == "jagged":
        data, indices, indptr, shape = _nodal_cell_data_to_spdata_(counts, ereg, nreg)
        ereg = JaggedArray(indices, cuts=counts)
        nreg = JaggedArray(data, cuts=counts)
        if return_counts:
            return counts, ereg, nreg
        return ereg, nreg

    return counts, ereg, nreg


@njit(nogil=True, cache=__cache)
def _nodal_adjacency_as_dol_np_(topo: ndarray, ereg: DoL) -> DoL:
    """Returns nodal adjacency as a dictionary of lists."""
    res = dict()
    for iP in ereg:
        res[iP] = np.unique(topo[ereg[iP], :])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _subtopo_1d_(
    topo1d: ndarray, cuts: ndarray, inds: ndarray, indptr: ndarray
) -> ndarray:
    nN = np.sum(cuts[inds])
    nE = len(inds)
    subindptr = np.zeros(nE + 1, dtype=cuts.dtype)
    subindptr[1:] = np.cumsum(cuts[inds])
    subtopo1d = np.zeros(nN, dtype=topo1d.dtype)
    for iE in prange(nE):
        subtopo1d[subindptr[iE] : subindptr[iE + 1]] = topo1d[
            indptr[inds[iE]] : indptr[inds[iE] + 1]
        ]
    return subtopo1d


@njit(nogil=True, cache=__cache)
def _nodal_adjacency_as_dol_ak_(topo1d: ndarray, cuts: ndarray, ereg: DoL) -> DoL:
    """Returns nodal adjacency as a dictionary of lists."""
    res = dict()
    nN = len(cuts)
    indptr = np.zeros(nN + 1, dtype=cuts.dtype)
    indptr[1:] = np.cumsum(cuts)
    for iP in ereg:
        res[iP] = np.unique(_subtopo_1d_(topo1d, cuts, ereg[iP], indptr))
    return res


@njit(nogil=True, parallel=False, fastmath=False, cache=__cache)
def dol_keys(dol: DoL) -> ndarray:
    nD = len(dol)
    res = np.zeros(nD, dtype=np.int64)
    c = 0
    for i in dol:
        res[c] = i
        c += 1
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def dol_to_jagged_data(dol: DoL) -> Tuple[ndarray, ndarray]:
    nD = len(dol)
    keys = dol_keys(dol)
    widths = np.zeros(nD, dtype=np.int64)
    for i in prange(nD):
        widths[i] = len(dol[keys[i]])
    indptr = np.zeros(nD + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(widths)
    N = np.sum(widths)
    data1d = np.zeros(N, dtype=np.int64)
    for i in prange(nD):
        data1d[indptr[i] : indptr[i + 1]] = dol[keys[i]]
    return widths, data1d


def detach(coords: CoordsLike, topo: TopoLike, inds: ndarray | None = None):
    """
    Given a topology array and the coordinate array it refers to,
    the function returns the coordinate array of the points involved
    in the topology, and a new topology array, with indices referencing
    the unique coordinate array.

    Parameters
    ----------
    coords: CoordsLike
        A 2d float array representing geometrical data of a mesh.
        If it is a `PointCloud` instance, indices may be included
        and parameter `inds` is obsolete.
    topo: TopoLike
        A 2d integer array (either jagged or not) representing topological
        data of a mesh.
    inds: ndarray, Optional
        Global indices of the coordinates, in `coords`. If provided, the
        coordinates of node `j` of cell `i` is accessible as

        ``coords[imap[topo[i, j]]``,

        where `imap` is a mapping from local to global indices, and gets
        automatically generated from `inds`.  Default is None.

    Returns
    -------
    ndarray
        NumPy array similar to `coords`, but possibly with less entries.
    TopoLike
        Integer array representing the topology, with a good cahnce of
        being jagged, depending on your input.
    """
    if isinstance(topo, ndarray):
        if inds is None and isinstance(coords, PointCloud):
            inds = coords.inds
        if isinstance(inds, ndarray):
            topo = rewire(topo, inds, True)
        return detach_mesh_bulk(coords, topo)
    elif hasattr(topo, "__array_function__"):
        if inds is None and isinstance(coords, PointCloud):
            inds = coords.inds
        if isinstance(inds, ndarray):
            topo = rewire(topo, inds, True)
        return detach_mesh_jagged(coords, topo)
    else:
        raise TypeError("Invalid topology with type <{}>".format(type(topo)))


def detach_mesh_jagged(coords: ndarray, topo: ndarray):
    """
    Given a topology array and the coordinate array it refers to,
    the function returns the coordinate array of the points involved
    in the topology, and a new topology array, with indices referencing
    the new coordinate array.
    """
    inds = np.unique(topo)
    cuts, topo1d = topo.flatten(return_cuts=True)
    imap = inds_to_invmap_as_dict(inds)
    topo1d = remap_topo_1d(topo1d, imap)
    topo = JaggedArray(topo1d, cuts=cuts)
    return coords[inds, :], topo


@njit(nogil=True, cache=__cache)
def detach_mesh_bulk(coords: ndarray, topo: ndarray):
    """
    Given a topology array and the coordinate array it refers to,
    the function returns the coordinate array of the points involved
    in the topology, and a new topology array, with indices referencing
    the new coordinate array.
    """
    inds = np.unique(topo)
    return coords[inds], remap_topo(topo, inds_to_invmap_as_dict(inds))


@njit(nogil=True, cache=__cache)
def detach_mesh_data_bulk(coords: ndarray, topo: ndarray, data: ndarray):
    """
    Given a subset of the topology of a mesh, the function returns the
    coordinates and nodal data of the points involved in the topology,
    and a new topology array, with indices referencing the new coordinate
    array.
    """
    inds = np.unique(topo)
    coords_ = coords[inds]
    data_ = data[inds]
    return coords_, data_, remap_topo(topo, inds_to_invmap_as_dict(inds))


@njit(nogil=True, cache=__cache)
def inds_to_invmap_as_dict(inds: np.ndarray):
    """
    Returns a mapping that maps global indices to local ones.

    Parameters
    ----------
    inds: numpy.ndarray
        An array of global indices.

    Returns
    -------
    dict
        Mapping from global to local.
    """
    res = dict()
    for i in range(len(inds)):
        res[inds[i]] = i
    return res


@njit(nogil=True, cache=__cache)
def arrays_to_imap_as_dict(source: np.ndarray, target: np.ndarray):
    """
    Turns to index array into a dicionary such that `result[source[i]] = target[i]`.

    Parameters
    ----------
    source: numpy.ndarray
        An index array.
    target: numpy.ndarray
        An index array.

    Returns
    -------
    dict
        Mapping from global to local.
    """
    res = dict()
    for i in range(len(source)):
        res[source[i]] = target[i]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def inds_to_invmap_as_array(inds: np.ndarray):
    """
    Returns a mapping that maps global indices to local ones
    as an array.

    Parameters
    ----------
    inds: numpy.ndarray
        An array of global indices.

    Returns
    -------
    numpy.ndarray
        Mapping from global to local.
    """
    res = np.zeros(inds.max() + 1, dtype=inds.dtype)
    for i in prange(len(inds)):
        res[inds[i]] = i
    return res


def nodal_adjacency(
    topo: TopoLike, *_, frmt: str = None, assume_regular: bool = False, **__
) -> Any:
    """
    Returns nodal adjacency information of a mesh.

    Parameters
    ----------
    topo: numpy.ndarray array or JaggedArray
        A 2d array (either jagged or not) representing topological data of a mesh.
    frmt: str
        A string specifying the output format. Valid options are
        'jagged', 'csr', 'nx' and 'scipy-csr'. See below for the details on the
        returned object.
    assume_regular: bool
        If the topology is regular, you can gain some speed with providing
        it as `True`. Default is `False`.

    Notes
    -----
    1) You need `networkx` to be installed for most of the functionality here.

    2) A node is adjacent to itself.

    Returns
    -------
    `frmt` = None
        A dictionary of numpy arrays for each node.
    `frmt` = 'csr'
        csr_matrix - A sparse matrix in a numba-jittable csr format.
    `frmt` = 'scipy-csr'
        An instance of scipy.linalg.sparse.csr_matrix.
    `frmt` = 'nx'
        A networkx Graph.
    `frmt` = 'jagged'
        A JaggedArray instance.
    """
    frmt = "" if frmt is None else frmt
    ereg, _ = cells_at_nodes(topo, frmt="dicts", assume_regular=assume_regular)
    if isinstance(topo, ndarray):
        dol = _nodal_adjacency_as_dol_np_(topo, ereg)
    elif isinstance(topo, akarray):
        cuts, topo1d = JaggedArray(topo).flatten(return_cuts=True)
        dol = _nodal_adjacency_as_dol_ak_(topo1d, cuts, ereg)
    if not __hasnx__ and frmt != "jagged":
        errorstr = "`networkx` must be installed for format '{}'."
        raise ImportError(errorstr.format(frmt))
    if frmt == "nx":
        return nx.from_dict_of_lists(dol)
    elif frmt == "scipy-csr":
        G = nx.from_dict_of_lists(dol)
        return nx.to_scipy_sparse_array(G).tocsr()
    elif frmt == "csr":
        G = nx.from_dict_of_lists(dol)
        csr = nx.to_scipy_sparse_array(G).tocsr()
        return csr_matrix(csr)
    elif frmt == "jagged":
        cuts, data1d = dol_to_jagged_data(dol)
        return JaggedArray(data1d, cuts=cuts)
    return dol


def unique_topo_data(topo3d: TopoLike) -> Tuple[ndarray, ndarray]:
    """
    Returns information about unique topological elements
    of a mesh. It can be used to return unique lines of a 2d
    mesh, unique faces of a 3d mesh, etc.

    Parameters
    ----------
    topo: numpy.ndarray
        Hierarchical topology array. The array must be 3 dimensional containing node
        indices for every node as a subarray. For instance for a 2d cell, the node
        indices of the j-th edge of the i-th element read as `topo[i, j]`. In general,
        the first axis runs for the elements, the second axis runs for edges (2d) or
        faces (3d).

    Returns
    -------
    numpy.ndarray
        The sorted unique topolical entities as integer arrays
        of node indices.
    numpy.ndarray
        Indices of the unique array, that can be used to
        reconstruct `topo`. See the examples.

    Examples
    --------
    Find unique edges of a mesh of Q4 quadrilaterals

    >>> from sigmaepsilon.mesh.grid import grid
    >>> from sigmaepsilon.mesh.utils.topodata import edges_Q4
    >>> import numpy as np

    >>> coords, topo = grid(size=(1, 1), shape=(10, 10), eshape='Q4')

    To get a 3d integer array listing all the edges of all quads:

    >>> edges3d = edges_Q4(topo)

    To find the unique edges of the mesh:

    >>> from sigmaepsilon.mesh.utils.topology import unique_topo_data
    >>> edges, edgeIDs = unique_topo_data(edges3d)

    Then, to reconstruct `edges3d`, do the following

    >>> edges3d_ = np.zeros_like(edges3d)
    >>> for i in range(edgeIDs.shape[0]):
    ...     for j in range(edgeIDs.shape[1]):
    ...         edges3d_[i, j, :] = edges[edgeIDs[i, j]]
    """
    if isinstance(topo3d, ndarray):
        nE, nD, nN = topo3d.shape
        topo3d = topo3d.reshape((nE * nD, nN))
        topo3d = np.sort(topo3d, axis=1)
        topo3d, topoIDs = np.unique(topo3d, axis=0, return_inverse=True)
        topoIDs = topoIDs.reshape((nE, nD))
        return topo3d, topoIDs
    elif isinstance(topo3d, JaggedArray):
        raise NotImplementedError
