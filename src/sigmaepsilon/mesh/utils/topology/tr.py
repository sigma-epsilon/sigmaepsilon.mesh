# !TODO  handle decimation in all transformations, template : T6_to_T3
import numpy as np
from awkward import Array
from numba import njit, prange
from typing import Union, Sequence, Tuple

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from numpy import ndarray, newaxis
from concurrent.futures import ThreadPoolExecutor

from ..tri import edges_tri
from ..utils import cells_coords
from ..topodata import (
    edgeIds_T3,
    edgeIds_TET4,
    edgeIds_H8,
    edges_Q4,
    edges_H8,
    faces_H8,
    edges_TET4,
    edges_W6,
    faces_W6,
)
from .topo import unique_topo_data
from .trimap import trimap_Q8

__cache = True


__all__ = [
    "transform_topology",
    "compose_trmap",
    "to_T3",
    "L2_to_L3",
    "T3_to_T6",
    "T3_to_L2",
    "T6_to_T3",
    "Q4_to_Q8",
    "Q4_to_Q9",
    "Q9_to_Q4",
    "Q8_to_T3",
    "Q9_to_T6",
    "Q4_to_T3",
    "H8_to_L2",
    "H8_to_Q4",
    "H8_to_H27",
    "H8_to_TET4",
    "H27_to_H8",
    "H27_to_TET10",
    "TET4_to_L2",
    "TET4_to_TET10",
    "TET10_to_TET4",
    "W6_to_W18",
    "W6_to_TET4",
    "W18_to_W6",
]


DataLike = Union[ndarray, Sequence[ndarray]]


def transform_topology(
    topo: ndarray,
    path: ndarray,
    data: ndarray | None = None,
    *_,
    MT: bool = True,
    max_workers: int = 4,
    **__,
) -> Union[ndarray, Tuple[ndarray]]:
    """
    Transforms a mesh by canging the celltype, but keeping the topology of
    the overall configuration.

    Parameters
    ----------
    topo: numpy.ndarray
        The input topology.
    path: numpy.ndarray
        The transformation path.
    data: numpy.ndarray, Optional
        Data attached to the points or the cells. Default is None.
    MT: bool, Optional
        If True, transformation is carried out using multiple threads. Default is True.
    max_workers: int, Optional
        The maximum number of worker threads. Only relevant if MT is True. Default is 4.

    Returns
    -------
    Union[numpy.ndarray, Tuple[numpy.ndarray]]
        Topology, and optionally data as NumPy arrays.

    Example
    -------
    For instance, this could be a way of producing a more coarse triangulation:

    >>> from sigmaepsilon.mesh.utils.topology import transform_topology
    >>> from sigmaepsilon.mesh.triang import triangulate
    >>> from sigmaepsilon.mesh.grid import grid
    >>> coords, topo, _ = triangulate(size=(1, 1), shape=(5, 10))
    >>> nE1 = topo.shape[0]
    >>> coords, topo = T3_to_T6(coords, topo)
    >>> coords, topo = T6_to_T3(coords, topo)
    >>> nE2 = topo.shape[0]
    >>> nE1 * 4 == nE2
    True

    The functions `T3_to_T6` and `T6_to_T3` call `transform_topology` to do
    the transformations. The same with calling `transform_topology` directly
    could be like this:

    >>> coords, topo, _ = triangulate(size=(1, 1), shape=(5, 10))
    >>> nE1 = topo.shape[0]
    >>> coords, topo = T3_to_T6(coords, topo)
    >>> path = np.array(
    ...     [[0, 3, 5], [3, 1, 4], [5, 4, 2], [5, 3, 4]], dtype=int
    ...            )
    >>> topo = transform_topology(topo, path)
    >>> nE2 = topo.shape[0]
    >>> nE1 * 4 == nE2
    True

    In both cases, the original triangulation is splitted to 4 subtriangles per
    each cell. Hence, `transform_topology` is more general, since you have more
    control over how you want to split the cells that make up the original mesh.
    """
    nD = len(path.shape)

    if nD == 1:
        path = path.reshape(1, len(path))

    assert nD <= 2, "Path must be 1 or 2 dimensional."

    if data is None:
        return _transform_topology_(topo, path)
    else:
        if isinstance(data, Array):
            try:
                data = data.to_numpy()
            except Exception:
                raise TypeError("Invalid data type '{}'".format(data.__class__))
        if isinstance(data, ndarray):
            data = transform_topology_data(topo, data, path)
            return _transform_topology_(topo, path), data
        elif isinstance(data, Iterable):

            def foo(d):
                return transform_topology_data(topo, d, path)

            if MT:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    dmap = executor.map(foo, data)
            else:
                dmap = map(foo, data)
            return _transform_topology_(topo, path), list(dmap)


def transform_topology_data(topo: ndarray, data: ndarray, path: ndarray) -> ndarray:
    if data.shape[:2] == topo.shape[:2]:
        # it is assumed that values are provided for each node of each cell
        res = repeat_cell_nodal_data(data, path)
    elif data.shape[0] == topo.shape[0]:
        # assume that data is constant over the elements
        res = np.repeat(data, path.shape[0], axis=0)
    else:
        raise NotImplementedError("Invalid data shape {}".format(data.shape))
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _transform_topology_(topo: ndarray, path: ndarray) -> ndarray:
    nE = len(topo)
    nSub, nSubN = path.shape
    res = np.zeros((nSub * nE, nSubN), dtype=topo.dtype)
    for iE in prange(nE):
        c = iE * nSub
        for jE in prange(nSubN):
            for kE in prange(nSub):
                res[c + kE, jE] = topo[iE, path[kE, jE]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def repeat_cell_nodal_data(edata: ndarray, path: ndarray) -> ndarray:
    nSub, nSubN = path.shape
    nE = edata.shape[0]
    res = np.zeros((nSub * nE, nSubN) + edata.shape[2:], dtype=edata.dtype)
    for i in prange(nE):
        ii = nSub * i
        for j in prange(nSub):
            jj = ii + j
            for k in prange(nSubN):
                res[jj, k] = edata[i, path[j, k]]
    return res


def compose_trmap(map1: ndarray, map2: ndarray) -> ndarray:
    nE1 = map1.shape[0]
    nE2, nNE2 = map2.shape
    res = np.zeros((int(nE1 * nE2), nNE2), dtype=map1.dtype)
    c = 0
    for i in range(nE1):
        for j in range(nE2):
            for k in range(nNE2):
                res[c, k] = map1[i, map2[j, k]]
            c += 1
    return res


def to_T3(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    **kwargs,
) -> Tuple[ndarray]:
    if path is None:
        raise TypeError("Expected Iterable for argument 'path', got 'NoneType'.")

    if not isinstance(path, ndarray):
        raise TypeError(f"Expected 'ndarray' for argument 'path', got {type(path)}.")

    if not len(path.shape) == 2:
        raise ValueError("'path' must be a 2d NumPy array")

    if not path.shape[1] == 3:
        raise ValueError("Invalid 'path'.")

    if data is None:
        return coords, +transform_topology(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topology(topo, path, data, *args, **kwargs)


def Q8_to_T3(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *,
    path: ndarray = None,
    **kwargs,
) -> Tuple[ndarray]:
    if path is None:
        path = trimap_Q8()
    elif isinstance(path, str):
        raise NotImplementedError
    return to_T3(coords, topo, data, path=path, **kwargs)


def T6_to_T3(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    subdivide: bool = True,
    **kwargs,
) -> Tuple[ndarray]:
    if isinstance(path, ndarray):
        assert path.shape[1] == 3
    else:
        if path is None:
            if subdivide:
                path = np.array(
                    [[0, 3, 5], [3, 1, 4], [5, 4, 2], [5, 3, 4]], dtype=topo.dtype
                )
            else:
                path = np.array([[0, 1, 2]], dtype=topo.dtype)
    if data is None:
        return coords, +transform_topology(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topology(topo, path, data, *args, **kwargs)


def H27_to_H8(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    subdivide: bool = True,
    **kwargs,
) -> Tuple[ndarray]:
    if isinstance(path, ndarray):
        assert path.shape[1] == 8
    else:
        if path is None:
            if subdivide:
                path = np.array(
                    [
                        [0, 8, 24, 11, 16, 22, 26, 20],
                        [8, 1, 9, 24, 22, 17, 21, 26],
                        [24, 9, 2, 10, 26, 21, 18, 23],
                        [11, 24, 10, 3, 20, 26, 23, 19],
                        [16, 22, 26, 20, 4, 12, 25, 15],
                        [22, 17, 21, 26, 12, 5, 13, 25],
                        [26, 21, 18, 23, 25, 13, 6, 14],
                        [20, 26, 23, 19, 15, 25, 14, 7],
                    ],
                    dtype=topo.dtype,
                )
            else:
                path = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=topo.dtype)
    if data is None:
        return coords, +transform_topology(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topology(topo, path, data, *args, **kwargs)


def Q9_to_Q4(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    **kwargs,
) -> Tuple[ndarray]:
    if isinstance(path, ndarray):
        assert path.shape[1] == 4
    else:
        if path is None:
            path = np.array(
                [[0, 4, 8, 7], [4, 1, 5, 8], [8, 5, 2, 6], [7, 8, 6, 3]],
                dtype=topo.dtype,
            )
        elif isinstance(path, str):
            if path == "grid":
                path = np.array(
                    [[0, 3, 4, 1], [3, 6, 7, 4], [4, 7, 8, 5], [1, 4, 5, 2]],
                    dtype=topo.dtype,
                )
    if data is None:
        return coords, +transform_topology(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topology(topo, path, data, *args, **kwargs)


def Q9_to_T6(coords: ndarray, topo: ndarray, path: ndarray = None):
    if path is None:
        path = np.array([[0, 8, 2, 4, 5, 1], [0, 6, 8, 3, 7, 4]], dtype=topo.dtype)
    return _Q9_to_T6(coords, topo, path)


@njit(nogil=True, parallel=True, cache=__cache)
def _Q9_to_T6(coords: ndarray, topo: ndarray, path: ndarray):
    nE = len(topo)
    topoT6 = np.zeros((2 * nE, 6), dtype=topo.dtype)
    for iE in prange(nE):
        c = iE * 2
        for jE in prange(6):
            topoT6[c, jE] = topo[iE, path[0, jE]]
            topoT6[c + 1, jE] = topo[iE, path[1, jE]]
    return coords, topoT6


def H8_to_TET4(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    **kwargs,
) -> Tuple[ndarray]:
    if isinstance(path, ndarray):
        assert path.shape[1] == 4
    else:
        if path is None:
            path = np.array(
                [[1, 2, 0, 5], [3, 0, 2, 7], [5, 4, 7, 0], [6, 5, 7, 2], [0, 2, 7, 5]],
                dtype=topo.dtype,
            )
        elif isinstance(path, str):
            raise NotImplementedError
    if data is None:
        return coords, +transform_topology(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topology(topo, path, data, *args, **kwargs)


def H27_to_TET10(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    **kwargs,
) -> Tuple[ndarray]:
    if isinstance(path, ndarray):
        assert path.shape[1] == 10
    else:
        if path is None:
            path = np.array(
                [
                    [1, 2, 0, 5, 9, 24, 8, 17, 21, 22],
                    [3, 0, 2, 7, 11, 24, 10, 19, 20, 23],
                    [5, 4, 7, 0, 12, 15, 25, 22, 16, 20],
                    [6, 5, 7, 2, 13, 25, 14, 18, 21, 23],
                    [0, 2, 7, 5, 24, 23, 20, 22, 21, 25],
                ],
                dtype=topo.dtype,
            )
        elif isinstance(path, str):
            raise NotImplementedError
    if data is None:
        return coords, +transform_topology(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topology(topo, path, data, *args, **kwargs)


def H8_to_Q4(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    **kwargs,
) -> Tuple[ndarray]:
    if isinstance(path, ndarray):
        assert path.shape[1] == 4
    else:
        if path is None:
            path = np.array(
                [
                    [0, 4, 7, 3],
                    [1, 2, 6, 5],
                    [0, 1, 5, 4],
                    [2, 3, 7, 6],
                    [0, 3, 2, 1],
                    [4, 5, 6, 7],
                ],
                dtype=topo.dtype,
            )
        elif isinstance(path, str):
            raise NotImplementedError
    if data is None:
        return coords, +transform_topology(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topology(topo, path, data, *args, **kwargs)


def T3_to_L2(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    **kwargs,
) -> Tuple[ndarray]:
    if isinstance(path, ndarray):
        assert path.shape[0] == 3, "Invalid shape!"
        assert path.shape[1] == 2, "Invalid shape!"
    else:
        if path is None:
            path = edgeIds_T3()
        else:
            raise NotImplementedError("Invalid path!")
    if data is None:
        nE = len(topo)
        nSub, nSubN = path.shape
        topo = np.reshape(transform_topology(topo, path), (nE, nSub, nSubN))
        edges, _ = unique_topo_data(topo)
        return coords, edges
    else:
        raise NotImplementedError("Data conversion is not available here!")


def TET4_to_L2(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    **kwargs,
) -> Tuple[ndarray]:
    if isinstance(path, ndarray):
        assert path.shape[0] == 6, "Invalid shape!"
        assert path.shape[1] == 2, "Invalid shape!"
    else:
        if path is None:
            path = edgeIds_TET4()
        else:
            raise NotImplementedError("Invalid path!")
    if data is None:
        nE = len(topo)
        nSub, nSubN = path.shape
        topo = np.reshape(transform_topology(topo, path), (nE, nSub, nSubN))
        edges, _ = unique_topo_data(topo)
        return coords, edges
    else:
        raise NotImplementedError("Data conversion is not available here!")


def H8_to_L2(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    **kwargs,
) -> Tuple[ndarray]:
    if isinstance(path, ndarray):
        assert path.shape[0] == 12, "Invalid shape!"
        assert path.shape[1] == 2, "Invalid shape!"
    else:
        if path is None:
            path = edgeIds_H8()
        else:
            raise NotImplementedError("Invalid path!")
    if data is None:
        nE = len(topo)
        nSub, nSubN = path.shape
        topo = np.reshape(transform_topology(topo, path), (nE, nSub, nSubN))
        edges, _ = unique_topo_data(topo)
        return coords, edges
    else:
        raise NotImplementedError("Data conversion is not available here!")


def Q4_to_T3(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    **kwargs,
) -> Tuple[ndarray]:
    if isinstance(path, ndarray):
        assert path.shape[1] == 3
    else:
        if path is None:
            path = np.array([[0, 1, 2], [0, 2, 3]], dtype=topo.dtype)
        elif isinstance(path, str):
            if path == "grid":
                path = np.array([[0, 2, 3], [0, 3, 1]], dtype=topo.dtype)
    if data is None:
        return coords, +transform_topology(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topology(topo, path, data, *args, **kwargs)


def L2_to_L3(coords: ndarray, topo: ndarray, order="ikj") -> Tuple[ndarray]:
    nP = len(coords)
    nodes, nodeIDs = unique_topo_data(topo[:, newaxis, :])
    new_coords = np.mean(coords[nodes], axis=1)
    new_topo = nodeIDs + nP
    topo = np.hstack((topo, new_topo))
    coords = np.vstack((coords, new_coords))
    if order == "ikj":
        _buf = np.copy(topo[:, 1])
        topo[:, 1] = topo[:, 2]
        topo[:, 2] = _buf
    else:
        raise NotImplementedError
    return coords, topo


def T3_to_T6(coords: ndarray, topo: ndarray) -> Tuple[ndarray]:
    nP = len(coords)
    edges, edgeIDs = unique_topo_data(edges_tri(topo))
    new_coords = np.mean(coords[edges], axis=1)
    new_topo = edgeIDs + nP
    topo = np.hstack((topo, new_topo))
    coords = np.vstack((coords, new_coords))
    return coords, topo


def Q4_to_Q8(coords: ndarray, topo: ndarray) -> Tuple[ndarray]:
    nP = len(coords)
    edges, edgeIDs = unique_topo_data(edges_Q4(topo))
    new_coords = np.mean(coords[edges], axis=1)
    new_topo = edgeIDs + nP
    topo_res = np.hstack((topo, new_topo))
    coords_res = np.vstack((coords, new_coords))
    return coords_res, topo_res


def Q4_to_Q9(coords: ndarray, topo: ndarray) -> Tuple[ndarray]:
    nP, nE = len(coords), len(topo)
    # new nodes on the edges
    edges, edgeIDs = unique_topo_data(edges_Q4(topo))
    coords_e = np.mean(coords[edges], axis=1)
    topo_e = edgeIDs + nP
    nP += len(coords_e)
    # new coords at element centers
    ecoords = cells_coords(coords, topo)
    coords_c = np.mean(ecoords, axis=1)
    topo_c = np.reshape(np.arange(nE) + nP, (nE, 1))
    # assemble
    topo_res = np.hstack((topo, topo_e, topo_c))
    coords_res = np.vstack((coords, coords_e, coords_c))
    return coords_res, topo_res


def TET4_to_TET10(coords: ndarray, topo: ndarray) -> Tuple[ndarray]:
    nP = len(coords)
    # new nodes on the edges
    edges, edgeIDs = unique_topo_data(edges_TET4(topo))
    coords_e = np.mean(coords[edges], axis=1)
    topo_e = edgeIDs + nP
    topo_res = np.hstack((topo, topo_e))
    coords_res = np.vstack((coords, coords_e))
    return coords_res, topo_res


def TET10_to_TET4(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    subdivide: bool = True,
    **kwargs,
) -> Tuple[ndarray]:
    if isinstance(path, ndarray):
        assert path.shape[1] == 4
    else:
        if path is None:
            if subdivide:
                path = np.array(
                    [
                        [4, 1, 5, 8],
                        [0, 4, 6, 7],
                        [7, 8, 9, 3],
                        [6, 5, 2, 9],
                        [7, 4, 6, 8],
                        [4, 5, 6, 8],
                        [6, 5, 9, 8],
                        [7, 6, 9, 8],
                    ],
                    dtype=topo.dtype,
                )
            else:
                path = np.array([[0, 1, 2, 3]], dtype=topo.dtype)
    if data is None:
        return coords, +transform_topology(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topology(topo, path, data, *args, **kwargs)


def H8_to_H27(coords: ndarray, topo: ndarray) -> Tuple[ndarray]:
    nP, nE = len(coords), len(topo)
    ecoords = cells_coords(coords, topo)
    # new nodes on the edges
    edges, edgeIDs = unique_topo_data(edges_H8(topo))
    coords_e = np.mean(coords[edges], axis=1)
    topo_e = edgeIDs + nP
    nP += len(coords_e)
    # new nodes on face centers
    faces, faceIDs = unique_topo_data(faces_H8(topo))
    coords_f = np.mean(coords[faces], axis=1)
    topo_f = faceIDs + nP
    nP += len(coords_f)
    # register new nodes in the cell centers
    coords_c = np.mean(ecoords, axis=1)
    topo_c = np.reshape(np.arange(nE) + nP, (nE, 1))
    # assemble
    topo_res = np.hstack((topo, topo_e, topo_f, topo_c))
    coords_res = np.vstack((coords, coords_e, coords_f, coords_c))
    return coords_res, topo_res


def W6_to_TET4(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    **kwargs,
):
    if isinstance(path, ndarray):
        assert path.shape[1] == 4
    else:
        if path is None:
            path = np.array(
                [[0, 1, 2, 4], [3, 5, 4, 2], [2, 5, 0, 4]],
                dtype=topo.dtype,
            )
        elif isinstance(path, str):
            raise NotImplementedError
    if data is None:
        return coords, +transform_topology(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topology(topo, path, data, *args, **kwargs)


def W18_to_W6(
    coords: ndarray,
    topo: ndarray,
    data: DataLike = None,
    *args,
    path: ndarray = None,
    subdivide: bool = True,
    **kwargs,
):
    if isinstance(path, ndarray):
        assert path.shape[1] == 6
    else:
        if path is None:
            if subdivide:
                path = np.array(
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
                    dtype=topo.dtype,
                )
            else:
                path = np.array([[0, 1, 2, 3, 4, 5]], dtype=topo.dtype)
        elif isinstance(path, str):
            raise NotImplementedError
    if data is None:
        return coords, +transform_topology(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topology(topo, path, data, *args, **kwargs)


def W6_to_W18(coords: ndarray, topo: ndarray) -> Tuple[ndarray]:
    nP = len(coords)
    # new nodes on the edges
    edges, edgeIDs = unique_topo_data(edges_W6(topo))
    coords_e = np.mean(coords[edges], axis=1)
    topo_e = edgeIDs + nP
    nP += len(coords_e)
    # new nodes on face centers
    faces, faceIDs = unique_topo_data(faces_W6(topo)[0])
    coords_f = np.mean(coords[faces], axis=1)
    topo_f = faceIDs + nP
    nP += len(coords_f)
    # assemble
    topo_res = np.hstack((topo, topo_e, topo_f))
    coords_res = np.vstack((coords, coords_e, coords_f))
    return coords_res, topo_res
