import numpy as np
from numpy import ndarray
from numba import njit, prange

from neumann.linalg import normalize, normalize2d, norm2d
from neumann import atleast2d

from .utils import center_of_points, cell_center, cell_coords
from .knn import k_nearest_neighbours

__cache = True


@njit(nogil=True, cache=__cache)
def frame_of_plane(coords: ndarray):
    """
    Returns the frame of a planar surface. It needs
    at least 3 pointds to work properly (len(coords>=3)).

    It takes the center, the first point and a point from
    the middle to form the coordinate axes.

    Parameters
    ----------
    coords: numpy.ndarray
        2d coordinate array

    Returns
    -------
    numpy.ndarray
        3x3 global -> local DCM matrix
    """
    tr = np.zeros((3, 3), dtype=coords.dtype)
    center = center_of_points(coords)
    tr[:, 0] = normalize(coords[0] - center)
    tr[:, 1] = normalize(coords[np.int(len(coords) / 2)] - center)
    tr[:, 2] = normalize(np.cross(tr[:, 0], tr[:, 1]))
    tr[:, 1] = np.cross(tr[:, 2], tr[:, 0])
    return center, tr


@njit(nogil=True, parallel=True, cache=__cache)
def frames_of_surfaces(coords: ndarray, topo: ndarray):
    """
    Returns the coordinates of the axes forming the local
    coordinate systems of the surfaces.

    Parameters
    ----------
    coords: numpy.ndarray
        2d coordinate array
    topo: numpy.ndarray
        2d point-based topology array

    Returns
    -------
    numpy.ndarray
        3d array of 3x3 transformation matrices
    """
    nE, nNE = topo.shape
    nNE -= 1
    tr = np.zeros((nE, 3, 3), dtype=coords.dtype)
    for iE in prange(nE):
        tr[iE, 0, :] = normalize(coords[topo[iE, 1]] - coords[topo[iE, 0]])
        tr[iE, 1, :] = normalize(coords[topo[iE, nNE]] - coords[topo[iE, 0]])
        tr[iE, 2, :] = normalize(np.cross(tr[iE, 0, :], tr[iE, 1, :]))
        tr[iE, 1, :] = np.cross(tr[iE, 2, :], tr[iE, 0, :])
    return tr


@njit(nogil=True, parallel=True, cache=__cache)
def tr_cell_glob_to_loc_bulk(coords: np.ndarray, topo: np.ndarray):
    """
    Returns the coordinates of the cells in their local coordinate
    system, the coordinates of their centers and the coordinates of
    the axes forming their local coordinate system. The local coordinate
    systems are located at the centers of the cells.

    Parameters
    ----------
    coords: numpy.ndarray
        2d coordinate array
    topo: numpy.ndarray
        2d point-based topology array

    Returns
    -------
    numpy.ndarray
        2d coordinate array of local coordinates
    numpy.ndarray
        2d array of cell centers
    numpy.ndarray
        3d array of 3x3 transformation matrices
    """
    nE, nNE = topo.shape
    tr = np.zeros((nE, 3, 3), dtype=coords.dtype)
    res = np.zeros((nE, nNE, 2), dtype=coords.dtype)
    centers = np.zeros((nE, 3), dtype=coords.dtype)
    for iE in prange(nE):
        centers[iE] = cell_center(cell_coords(coords, topo[iE]))
        tr[iE, 0, :] = normalize(coords[topo[iE, 1]] - coords[topo[iE, 0]])
        tr[iE, 1, :] = normalize(coords[topo[iE, nNE - 1]] - coords[topo[iE, 0]])
        tr[iE, 2, :] = normalize(np.cross(tr[iE, 0, :], tr[iE, 1, :]))
        tr[iE, 1, :] = np.cross(tr[iE, 2, :], tr[iE, 0, :])
        for jN in prange(nNE):
            vj = coords[topo[iE, jN]] - centers[iE]
            res[iE, jN, 0] = np.dot(tr[iE, 0, :], vj)
            res[iE, jN, 1] = np.dot(tr[iE, 1, :], vj)
    return res, centers, tr


@njit(nogil=True, parallel=True, cache=__cache)
def _frames_of_lines_auto(coords: ndarray, topo: ndarray) -> ndarray:
    nE = topo.shape[0]
    ijk = np.eye(3)
    tr = np.zeros((nE, 3, 3), dtype=coords.dtype)
    for iE in prange(nE):
        tr[iE, 0, :] = normalize(coords[topo[iE, -1]] - coords[topo[iE, 0]])
        _dot = ijk @ tr[iE, 0, :]
        i2 = np.argmin(np.absolute(_dot))
        _dot = np.dot(ijk[i2], tr[iE, 0, :])
        tr[iE, 2, :] = normalize(ijk[i2] - tr[iE, 0, :] * _dot)
        tr[iE, 1, :] = np.cross(tr[iE, 2, :], tr[iE, 0, :])
    return tr


@njit(nogil=True, parallel=True, cache=__cache)
def _frames_of_lines_ref(coords: ndarray, topo: ndarray, refZ: ndarray):
    nE, nNE = topo.shape
    nNE -= 1
    tr = np.zeros((nE, 3, 3), dtype=coords.dtype)
    for iE in prange(nE):
        tr[iE, 0, :] = normalize(coords[topo[iE, nNE]] - coords[topo[iE, 0]])
        k = refZ[iE] - coords[topo[iE, 0]]
        tr[iE, 2, :] = normalize(k - tr[iE, 0, :] * np.dot(tr[iE, 0, :], k))
        tr[iE, 1, :] = np.cross(tr[iE, 2, :], tr[iE, 0, :])
    return tr


def frames_of_lines(coords: ndarray, topo: ndarray, refZ: ndarray = None) -> ndarray:
    """
    Returns coordinate frames of line elements defined by a coordinate array
    and a topology array. The cross-sections of the line elements are
    in the local y-z plane. The direction of the local z axis can be set by
    providing reference points in the local x-z planes of the lines. If there
    are no references provided, local z axes lean towards global z.
    Other properties are determined in a way, so that x-y-z form a
    right-handed orthonormal basis.

    Parameters
    ----------
    coords: numpy.ndarray
        2d coordinate array
    topo: numpy.ndarray
        2d point-based topology array
    refZ: numpy.ndarray, Optional
        1d or 2d float array of reference points. If it is 2d, it must
        contain values for all lines defined by `topo`.
        Default is None.

    Returns
    -------
    numpy.ndarray
        3d array of 3x3 transformation matrices
    """
    topo = atleast2d(topo)
    if isinstance(refZ, ndarray):
        if len(topo.shape) == 2 and len(refZ.shape) == 1:
            _refZ = np.zeros((topo.shape[0], 3))
            _refZ[:] = refZ
        else:
            _refZ = refZ
        return _frames_of_lines_ref(coords, topo, _refZ)
    else:
        return _frames_of_lines_auto(coords, topo)


@njit(nogil=True, parallel=True, cache=__cache)
def is_planar_surface(normals: ndarray, tol: float = 1e-8) -> bool:
    """
    Returns true if all the normals point in the same direction.
    The provided normal vectors are assumed to be normalized.

    Parameters
    ----------
    normals: numpy.ndarray
        2d float array of surface normals
    tol: float
        Floating point tolerance as maximum deviation.

    Returns
    -------
    bool
        True if the surfaces whose normal vectors are provided form
        a flat surface, False otherwise.
    """
    nE = normals.shape[0]
    diffs = np.zeros(nE, dtype=normals.dtype)
    for i in prange(1, nE):
        diffs[i] = np.abs(normals[i] @ normals[0] - 1)
    return diffs.max() <= tol


@njit(nogil=True, cache=__cache)
def distances_from_point(
    coords: ndarray, p: ndarray, normalize: bool = False
) -> ndarray:
    if normalize:
        return norm2d(normalize2d(coords - p))
    else:
        return norm2d(coords - p)


@njit(nogil=True, parallel=True, cache=__cache)
def distance_matrix(x: ndarray, y: ndarray) -> ndarray:
    N = x.shape[0]
    M = y.shape[0]
    res = np.zeros((N, M), dtype=x.dtype)
    for n in prange(N):
        for m in prange(M):
            res[n, m] = np.linalg.norm(x[n] - y[m])
    return res


def index_of_closest_point(coords: ndarray, target: ndarray) -> int:
    """
    Returs the index of the point in 'coords', being closest to
    one or more targets.

    Parameters
    ----------
    coords: numpy.ndarray
        2d float array of vertex coordinates.
    target: numpy.ndarray
        1d or 2d coordinate array of the target point(s).

    Returns
    -------
    int or Iterable[int]
        One or more indices of 'coords', for which the distance from
        one or more points described by 'target' is minimal.
    """
    if len(target.shape) == 1:
        assert (
            coords.shape[1] == target.shape[0]
        ), "The dimensions of `coords` and `target` are not compatible."
        return _index_of_closest_point(coords, target)
    else:
        return k_nearest_neighbours(coords, target)


def index_of_furthest_point(coords: ndarray, target: ndarray) -> int:
    """
    Returs the index of the point in 'coords', being furthest from
    one or more targets.

    Parameters
    ----------
    coords: numpy.ndarray
        2d float array of vertex coordinates.
    target: numpy.ndarray
        1d or 2d coordinate array of the target point(s).

    Returns
    -------
    int or Iterable[int]
        One or more indices of 'coords', for which the distance from
        one or more points described by 'target' is maximal.
    """
    if len(target.shape) == 1:
        assert (
            coords.shape[1] == target.shape[0]
        ), "The dimensions of `coords` and `target` are not compatible."
        return _index_of_furthest_point(coords, target)
    else:
        d = distance_matrix(target, coords)
        return np.argmax(d, axis=1)


@njit(nogil=True, cache=__cache)
def _index_of_closest_point(coords: ndarray, p: ndarray) -> int:
    return np.argmin(distances_from_point(coords, p))


@njit(nogil=True, cache=__cache)
def _index_of_furthest_point(coords: ndarray, p: ndarray) -> int:
    return np.argmax(distances_from_point(coords, p))


@njit(nogil=True, parallel=True, cache=__cache)
def is_line(coords: ndarray, tol=1e-8) -> bool:
    """
    Returns true if all the normals point in the same direction.
    The provided normal vectors are assumed to be normalized.

    Parameters
    ----------
    coords: numpy.ndarray
        2d float array of point coordinates
    tol: float
        Floating point tolerance as maximum deviation.

    Returns
    -------
    bool
        True if all absolute deviations from the line between the first
        and the last point is smaller than 'tol'.
    """
    nP = coords.shape[0]
    c = normalize2d(move_points(coords, -coords[0]))
    d = c[-1] - c[0]
    diffs = np.zeros(nP, dtype=coords.dtype)
    for i in prange(1, nP):
        diffs[i] = np.abs(c[i] @ d)
    return diffs.max() <= tol


@njit(nogil=True, parallel=True, cache=__cache)
def is_planar(coords: ndarray, tol: float = 1e-8) -> bool:
    """
    Returns true if all the points fit on a planar surface.

    Parameters
    ----------
    coords: numpy.ndarray
        2d float array of point coordinates
    tol: float
        Floating point tolerance as maximum deviation.

    Returns
    -------
    bool
    """
    nP = coords.shape[0]
    dA = distances_from_point(coords, coords[0])
    iB = np.argmax(dA)
    dB = distances_from_point(coords, coords[iB])
    iC = np.argmax(dA + dB)
    inds = np.array([0, iB, iC])
    d = np.zeros(3)
    d[:] = frame_of_plane(coords[inds, :])[1][:, 2]
    diffs = np.zeros(nP, dtype=coords.dtype)
    for i in prange(1, nP):
        diffs[i] = np.abs(coords[i] @ d)
    return diffs.max() <= tol


@njit(nogil=True, parallel=True, cache=__cache)
def move_points(coords: ndarray, p: ndarray) -> ndarray:
    res = np.zeros_like(coords)
    for i in prange(coords.shape[0]):
        res[i] = coords[i] + p
    return res
