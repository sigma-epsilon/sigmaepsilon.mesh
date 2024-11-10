# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
from numpy import ndarray
from numba import njit, prange, vectorize

from sigmaepsilon.math.linalg import normalize, norm

from ..utils.utils import cells_coords, cell_coords

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def triangulate_cell_coords(ecoords: ndarray, trimap: ndarray) -> ndarray:
    nE = ecoords.shape[0]
    nTE, nNTE = trimap.shape
    nT = int(nE * nTE)
    nD = ecoords.shape[-1]
    res = np.zeros((nT, nNTE, nD), dtype=ecoords.dtype)
    for iE in prange(nE):
        for iTE in prange(nTE):
            iT = iE * nTE + iTE
            for iNTE in prange(nNTE):
                res[iT, iNTE, :] = ecoords[iE, trimap[iTE, iNTE], :]
    return res


@njit(nogil=True, cache=__cache)
def monoms_tri_loc(lcoord: ndarray) -> ndarray:
    return np.array([1, lcoord[0], lcoord[1]], dtype=lcoord.dtype)


@njit(nogil=True, cache=__cache)
def monoms_tri_loc_bulk(lcoord: ndarray) -> ndarray:
    res = np.ones((lcoord.shape[0], 3), dtype=lcoord.dtype)
    res[:, 1] = lcoord[:, 0]
    res[:, 2] = lcoord[:, 1]
    return res


@njit(nogil=True, cache=__cache)
def lcoords_tri(center: ndarray = None) -> ndarray:
    """
    Returns the local coordinates (r, s) of the vertices of a triangle.

    By default, it is assumed that the origo of the (r, s) system is at
    the geometric center of the triangle, unless the coordinates of geometric
    center are provided with the argument 'center'.

    Example
    -------
    >>> import numpy as np
    >>> from sigmaepsilon.mesh.utils.tri import lcoords_tri
    >>> lcoords = lcoords_tri(np.array([1/3, 1/3]))
    """
    res = np.array([[-1 / 3, -1 / 3], [2 / 3, -1 / 3], [-1 / 3, 2 / 3]])
    if center is not None:
        res += center
    return res


@njit(nogil=True, cache=__cache)
def ncenter_tri() -> ndarray:
    """
    Returns the area coordinates of the geometric center of the
    master triangle.
    """
    return np.array([1 / 3, 1 / 3, 1 / 3])


@njit(nogil=True, cache=__cache)
def shp_tri_loc(lcoord: ndarray, center: ndarray = None) -> ndarray:
    """
    Evaluates the shape functions at the parametric coordinates (r, s).

    By default, it is assumed that the origo of the (r, s) system is at
    the geometric center of the triangle, unless the coordinates of geometric
    center are provided with the argument 'center'.

    Example
    -------
    For a master triangle with centroid at the first vertex:
    >>> import numpy as np
    >>> from sigmaepsilon.mesh.utils.tri import shp_tri_loc
    >>> A1, A2, A3 = shp_tri_loc(np.array([0.0, 0.0]), np.array([1/3, 1/3]))
    """
    r, s = lcoord
    M = np.ones((3, 3), dtype=lcoord.dtype)
    M[1:, :] = lcoords_tri(center).T
    return np.linalg.inv(M) @ np.array([1, r, s], dtype=lcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_tri_loc(
    lcoord: ndarray, nDOFN: int = 2, center: ndarray = None
) -> ndarray:
    eye = np.eye(nDOFN, dtype=lcoord.dtype)
    shp = shp_tri_loc(lcoord, center)
    res = np.zeros((nDOFN, 3 * nDOFN), dtype=lcoord.dtype)
    for i in prange(3):
        res[:, i * 3 : (i + 1) * 3] = eye * shp[i]
    return res


@njit(nogil=True, cache=__cache)
def center_tri_2d(ecoords: ndarray) -> ndarray:
    """Calculates the center of a single triangle in 2d space."""
    return np.array(
        [np.mean(ecoords[:, 0]), np.mean(ecoords[:, 1])], dtype=ecoords.dtype
    )


@njit(nogil=True, cache=__cache)
def center_tri_3d(ecoords: ndarray) -> ndarray:
    """Calculates the center of a single triangle in 3d space."""
    return np.array(
        [np.mean(ecoords[:, 0]), np.mean(ecoords[:, 1]), np.mean(ecoords[:, 2])],
        dtype=ecoords.dtype,
    )


@njit(nogil=True, parallel=True, cache=__cache)
def center_tri_bulk_3d(
    points: ndarray[float], triangles: ndarray[int]
) -> ndarray[float]:
    """Calculates centers of triangles in 3d space."""
    out = np.zeros((len(triangles), 3), dtype=points.dtype)
    for i in prange(len(triangles)):
        out[i] = center_tri_3d(points[triangles[i]])
    return out


@njit(nogil=True, cache=__cache)
def area_tri(ecoords: ndarray) -> ndarray:
    """
    Returns the the signed area of a single 3-noded triangle.

    Parameters
    ----------
    ecoords: numpy.ndarray
        Element coordinates.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.

    Returns
    -------
    float
        Returns a positive number if the vertices are listed counterclockwise,
        negative if they are listed clockwise.
    """
    A = (
        (ecoords[1, 0] * ecoords[2, 1] - ecoords[2, 0] * ecoords[1, 1])
        + (ecoords[2, 0] * ecoords[0, 1] - ecoords[0, 0] * ecoords[2, 1])
        + (ecoords[0, 0] * ecoords[1, 1] - ecoords[1, 0] * ecoords[0, 1])
    )
    return A / 2


@njit(nogil=True, cache=__cache)
def inscribed_radius(ecoords: ndarray) -> ndarray:
    """
    Returns the radius of the inscribed circle of a single triangle.

    Parameters
    ----------
    ecoords: numpy.ndarray
        2d float numpy array of element coordinates.

    Notes
    -----
    If the sides have length a, b, c, we define the semiperimeter s
    to be half their sum, so s = (a+b+c)/2. Given this, the radius is
    given using the following:

        r2 = (s - a)*(s - b)*(s - c) / s.

    This function is numba-jittable in 'nopython' mode.

    Returns
    -------
    float
    """
    a = norm(ecoords[1] - ecoords[0])
    b = norm(ecoords[2] - ecoords[1])
    c = norm(ecoords[2] - ecoords[0])
    s = (a + b + c) / 2
    return np.sqrt((s - a) * (s - b) * (s - c) / s)


@njit(nogil=True, parallel=True, cache=__cache)
def inscribed_radii(ecoords: ndarray) -> ndarray:
    """
    Returns the radii of the inscribed circle of several triangles.

    Parameters
    ----------
    ecoords: numpy.ndarray
        3d float numpy array of element coordinates for multiple cells.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.

    Returns
    -------
    numpy.ndarray
        1d numpy float array
    """
    nE = ecoords.shape[0]
    res = np.zeros(nE)
    for i in prange(nE):
        res[i] = inscribed_radius(ecoords[i])
    return res


@njit(nogil=True, parallel=False, cache=__cache)
def areas_tri(ecoords: ndarray) -> ndarray:
    """
    Returns the total sum of signed areas of several triangles.

    Parameters
    ----------
    ecoords: numpy.ndarray
        3d float numpy array of element coordinates for multiple cells.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.

    Returns
    -------
    float
        The sum of areas of all triangles.
    """
    A = 0.0
    nE = len(ecoords)
    for i in prange(nE):
        A += (
            (ecoords[i, 1, 0] * ecoords[i, 2, 1] - ecoords[i, 2, 0] * ecoords[i, 1, 1])
            + (
                ecoords[i, 2, 0] * ecoords[i, 0, 1]
                - ecoords[i, 0, 0] * ecoords[i, 2, 1]
            )
            + (
                ecoords[i, 0, 0] * ecoords[i, 1, 1]
                - ecoords[i, 1, 0] * ecoords[i, 0, 1]
            )
        )
    return A / 2


@njit(nogil=True, parallel=True, cache=__cache)
def area_tri_bulk(ecoords: ndarray) -> ndarray:
    """
    Returns the signed area of several triangles.

    Parameters
    ----------
    ecoords: numpy.ndarray
        3d float numpy array of element coordinates for multiple cells.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.

    Returns
    -------
    numpy.ndarray
        1d numpy float array
    """
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    for i in prange(nE):
        res[i] = (
            (ecoords[i, 1, 0] * ecoords[i, 2, 1] - ecoords[i, 2, 0] * ecoords[i, 1, 1])
            + (
                ecoords[i, 2, 0] * ecoords[i, 0, 1]
                - ecoords[i, 0, 0] * ecoords[i, 2, 1]
            )
            + (
                ecoords[i, 0, 0] * ecoords[i, 1, 1]
                - ecoords[i, 1, 0] * ecoords[i, 0, 1]
            )
        )
    return res / 2


@vectorize("f8(f8, f8, f8, f8, f8, f8)", target="parallel", cache=__cache)
def area_tri_u(x1, y1, x2, y2, x3, y3) -> float:
    """
    Vectorized implementation of `area_tri_bulk`.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    return (x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3 + x1 * y2 - x2 * y1) / 2


@vectorize("f8(f8, f8, f8, f8, f8, f8)", target="parallel", cache=__cache)
def area_tri_u2(x1, x2, x3, y1, y2, y3) -> float:
    """
    Another vectorized implementation of `area_tri_bulk` with a different
    order of arguments.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    return (x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3 + x1 * y2 - x2 * y1) / 2


@njit(nogil=True, cache=__cache)
def loc_to_glob_tri(
    lcoord: ndarray, gcoords: ndarray, center: ndarray = None
) -> ndarray:
    """
    Transformation from local to global coordinates within a triangle.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    return gcoords.T @ shp_tri_loc(lcoord, center)


@njit(nogil=True, cache=__cache)
def glob_to_loc_tri(
    gcoord: ndarray, gcoords: ndarray, center: ndarray = None
) -> ndarray:
    """
    Transformation from global to local coordinates within a triangle.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    monoms = monoms_tri_loc_bulk(gcoords)
    coeffs = np.linalg.inv(monoms)
    shp = coeffs.T @ monoms_tri_loc(gcoord)
    return lcoords_tri(center).T @ shp


@njit(nogil=True, cache=__cache)
def glob_to_nat_tri(gcoord: ndarray, ecoords: ndarray) -> ndarray:
    """
    Transformation from global to natural coordinates within a triangle.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    x, y = gcoord[0:2]
    (x1, x2, x3), (y1, y2, y3) = ecoords[:, 0], ecoords[:, 1]
    A2 = np.abs(x2 * (y3 - y1) + x1 * (y2 - y3) + x3 * (y1 - y2))
    n1 = (x2 * y3 - x * y3 - x3 * y2 + x * y2 + x3 * y - x2 * y) / A2
    n2 = (x * y3 - x1 * y3 + x3 * y1 - x * y1 - x3 * y + x1 * y) / A2
    return np.array([n1, n2, 1 - n1 - n2], dtype=gcoord.dtype)


@njit(nogil=True, parallel=True, cache=__cache)
def _glob_to_nat_tri_bulk_(points: ndarray, ecoords: ndarray) -> ndarray:
    nE = ecoords.shape[0]
    nP = points.shape[0]
    res = np.zeros((nP, nE, 3), dtype=points.dtype)
    for i in prange(nP):
        for j in prange(nE):
            res[i, j] = glob_to_nat_tri(points[i], ecoords[j])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def __pip_tri_bulk__(nat: ndarray, tol: float = 1e-12) -> ndarray:
    nP, nE = nat.shape[:2]
    res = np.zeros((nP, nE), dtype=np.bool_)
    for i in prange(nP):
        for j in prange(nE):
            c1 = np.all(nat[i, j] > (-tol))
            c2 = np.all(nat[i, j] < (1 + tol))
            res[i, j] = c1 and c2
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _pip_tri_bulk_(points: ndarray, ecoords: ndarray, tol: float = 1e-12) -> ndarray:
    nat = _glob_to_nat_tri_bulk_(points, ecoords)
    return __pip_tri_bulk__(nat, tol)


@njit(nogil=True, parallel=True, cache=__cache)
def _glob_to_nat_tri_bulk_knn_(
    points: ndarray, ecoords: ndarray, neighbours: ndarray
) -> ndarray:
    kE = neighbours.shape[1]
    nP = points.shape[0]
    res = np.zeros((nP, kE, 3), dtype=points.dtype)
    for i in prange(nP):
        for k in prange(kE):
            res[i, k, :] = glob_to_nat_tri(points[i], ecoords[neighbours[i, k]])
    return res


@njit(nogil=True, cache=__cache)
def _pip_tri_bulk_knn_(
    points: ndarray, ecoords: ndarray, neighbours: ndarray, tol: float = 1e-12
) -> ndarray:
    nat = _glob_to_nat_tri_bulk_knn_(points, ecoords, neighbours)
    return __pip_tri_bulk__(nat, tol)


@njit(nogil=True, cache=__cache)
def nat_to_glob_tri(ncoord: ndarray, ecoords: ndarray) -> ndarray:
    """
    Transformation from natural to global coordinates within a triangle.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    return ecoords.T @ ncoord


@njit(nogil=True, cache=__cache)
def loc_to_nat_tri(lcoord: ndarray, center: ndarray = None) -> ndarray:
    """
    Transformation from local to natural coordinates within a triangle.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    return shp_tri_loc(lcoord, center)


@njit(nogil=True, cache=__cache)
def nat_to_loc_tri(
    acoord: ndarray, lcoords: ndarray = None, center: ndarray = None
) -> ndarray:
    """
    Transformation from natural to local coordinates within a triangle.

    Parameters
    ----------
    acoord: numpy.ndarray
        1d NumPy array of area coordinates of a point.
    lcoords: numpy.ndarray, Optional
        2d NumPy array of parametric coordinates (r, s) of the
        master cell of a triangle.
    center: numpy.ndarray
        The local coordinates (r, s) of the geometric center
        of the master triangle. If not provided it is assumed to
        be at (0, 0).

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    if lcoords is None:
        lcoords = lcoords_tri(center)
    return acoord.T @ lcoords


@njit(nogil=True, parallel=True, cache=__cache)
def localize_points(
    points: ndarray, triangles: ndarray, coords: ndarray
) -> Tuple[ndarray, ndarray]:
    nE = triangles.shape[0]
    nC = coords.shape[0]
    ecoords = cells_coords(points, triangles)
    res = np.full(nC, -1, dtype=triangles.dtype)
    shp = np.zeros((nC, 3), dtype=points.dtype)
    for iC in prange(nC):
        for iE in prange(nE):
            nat = glob_to_nat_tri(coords[iC], ecoords[iE])
            if np.max(nat) <= 1.0:
                res[iC] = iE
                shp[iC] = nat
                break
    return res, shp


@njit(nogil=True, parallel=True, cache=__cache)
def _get_points_inside_triangles(
    points: ndarray, topo: ndarray, coords: ndarray
) -> ndarray:
    inds, _ = localize_points(points, topo, coords)
    inds[inds > -1] = 1
    inds[inds < 0] = 0
    return inds


def get_points_inside_triangles(
    points: ndarray, topo: ndarray, coords: ndarray
) -> ndarray:
    return _get_points_inside_triangles(points, topo, coords).astype(bool)


@njit(nogil=True, parallel=True, cache=__cache)
def approx_data_to_points(
    points: ndarray,
    triangles: ndarray,
    data: ndarray,
    coords: ndarray,
    defval: float = 0.0,
) -> ndarray:
    nC = coords.shape[0]
    nD = data.shape[1]
    inds, shp = localize_points(points, triangles, coords)
    res = np.full((nC, nD), defval, dtype=data.dtype)
    for iC in prange(nC):
        i = inds[iC]
        if i > -1:
            for j in prange(nD):
                res[i, j] = np.sum(data[triangles[i], j] * shp[iC])
    return res


def offset_tri(coords: ndarray, topo: ndarray, data: ndarray) -> ndarray:
    if isinstance(data, ndarray):
        alpha = np.abs(data)
        amax = alpha.max()
        if amax > 1.0:
            alpha /= amax
        return _offset_tri_(coords, topo, alpha)
    elif isinstance(data, float):
        alpha = min(abs(data), 1.0)
        return offset_tri_uniform(coords, topo, alpha)
    else:
        raise RuntimeError


@njit(nogil=True, cache=__cache)
def offset_tri_uniform(coords: ndarray, topo: ndarray, alpha: float = 0.9) -> ndarray:
    cellcoords = cells_coords(coords, topo)
    ncenter = ncenter_tri(coords.dtype)
    eye = np.eye(3, dtype=coords.dtype)
    ncoords = ncenter + (eye - ncenter) * alpha
    nE = len(topo)
    res = np.zeros(cellcoords.shape, dtype=cellcoords.dtype)
    ncoords = ncoords.astype(cellcoords.dtype)
    for iE in prange(nE):
        res[iE, :, :] = ncoords @ cellcoords[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _offset_tri_(coords: ndarray, topo: ndarray, alpha: ndarray) -> ndarray:
    cellcoords = cells_coords(coords, topo)
    ncenter = ncenter_tri()
    dn = np.eye(3, dtype=coords.dtype) - ncenter
    nE = len(topo)
    res = np.zeros(cellcoords.shape, dtype=cellcoords.dtype)
    alpha = alpha.astype(cellcoords.dtype)
    for iE in prange(nE):
        ncoords = ncenter + dn * alpha[iE]
        res[iE, :, :] = ncoords @ cellcoords[iE]
    return res


def edges_tri(triangles: ndarray) -> ndarray:
    shp = triangles.shape
    if len(shp) == 2:
        return _edges_tri(triangles)
    elif len(shp) == 3:
        return _edges_tri_pop(triangles)
    else:
        raise NotImplementedError


@njit(nogil=True, cache=__cache)
def _edges_tri(triangles: ndarray) -> ndarray:
    nE = len(triangles)
    edges = np.zeros((nE, 3, 2), dtype=triangles.dtype)
    edges[:, 0, 0] = triangles[:, 0]
    edges[:, 0, 1] = triangles[:, 1]
    edges[:, 1, 0] = triangles[:, 1]
    edges[:, 1, 1] = triangles[:, 2]
    edges[:, 2, 0] = triangles[:, 2]
    edges[:, 2, 1] = triangles[:, 0]
    return edges


@njit(nogil=True, parallel=True, cache=__cache)
def _edges_tri_pop(triangles: ndarray) -> ndarray:
    nPop, nE, _ = triangles.shape
    res = np.zeros((nPop, nE, 3, 2), dtype=triangles.dtype)
    for i in prange(nPop):
        res[i] = _edges_tri(triangles[i])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tri_glob_to_loc(
    points: ndarray, triangles: ndarray
) -> Tuple[ndarray, ndarray, ndarray]:
    nE = triangles.shape[0]
    tr = np.zeros((nE, 3, 3), dtype=points.dtype)
    res = np.zeros((nE, 3, 2), dtype=points.dtype)
    centers = np.zeros((nE, 3), dtype=points.dtype)
    for iE in prange(nE):
        centers[iE] = center_tri_3d(cell_coords(points, triangles[iE]))
        tr[iE, 0, :] = normalize(points[triangles[iE, 1]] - points[triangles[iE, 0]])
        tr[iE, 1, :] = normalize(points[triangles[iE, 2]] - points[triangles[iE, 0]])
        tr[iE, 2, :] = np.cross(tr[iE, 0, :], tr[iE, 1, :])
        tr[iE, 1, :] = np.cross(tr[iE, 2, :], tr[iE, 0, :])
        for jN in prange(3):
            vj = points[triangles[iE, jN]] - centers[iE]
            res[iE, jN, 0] = np.dot(tr[iE, 0, :], vj)
            res[iE, jN, 1] = np.dot(tr[iE, 1, :], vj)
    return res, centers, tr


if __name__ == "__main__":
    from sigmaepsilon.mesh.triang import triangulate
    from sigmaepsilon.mesh.utils.space import frames_of_surfaces, is_planar_surface
    from sigmaepsilon.math import ascont
    from time import time

    points, triangles, triobj = triangulate(size=(800, 600), shape=(100, 100))
    tricoords = cells_coords(points, triangles)

    area0 = 800 * 600

    t0 = time()
    for i in range(10):
        area1 = np.sum(area_tri_bulk(tricoords))
    print(time() - t0)

    t0 = time()
    for i in range(10):
        area2 = areas_tri(tricoords)
    print(time() - t0)

    x1 = tricoords[:, 0, 0]
    x2 = tricoords[:, 1, 0]
    x3 = tricoords[:, 2, 0]
    y1 = tricoords[:, 0, 1]
    y2 = tricoords[:, 1, 1]
    y3 = tricoords[:, 2, 1]
    t0 = time()
    for i in range(10):
        area3 = np.sum(area_tri_u2(x1, x2, x3, y1, y2, y3))
    print(time() - t0)

    for i in range(10):
        area4 = np.sum(area_tri_u(x1, y1, x2, y2, y3, y3))
    print(time() - t0)

    tri_glob_to_loc(points, triangles)
    frames = frames_of_surfaces(points, triangles)
    normals = ascont(frames[:, 2, :])
    print(is_planar_surface(normals))
