"""Benchmark for nested Numba parallel regions (issue #159)."""

# Some kernels call a ``parallel=True`` function from inside the ``prange``
# loop of another ``parallel=True`` function. This crashes with Numba's
# ``workqueue`` threading layer. The fix makes the inner call serial. This
# script compares the two variants ("nested", the current code, and "serial",
# the fix) for every pattern touched by the fix, so the change can be checked
# on each platform before it is merged.
#
# Run it with
#
#     uv run python benchmarks/nested_parallel.py
#
# The "nested" variants abort the process with the ``workqueue`` layer, so on
# macOS run it with OpenMP available (see the installation guide) or set
# ``--serial-only``.

import argparse
import os
import platform
import sys
import time

import numba
import numpy as np
from numba import njit, prange

from sigmaepsilon.mesh.utils.cells.h8 import monoms_H8_single
from sigmaepsilon.mesh.utils.cells.h27 import monoms_H27_single, shp_H27

# ---------------------------------------------------------------------------
# monoms_*_bulk_multi (same pattern for H8, H27, TET4, TET10, W6, W18)
# ---------------------------------------------------------------------------


@njit(nogil=True, parallel=True)
def monoms_H27_multi_par(x):
    nP = x.shape[0]
    res = np.zeros((nP, 27), dtype=x.dtype)
    for i in prange(nP):
        res[i] = monoms_H27_single(x[i])
    return res


@njit(nogil=True, parallel=True)
def monoms_H27_bulk_nested(x):
    nE = x.shape[0]
    res = np.zeros((nE, 27, 27), dtype=x.dtype)
    for i in prange(nE):
        res[i] = monoms_H27_multi_par(x[i])
    return res


@njit(nogil=True, parallel=True)
def monoms_H27_bulk_serial(x):
    nE, nP = x.shape[:2]
    res = np.zeros((nE, nP, 27), dtype=x.dtype)
    for i in prange(nE):
        for j in range(nP):
            res[i, j] = monoms_H27_single(x[i, j])
    return res


@njit(nogil=True, parallel=True)
def monoms_H8_multi_par(x):
    nP = x.shape[0]
    res = np.zeros((nP, 8), dtype=x.dtype)
    for i in prange(nP):
        res[i] = monoms_H8_single(x[i])
    return res


@njit(nogil=True, parallel=True)
def monoms_H8_bulk_nested(x):
    nE = x.shape[0]
    res = np.zeros((nE, 8, 8), dtype=x.dtype)
    for i in prange(nE):
        res[i] = monoms_H8_multi_par(x[i])
    return res


@njit(nogil=True, parallel=True)
def monoms_H8_bulk_serial(x):
    nE, nP = x.shape[:2]
    res = np.zeros((nE, nP, 8), dtype=x.dtype)
    for i in prange(nE):
        for j in range(nP):
            res[i, j] = monoms_H8_single(x[i, j])
    return res


# ---------------------------------------------------------------------------
# shape_function_matrix_*_multi (same pattern for H27 and TET4)
# ---------------------------------------------------------------------------


@njit(nogil=True, parallel=True)
def sfm_H27_par(pcoord, ndof):
    eye = np.eye(ndof, dtype=pcoord.dtype)
    shp = shp_H27(pcoord)
    res = np.zeros((ndof, ndof * 27), dtype=pcoord.dtype)
    for i in prange(27):
        res[:, i * ndof : (i + 1) * ndof] = eye * shp[i]
    return res


@njit(nogil=True, parallel=False)
def sfm_H27_ser(pcoord, ndof):
    eye = np.eye(ndof, dtype=pcoord.dtype)
    shp = shp_H27(pcoord)
    res = np.zeros((ndof, ndof * 27), dtype=pcoord.dtype)
    for i in range(27):
        res[:, i * ndof : (i + 1) * ndof] = eye * shp[i]
    return res


@njit(nogil=True, parallel=True)
def sfm_H27_multi_nested(pcoords, ndof):
    nP = pcoords.shape[0]
    res = np.zeros((nP, ndof, ndof * 27), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = sfm_H27_par(pcoords[iP], ndof)
    return res


@njit(nogil=True, parallel=True)
def sfm_H27_multi_serial(pcoords, ndof):
    nP = pcoords.shape[0]
    res = np.zeros((nP, ndof, ndof * 27), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP] = sfm_H27_ser(pcoords[iP], ndof)
    return res


# ---------------------------------------------------------------------------
# cell_coords inside prange (tr_cell_glob_to_loc_bulk, tri_glob_to_loc)
# ---------------------------------------------------------------------------


@njit(nogil=True, parallel=True)
def cell_coords_par(coords, topo):
    nNE = len(topo)
    res = np.zeros((nNE, coords.shape[1]), dtype=coords.dtype)
    for iNE in prange(nNE):
        res[iNE] = coords[topo[iNE]]
    return res


@njit(nogil=True, parallel=False)
def cell_coords_ser(coords, topo):
    nNE = len(topo)
    res = np.zeros((nNE, coords.shape[1]), dtype=coords.dtype)
    for iNE in range(nNE):
        res[iNE] = coords[topo[iNE]]
    return res


@njit(nogil=True, parallel=True)
def centers_nested(coords, topo):
    nE = topo.shape[0]
    res = np.zeros((nE, coords.shape[1]), dtype=coords.dtype)
    for iE in prange(nE):
        ec = cell_coords_par(coords, topo[iE])
        for k in range(ec.shape[1]):
            res[iE, k] = np.mean(ec[:, k])
    return res


@njit(nogil=True, parallel=True)
def centers_serial(coords, topo):
    nE = topo.shape[0]
    res = np.zeros((nE, coords.shape[1]), dtype=coords.dtype)
    for iE in prange(nE):
        ec = cell_coords_ser(coords, topo[iE])
        for k in range(ec.shape[1]):
            res[iE, k] = np.mean(ec[:, k])
    return res


# ---------------------------------------------------------------------------
# _cells_around_MT_ (norms inside prange)
# ---------------------------------------------------------------------------


@njit(nogil=True, parallel=True)
def norms_par(a):
    nI = len(a)
    res = np.zeros(nI)
    for iI in prange(len(a)):
        res[iI] = np.dot(a[iI], a[iI])
    return np.sqrt(res)


@njit(nogil=True, parallel=True)
def cells_around_nested(centers, r_max, n_max):
    nE = len(centers)
    res = np.zeros((nE, n_max), dtype=np.int64)
    widths = np.zeros(nE, dtype=np.int64)
    for iE in prange(nE):
        inds = np.where(norms_par(centers - centers[iE]) <= r_max)[0]
        if inds.shape[0] <= n_max:
            res[iE, : inds.shape[0]] = inds
            widths[iE] = len(inds)
        else:
            res[iE, :] = inds[:n_max]
            widths[iE] = n_max
    return res, widths


@njit(nogil=True, parallel=True)
def cells_around_serial(centers, r_max, n_max):
    nE, nD = centers.shape
    res = np.zeros((nE, n_max), dtype=np.int64)
    widths = np.zeros(nE, dtype=np.int64)
    for iE in prange(nE):
        n = 0
        for jE in range(nE):
            d2 = 0.0
            for k in range(nD):
                diff = centers[jE, k] - centers[iE, k]
                d2 += diff * diff
            if np.sqrt(d2) <= r_max:
                res[iE, n] = jE
                n += 1
                if n == n_max:
                    break
        widths[iE] = n
    return res, widths


# ---------------------------------------------------------------------------


def best_of(f, args, repeat):
    f(*args)  # compile and warm up
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--serial-only", action="store_true")
    parser.add_argument("--size", type=int, default=20_000, help="number of cells")
    opts = parser.parse_args()

    rng = np.random.default_rng(0)
    nE = opts.size
    coords = rng.random((200_000, 3))
    centers = rng.random((5_000, 3))
    cases = [
        (
            "monoms_H27_bulk_multi",
            monoms_H27_bulk_nested,
            monoms_H27_bulk_serial,
            (rng.random((nE, 27, 3)),),
        ),
        (
            "monoms_H8_bulk_multi",
            monoms_H8_bulk_nested,
            monoms_H8_bulk_serial,
            (rng.random((10 * nE, 8, 3)),),
        ),
        (
            "shape_function_matrix_H27_multi",
            sfm_H27_multi_nested,
            sfm_H27_multi_serial,
            (rng.random((nE, 3)), 3),
        ),
        (
            "cell_coords in prange",
            centers_nested,
            centers_serial,
            (coords, rng.integers(0, len(coords), (10 * nE, 8))),
        ),
        (
            "_cells_around_MT_",
            cells_around_nested,
            cells_around_serial,
            (centers, 0.1, 10),
        ),
    ]

    # check that the variants agree before timing them
    if not opts.serial_only:
        for name, f_nested, f_serial, args in cases:
            print(f"checking {name}", flush=True)
            a, b = f_nested(*args), f_serial(*args)
            for x, y in zip(a, b) if isinstance(a, tuple) else [(a, b)]:
                if not np.allclose(x, y):
                    raise RuntimeError(f"{name}: the nested and serial variants differ")

    print(f"platform        : {platform.platform()} ({platform.machine()})")
    print(f"python          : {sys.version.split()[0]}")
    print(f"numba           : {numba.__version__}")
    print(f"threading layer : {numba.threading_layer()}")
    print(f"threads         : {numba.get_num_threads()} (cpu_count={os.cpu_count()})")
    print()
    print(f"{'kernel':<34}{'nested [ms]':>12}{'serial [ms]':>12}{'speedup':>9}")
    for name, f_nested, f_serial, args in cases:
        t_s = best_of(f_serial, args, opts.repeat)
        if opts.serial_only:
            print(f"{name:<34}{'-':>12}{t_s * 1e3:>12.2f}{'-':>9}")
            continue
        t_n = best_of(f_nested, args, opts.repeat)
        print(f"{name:<34}{t_n * 1e3:>12.2f}{t_s * 1e3:>12.2f}{t_n / t_s:>8.2f}x", flush=True)


if __name__ == "__main__":
    main()
