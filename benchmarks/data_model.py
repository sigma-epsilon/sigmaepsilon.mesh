"""Benchmarks for the data model proposal (P0.3 of the implementation plan)."""

# Measures the numbers that section 2 of ``design/data-model-proposal.md``
# quotes, so the phases that change the storage can show their effect
# against the same script:
#
# - ``to_ak`` on 216k H8 cells, which goes through Python lists (section 2.7,
#   quoted as 2.9 s);
# - memory of per-cell frames, which are stored expanded to (nE, 3, 3) even
#   when every cell shares one frame (section 2.4, quoted as 2160 bytes for
#   30 cells);
# - ``topology()`` and ``volumes()`` on a large mixed subtree, which walk the
#   blocks of the subtree (section 2.9).
#
# Run it with
#
#     uv run python benchmarks/data_model.py
#
# and use ``--json results.json`` to keep the numbers for a later comparison.
# The script is not part of CI. Timings are the best of ``--repeat`` runs,
# after ``--warmup`` runs that are not timed (they include JIT compilation).

import argparse
import json
import platform
import time
from importlib.metadata import version

import numpy as np

from sigmaepsilon.mesh import CartesianFrame, PointData, PolyData
from sigmaepsilon.mesh.cells import H8, L2
from sigmaepsilon.mesh.grid import grid
from sigmaepsilon.mesh.utils.topology import H8_to_L2


def best_time(func, repeat: int, warmup: int) -> float:
    """Returns the best wall time of `repeat` calls of `func`, in seconds."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


def solid_mesh(n: int) -> PolyData:
    """A unit cube of n x n x n H8 cells in one block under the root."""
    frame = CartesianFrame(dim=3)
    coords, topology = grid(size=(1, 1, 1), shape=(n, n, n), eshape="H8")
    mesh = PolyData(PointData(coords=coords, frame=frame))
    mesh["Solids"] = PolyData(H8(topo=topology, frames=frame))
    return mesh


def mixed_mesh(n: int) -> PolyData:
    """
    The mesh of `solid_mesh` plus L2 beams along the edges of the bottom
    layer of cells, all sharing the points of the root (section 2.9 at scale).
    """
    frame = CartesianFrame(dim=3)
    coords, topology = grid(size=(1, 1, 1), shape=(n, n, n), eshape="H8")
    bottom = topology[coords[topology][:, :, 2].min(axis=1) == 0]
    _, beams = H8_to_L2(coords, bottom)
    mesh = PolyData(PointData(coords=coords, frame=frame))
    mesh["Solids"] = PolyData(H8(topo=topology, frames=frame))
    mesh["Beams"] = PolyData(L2(topo=beams, frames=frame))
    return mesh


def bench_to_ak(n: int, repeat: int, warmup: int) -> dict:
    mesh = solid_mesh(n)
    seconds = best_time(mesh.to_ak, repeat, warmup)
    return {"cells": len(mesh["Solids"].cd), "seconds": seconds}


def bench_frames(n: int) -> dict:
    """Bytes held by the frames of a block whose cells share one frame."""
    frame = CartesianFrame(dim=3)
    lines = np.stack([np.arange(30), np.arange(1, 31)], axis=1)
    small = L2(topo=lines, frames=frame)
    large = solid_mesh(n)["Solids"].cd
    return {
        "shared_frame_bytes": frame.axes.nbytes,
        "l2_30_cells_bytes": small.frames.nbytes,
        "h8_cells": len(large),
        "h8_bytes": large.frames.nbytes,
    }


def bench_subtree(n: int, repeat: int, warmup: int) -> dict:
    mesh = mixed_mesh(n)
    return {
        "h8_cells": len(mesh["Solids"].cd),
        "l2_cells": len(mesh["Beams"].cd),
        "topology_seconds": best_time(mesh.topology, repeat, warmup),
        "volumes_seconds": best_time(mesh.volumes, repeat, warmup),
    }


def environment() -> dict:
    packages = ["sigmaepsilon.mesh", "sigmaepsilon.math", "numpy", "awkward", "numba"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        **{name: version(name) for name in packages},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        default=60,
        help="cells per side of the H8 grid (default: 60, i.e. 216k cells)",
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="timed runs per measurement (default: 3)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="untimed runs before timing (default: 1)"
    )
    parser.add_argument(
        "--json", metavar="PATH", help="also write the results to a JSON file"
    )
    args = parser.parse_args()

    n, repeat, warmup = args.size, args.repeat, args.warmup
    results = {
        "environment": environment(),
        "parameters": {"size": n, "repeat": repeat, "warmup": warmup},
        "to_ak": bench_to_ak(n, repeat, warmup),
        "frames": bench_frames(n),
        "subtree": bench_subtree(n, repeat, warmup),
    }

    to_ak, frames, subtree = results["to_ak"], results["frames"], results["subtree"]
    print(f"to_ak, {to_ak['cells']} H8 cells: {to_ak['seconds']:.3f} s")
    print(
        f"frames, one shared frame: {frames['shared_frame_bytes']} bytes, "
        f"stored for 30 L2 cells: {frames['l2_30_cells_bytes']} bytes, "
        f"for {frames['h8_cells']} H8 cells: {frames['h8_bytes'] / 2**20:.1f} MiB"
    )
    print(
        f"subtree of {subtree['h8_cells']} H8 + {subtree['l2_cells']} L2 cells: "
        f"topology() {subtree['topology_seconds']:.3f} s, "
        f"volumes() {subtree['volumes_seconds']:.3f} s"
    )

    if args.json:
        with open(args.json, "w") as file:
            json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
