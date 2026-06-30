# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`sigmaepsilon.mesh` is a Python library for building, manipulating, and analyzing polygonal meshes with complex (arbitrarily nested) topologies. It is part of the SigmaEpsilon ecosystem and depends on sibling packages `sigmaepsilon.math`, `sigmaepsilon.deepdict`, and `sigmaepsilon.core` (namespace package `sigmaepsilon`). Performance-critical numeric code is JIT-compiled with Numba; everything else relies on NumPy/Awkward/SciPy. VTK, PyVista, Matplotlib, Plotly, and k3d are optional — guarded by feature flags in `src/sigmaepsilon/mesh/config.py` (`__hasvtk__`, `__haspyvista__`, `__hasmatplotlib__`, `__hasplotly__`, `__hask3d__`, `__hasnx__`). Always check these flags before importing the corresponding optional dependency.

## Development environment

This project uses **uv**, not pip/venv directly.

```console
pip install uv
uv sync --all-groups
```

Dependency groups (`dev`, `test`, `docs`) are declared under `[dependency-groups]` in `pyproject.toml` (PEP 735); pass `--group <name>` to `uv sync` to install a subset (e.g. `uv sync --no-default-groups --group test`), or `--all-groups` for everything.

If experimenting with a new optional library before adding it as a dependency, `uv pip install` it manually into the project's `.venv` — don't add it to `pyproject.toml` until you're sure it's needed (`uv add <pkg> --group test|dev|docs`).

For local development against another sigmaepsilon package (e.g. `sigmaepsilon.math`), install it in editable mode from a sibling directory:

```console
uv add --editable ../sigmaepsilon.math
```

After adding/removing a dependency, regenerate the third-party license file (requires a clean env without optional deps):

```console
uv run pip-licenses --format=plain --output-file=THIRD-PARTY-LICENSES --order=license
```

## Testing

Tests live in `tests/` and mirror the package structure (e.g. `test_mesh_1d.py`, `test_section.py`, `test_voxelization.py`, `test_topo.py`). Notebook-based tests live in `tests/testbooks/` and are run via `testbook` (see `test_testbooks.py`). `pytest.ini` enables `--doctest-modules`, so doctests embedded in docstrings throughout `src/` are also exercised on every test run — keep example code in docstrings runnable.

Run the full suite:

```console
uv run pytest
```

Run a single test file/test:

```console
uv run pytest tests/test_mesh_1d.py
uv run pytest tests/test_mesh_1d.py::test_name -v
```

Run with coverage (also produces `htmlcov/index.html`):

```console
./run_tests_with_coverage.sh                 # JIT enabled (uses .coveragerc)
./run_tests_with_coverage.sh NUMBA_DISABLE_JIT  # JIT disabled (uses .coveragerc_nojit)
```

Numba-jitted (`@njit`/`@jit`) lines are excluded from coverage reporting (see `tox.ini`); when adding numerically heavy code, put the `@njit` kernel in a small free function in `utils/` and keep orchestration/coverage-relevant logic in the calling class.

CI (CircleCI, `.circleci/config.yml`) runs the full test suite on every push and additionally builds/publishes to TestPyPI on the `test` branch and to PyPI on `main`.

## Architecture

### Layering

- `space/` — `PointCloud`, `CartesianFrame`/reference frames (built on `sigmaepsilon.math.linalg`). Foundation for all coordinate handling.
- `data/` — the core data model:
  - `PointData` — nodal data, wraps coordinates + a reference frame.
  - `CellData` / `PolyCell` (`polycell.py`) — base class for all finite-element-like cell containers (topology + nodal mapping + geometry operations). Concrete cell types subclass `PolyCell`.
  - `PolyData` (`polydata.py`) — **the central class of the library**. It is a `DeepDict` (arbitrarily nested dict, from `sigmaepsilon.deepdict`) whose leaves are `PointData`/cell-block pairs. A mesh is a tree of `PolyData` nodes; each leaf node owns a `PolyCell` block sharing the same point cloud. Most mesh-level operations (join/split, nodal distribution, I/O, plotting, search) are methods on `PolyData`.
  - `LineData`/`TriMesh` — specializations of `PolyData` for 1D meshes and triangulations respectively.
  - `AkWrapper` — thin wrapper exposing Awkward Array semantics consistently across `PointData`/`CellData`.
- `cells/` — concrete Lagrangian cell types (`L2`, `L3`, `T3`, `T6`, `Q4`, `Q8`, `Q9`, `H8`, `H27`, `TET4`, `TET10`, `W6`, `W18`), each a `PolyCell` subclass pairing a `Geometry` (from `geometry/`) with shape-function/topology utilities from `utils/cells/`.
- `geometry/` — `PolyCellGeometry1d/2d/3d`: symbolic (SymPy-based) shape function generation, lambdified into fast evaluators, plus topological metadata (VTK/meshio cell IDs, number of nodes/dimensions).
- `typing/` — `Protocol` definitions (`PolyDataProtocol`, `PolyCellProtocol`, `PointDataProtocol`, `CellDataProtocol`, `GeometryProtocol`) and the `ABC_PolyCell` abstract base. These decouple `data/`, `cells/`, and `geometry/` from each other — prefer typing against the protocol, not a concrete class, when writing generic code.
- `utils/` — free functions, many Numba-jitted (`@njit`), doing the actual bulk numeric work (shape functions, Jacobians, point-in-polyhedron tests, KNN, topology transforms). `utils/cells/` and `utils/topology/` are further split by cell type / topological operation. Class methods on `PolyCell`/`PolyData` are generally thin wrappers that prepare arrays and delegate to these.
- `io/`, `plotting/` — conversion to/from VTK, PyVista, meshio, k3d (`io/`) and plotting backends for PyVista/Matplotlib/Plotly (`plotting/`), all conditional on the `config.py` feature flags.
- `domains/section.py` — domain-specific mesh type for cross-sections (wraps `sectionproperties`).
- Top-level helpers: `grid.py` (structured grid generation), `triang.py` (triangulation), `voxelize.py` (tetra/tri voxelization), `extrude.py`, `topoarray.py` (`TopologyArray`, a jagged-array topology container), `indexmanager.py`, `cellapproximator.py` (`LagrangianCellApproximator`), `recipes.py`/`pop.py`/`tile.py`.

### Key conventions

- Meshes are composed like dictionaries: `mesh['A']['Part1'] = PolyData(cd=H27(topo=..., frames=frame))`. Traversal/joining/splitting of the `PolyData` tree relies on `DeepDict` semantics from `sigmaepsilon.deepdict`.
- Nodal data is attached via Awkward Arrays, enabling variable-sized/jagged per-node or per-cell records; this is what `AkWrapper` and `TopologyArray` exist to manage.
- Reference frames (`CartesianFrame`, `ReferenceFrame` from `sigmaepsilon.math.linalg`) are passed explicitly when constructing `PointData`/cells — most geometric ops are frame-aware.
- Generic classes are parameterized over `PointDataLike`/`PolyCellLike`/`MeshDataLike` type variables bound to the corresponding protocols — follow this pattern when adding new cell or mesh container types instead of hardcoding concrete classes.
