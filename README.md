# **SigmaEpsilon.Mesh**

![ ](https://github.com/sigma-epsilon/sigmaepsilon.mesh/blob/main/logo.png?raw=true)

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/sigma-epsilon/sigmaepsilon.mesh/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/sigma-epsilon/sigmaepsilon.mesh/tree/main)
[![codecov](https://codecov.io/gh/sigma-epsilon/sigmaepsilon.mesh/graph/badge.svg?token=7JKJ3HHSX3)](https://codecov.io/gh/sigma-epsilon/sigmaepsilon.mesh)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f25dd5fe8d8a484ba27c082923461d72)](https://app.codacy.com/gh/sigma-epsilon/sigmaepsilon.mesh/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Documentation Status](https://readthedocs.org/projects/sigmaepsilonmesh/badge/?version=latest)](https://sigmaepsilonmesh.readthedocs.io/en/latest/?badge=latest)
![[License]](https://badgen.net/pypi/license/sigmaepsilon.mesh)
![[PyPI - Version]](https://badgen.net/pypi/v/sigmaepsilon.mesh)
![[Python]](https://badgen.net/pypi/python/sigmaepsilon.mesh)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Requirements Status](https://dependency-dash.repo-helper.uk/github/sigma-epsilon/sigmaepsilon.mesh/badge.svg)](https://dependency-dash.repo-helper.uk/github/sigma-epsilon/sigmaepsilon.mesh)

**Build, manipulate, and analyze polygonal meshes** — from simple grids to arbitrarily nested topologies — with a Numba-accelerated, dictionary-style data model.

[sigmaepsilon.mesh](https://sigmaepsilonmesh.readthedocs.io/en/latest/) lets you compose meshes like nested dictionaries, attach structured or variable-sized data to any point or cell using [Awkward Array](https://awkward-array.org/), and export to virtually any format. Performance-critical operations are JIT-compiled with [Numba](https://numba.pydata.org/), making it suitable for large-scale problems.

## Highlights

### Data Model

- Compose meshes like nested dictionaries with arbitrarily complex topologies — join, split, and traverse using familiar Python idioms.
- Attach nested, variable-sized data to points or cells via [Awkward Array](https://awkward-array.org/), with built-in interfaces to [Pandas](https://pandas.pydata.org/) and [PyArrow](https://arrow.apache.org/docs/python/index.html).
- Save and load mesh data to and from disk (Parquet, VTK, and more).

### Performance

- Computationally intensive operations are JIT-compiled with [Numba](https://numba.pydata.org/) for speed even on large meshes.
- Built on [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) throughout.

### Geometry & Topology

- Grid generation in 1, 2, and 3 dimensions for 13 Lagrangian cell types (`L2`, `T3`, `Q4`, `H8`, `TET4`, and more).
- Symbolic shape function generation via [SymPy](https://www.sympy.org/), lambdified into fast evaluators.
- Voxelization for tetrahedral and triangular meshes.
- Customizable nodal distribution — pass data between points and cells effortlessly.
- Pseudo peripheral nodes, rooted level structures, and adjacency matrices for arbitrary polygonal meshes.
- Geometric and topological transformations: translate, rotate, extrude, tile, rewire.

### I/O & Visualization

- Read from and write to a wide range of formats via [VTK](https://vtk.org/), [PyVista](https://docs.pyvista.org/), and [meshio](https://github.com/nschloe/meshio).
- Plot with [PyVista](https://docs.pyvista.org/), [K3D](http://k3d-jupyter.org/), [Matplotlib](https://matplotlib.org/), or [Plotly](https://plotly.com/python/).

## Quick Start

```python
from sigmaepsilon.mesh import PolyData, PointData, grid
from sigmaepsilon.mesh.cells import H8

# Generate a structured 4×4×4 hex mesh
coords, topo = grid(size=(1, 1, 1), shape=(4, 4, 4), eshape="H8")

# Compose mesh like a dictionary
mesh = PolyData(pd=PointData(coords=coords), cd=H8(topo=topo))

# Attach data to nodes or cells, then plot
mesh.plot()
```

Meshes can be composed from multiple cell blocks and nested arbitrarily:

```python
mesh = PolyData()
mesh["Part A"] = PolyData(pd=PointData(coords=coords_a), cd=H8(topo=topo_a))
mesh["Part B"] = PolyData(pd=PointData(coords=coords_b), cd=TET4(topo=topo_b))
```

## Installation

```console
pip install sigmaepsilon.mesh
```

This installs the core library. For full visualization and I/O support, install the optional dependencies:

```console
pip install sigmaepsilon.mesh vtk pyvista meshio
```

For the complete list of optional extras and development setup, refer to the [installation guide](https://sigmaepsilonmesh.readthedocs.io/en/latest/installation.html).

## Documentation

The [documentation](https://sigmaepsilonmesh.readthedocs.io/en/latest/) is built with [Sphinx](https://www.sphinx-doc.org/en/master/) using the [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html) and hosted on [ReadTheDocs](https://readthedocs.org/). It includes:

- **User Guide** — concepts, data model, and how-tos
- **Example Gallery** — 20+ worked examples from simple grids to DEM and Minecraft-style models
- **API Reference** — full docstring coverage

## Projects using sigmaepsilon.mesh

- Many of the other packages in the SigmaEpsilon ecosystem.
- [PyAxisVM](https://github.com/AxisVM/pyaxisvm) — the official Python package of [AxisVM](https://axisvm.eu/), a popular structural analysis and design software.

## Related blog posts and articles

- [Best Python packages in 2024 for civil and structural engineers. Part I — Visualization.](https://medium.com/structural-and-civil-engineering/best-python-packages-in-2024-for-civil-and-structural-engineers-0d719cc61b7f?sk=0432097e3e4f1c206d7869ce3a170be1) (Medium)
- [Here is how to make digital elevation maps in Python in a matter of minutes using TouchTerrain and SigmaEpsilon.Mesh with a few lines of code.](https://medium.com/structural-and-civil-engineering/here-is-how-to-make-digital-elevation-maps-in-python-in-a-matter-of-minutes-using-touchterrain-e0b98b66e754?sk=fc9c9f63f35cfe2c93e78fbb3b132133) (Medium)

## Changes and versioning

See the [changelog](CHANGELOG.md) for the most notable changes between releases.

The project adheres to [semantic versioning](https://semver.org/).

## Instructions for Developers

If you are a developer or planning to contribute, you find all related information in the [Developer Guide](DEV.md).

## How to contribute?

Contributions are welcome in any of the following ways:

- **Bug reports** — if you run into trouble, raise an issue on GitHub.
- **Feedback** — all ideas are welcome; if something feels unclear after reading the user guide, let us know.
- **Feature requests** — tell us what you think is missing.
- **Examples** — if you've done something interesting with the library, get in touch and we will happily include it in the documentation.
- **Sharing** — if you like the library, share it with friends or colleagues.

In all cases, read the [contributing guidelines](CONTRIBUTING.md) first.

## Acknowledgements

Although `sigmaepsilon.mesh` works without `VTK` or `PyVista` installed, it is deeply influenced by these libraries and works best with them around. Shout-out to the developers of `NumPy`, `SciPy`, `Numba`, `Awkward`, `meshio`, and all other third-party libraries involved in the project. Without these libraries, writing performant yet elegant Python code would be much harder.

**Many packages mentioned here have citable research papers. If you use them through `sigmaepsilon.mesh`, take a moment to check their documentation and cite their work.**

Also, funding for these libraries is partly based on the size of the community they support. If your work strongly relies on them, press the :star: button to show your support.

## License

This package is licensed under the [MIT license](LICENSE).

## Third-party licenses

License information for all dependencies can be found in [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES).
