# **SigmaEpsilon.Mesh** - Data Structures, Computation and Visualization for Complex Polygonal Meshes in Python

![ ](logo.png)

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/sigma-epsilon/sigmaepsilon.mesh/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/sigma-epsilon/sigmaepsilon.mesh/tree/main)
[![codecov](https://codecov.io/gh/sigma-epsilon/sigmaepsilon.mesh/graph/badge.svg?token=7JKJ3HHSX3)](https://codecov.io/gh/sigma-epsilon/sigmaepsilon.mesh)
[![Documentation Status](https://readthedocs.org/projects/sigmaepsilonmesh/badge/?version=latest)](https://sigmaepsilonmesh.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/sigmaepsilon.mesh.svg)](https://pypi.org/project/sigmaepsilon.mesh)
[![Python 3.7‒3.10](https://img.shields.io/badge/python-3.7%E2%80%923.10-blue)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Requirements Status](https://dependency-dash.repo-helper.uk/github/sigma-epsilon/sigmaepsilon.mesh/badge.svg)](https://dependency-dash.repo-helper.uk/github/sigma-epsilon/sigmaepsilon.mesh)

The [sigmaepsilon.mesh](https://sigmaepsilon.mesh.readthedocs.io/en/latest/) library aims to provide the tools to build and analyse poligonal meshes with complex topologies. Meshes can be built like a dictionary, using arbitarily nested layouts and then be translated to other formats including [VTK](https://vtk.org/) and [PyVista](https://docs.pyvista.org/). For plotting, there is also support for [K3D](http://k3d-jupyter.org/), [Matplotlib](https://matplotlib.org/) and [Plotly](https://plotly.com/python/).

The data model is built around [Awkward](https://awkward-array.org/doc/main/), which makes it possible to attach nested, variable-sized data to the points or the cells in a mesh, also providing interfaces to other popular libraries like [Pandas](https://vtk.org/) or [PyArrow](https://arrow.apache.org/docs/python/index.html). Implementations are fast as they rely on the vector math capabilities of [NumPy](https://numpy.org/doc/stable/index.html), while other computationally sensitive calculations are JIT-compiled using [Numba](https://numba.pydata.org/).

Here and there we also use [NetworkX](https://networkx.org/documentation/stable/index.html#), [SciPy](https://scipy.org/), [SymPy](https://www.sympy.org/en/index.html) and [scikit-learn](https://scikit-learn.org/stable/).

> **Note**
> Implementation of the performance critical parts of the library rely on the JIT-compilation capabilities of Numba. This means that the library performs well even for large scale problems, on the expense of a longer first call.

## Highlights

* Classes to handle points, pointclouds, reference frames and jagged topologies.
* Array-like mesh composition with a Numba-jittable database model. Join or split meshes, attach numerical data and save to and load from disk.
* Simplified and preconfigured plotting facility using PyVista.
* Grid generation in 1, 2 and 3 dimensions for arbitrarily structured Lagrangian cells.
* A mechanism for all sorts of geometrical and topological transformations.
* A customizable nodal distribution mechanism to effortlessly pass around data between points and cells.
* Generation of *Pseudo Peripheral Nodes*, *Rooted Level Structures* and *Adjancency Matrices* for arbitrary polygonal meshes.
* Symbolic shape function generation for arbitrarily structured Lagrangian cells in 1, 2 and 3 dimensions with an extendible interpolation and extrapolation mechanism.
* Connections to popular third party libraries like `networkx`, `pandas`, `vtk`, `PyVista` and more.
* The ability to read from a wide range of formats thanks to the combined power of `vtk`, `PyVista` and `meshio`.

## Projects using sigmaepsilon.mesh

* [SigmaEpsilon.Solid](https://github.com/sigma-epsilon/sigmaepsilon.solid) - A Python library for computational solid mechanics.
* [PyAxisVM](https://github.com/AxisVM/pyaxisvm) - The official Python package of [AxisVM](https://axisvm.eu/), a popular structural analysis and design software.

## Documentation

The [documentation](https://sigmaepsilonmesh.readthedocs.io/en/latest/) is built with [Sphinx](https://www.sphinx-doc.org/en/master/) using the [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html) and hosted on [ReadTheDocs](https://readthedocs.org/). Check it out for the user guide, an ever growing set of examples, and API Reference.

## Installation

sigmaepsilon.mesh can be installed from PyPI using `pip` on Python >= 3.7:

```console
>>> pip install sigmaepsilon.mesh
```

or chechkout with the following command using GitHub CLI

```console
gh repo clone sigma-epsilon/sigmaepsilon.mesh
```

and install from source by typing

```console
>>> pip install .
```

If you want to run the tests, you can install the package along with the necessary optional dependencies like this

```console
>>> pip install ".[test]"
```

### Development mode

If you are a developer and want to install the library in development mode, the suggested way is by using this command:

```console
>>> pip install "-e .[test, dev]"
```

### Checking your installation

You should be able to import sigmaepsilon.mesh from the Python prompt:

```console
$ python
Python 3.8.10 (tags/v3.8.10:3d8993a, May  3 2021, 11:48:03) [MSC v.1928 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import sigmaepsilon.mesh
>>> sigmaepsilon.mesh.__version__
'2.2.0'
```

## Testing and coverage

The following command runs all tests and creates a html report in a folder named `htmlcov` (the settings are governed by the `.coveragerc` file):

```console
python -m pytest --cov-report html --cov-config=.coveragerc --cov sigmaepsilon.mesh
```

Open `htmlcov/index.html` to see the results.

## Changes and versioning

See the [changelog](CHANGELOG.md), for the most notable changes between releases.

The project adheres to [semantic versioning](https://semver.org/).

## How to contribute?

Contributions are currently expected in any the following ways:

* finding bugs
  If you run into trouble when using the library and you think it is a bug, feel free to raise an issue.
* feedback
  All kinds of ideas are welcome. For instance if you feel like something is still shady (after reading the user guide), we want to know. Be gentle though, the development of the library is financially not supported yet.
* feature requests
  Tell us what you think is missing (with realistic expectations).
* examples
  If you've done something with the library and you think that it would make for a good example, get in touch with the developers and we will happily inlude it in the documention.
* sharing is caring
  If you like the library, share it with your friends or colleagues so they can like it too.

In all cases, read the [contributing guidelines](CONTRIBUTING.md) before you do anything.

## Acknowledgements

Although `sigmaepsilon.mesh` works without `VTK` or `PyVista` being installed, it is highly influenced by these libraries and works best with them around. Also shout-out for the developers of `NumPy`, `Scipy`, `Numba`, `Awkward`, `meshio` and all the third-party libraries involved in the project. Whithout these libraries the concept of writing performant, yet elegant Python code would be much more difficult.

**A lot of the packages mentioned on this document here and the introduction have a citable research paper. If you use them in your work through sigmaepsilon.mesh, take a moment to check out their documentations and cite their papers.**

Also, funding of these libraries is partly based on the size of the community they are able to support. If what you are doing strongly relies on these libraries, don't forget to press the :star: button to show your support.

## License

This package is licensed under the [MIT license](LICENSE.txt).
