# Changelog

All notable changes to this project will be documented in this file. If you are interested in bug fixes, enhancements etc., best follow the project on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.4.1]

### Fixed

- `Gauss_Legendre_Tri_6` returned six identical points on one corner of the triangle. It now returns the correct points and is exact for polynomials up to degree 4 (#168).

## [3.4.0]

### Changed

- `H8_to_TET4` and `H27_to_TET10` now split every hexahedron into 6 tetrahedra instead of 5 (the Kuhn split around the diagonal from local node 0 to local node 6). Code that relies on the old number or order of the tetrahedra must be updated.

### Fixed

- `H8_to_TET4` and `H27_to_TET10` returned non-conforming tetrahedral meshes, which also affected `ribbed_plate(..., tetrahedralize=True)` (#165).

## [3.3.1]

### Fixed

- Nested Numba parallel regions that aborted the process with the `workqueue` threading layer, e.g. in `PolyData.cells_around_cells` and H27 cells (#159).

## [3.2.0]

### Added

- `PolyData.cell_ids`, returning the ids of the cells along the walk.

### Changed

- The build system is now Hatchling, and the project is managed with uv instead of Poetry.
- `PolyData.cell_indices` is deprecated in favour of `PolyData.cell_ids`.

### Removed

- The `has_gid` property and the global index field of `CellData` and `PointData`.

### Fixed

- Cell indices of submeshes that have their own hosting pointcloud (#149).
- A block removed from a `PolyData` instance now gets its own point and cell index managers.

## [3.1.0]

### Added

- Voxelization for triangular meshes as `sigmaepsilon.mesh.voxelize.voxelize_T3_H8`.
- Set operations for cells like `TET4_in_H8` etc. that can be used to calculate intersections for all kinds of meshes. The original reason behind these utilitiy functions is voxelization.

### Changed

- Updated the capabilities of the PyVista plotter with new arguments for opacity, plot style and legend. See the details in the documentation for the `sigmaepsilon.mesh.plotting.pvplot.pvplot` function.

### Fixed

- Fixed the voxelizer `sigmaepsilon.mesh.voxelize.voxelize_TET4_H8`. The new implementation is based on the separating axis theorem and is much more precise.

## [3.0.0]

### Added

- **Voxelization** in the direction H8 -> TET4

### Removed

- **TetGen Dependency and Related Feature**: Removed functionality that was exclusively dependent on TetGen due to licensing incompatibility. TetGen is licensed under the GNU Affero General Public License v3 (AGPL-3.0), which is not compatible with our project's MIT license. This change ensures that our project remains compliant with the MIT license terms.

### Notes

- Users who require the removed functionality can consider using TetGen directly in their own projects while complying with the AGPL-3.0 license. For more information on TetGen and its licensing, please visit the [TetGen project page](http://wias-berlin.de/software/tetgen/).

## [2.0.0] - 2023-09-27

### Added

- Protocols for PointData, CellData, PolyData and PolyCell classes
- Cell interpolators (#7)

### Fixed

- Copy and deepcopy operations (#29).
- Cell class generation (#36).

### Changed

- Class architecture of data structures. Now the geometry is a nested class.

### Removed

- Intermediate, unnecessary geometry classes.
