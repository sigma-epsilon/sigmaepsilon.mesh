# Changelog

All notable changes to this project will be documented in this file. If you are interested in bug fixes, enhancements etc., best follow the project on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
