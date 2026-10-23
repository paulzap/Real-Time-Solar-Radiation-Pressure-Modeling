# Changelog

All notable changes to SolaRT (the solar radiation pressure library for GPU RT
cores, formerly "Solar Radiation Pressure Library — GPU & RT-Cores").
Versions follow [Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-10-23

### Fixed

- **Articulation was ignored by the GPU and RTX methods.** `CentroidGPU`,
  `CentroidRTX`, `PixelGridGPU` and `PixelGridRTX` keep the triangle list and its
  BVH in GPU memory between calls and rebuilt that copy only when the triangle
  *count* changed. A rotated or translated component keeps the count, so after
  the first call these methods silently computed with the old pose (CPU methods
  were correct). Example: rotating a solar wing by 45° changed the CPU force by
  40 %, while the centroid GPU/RTX methods returned an unchanged force and the
  pixel-grid GPU/RTX methods changed by only 7.5 %. `SatelliteDataset` now has a
  process-wide `geometryVersion()` and `SRPEngine` re-uploads the geometry
  whenever it changes. The re-upload (BVH build + copy) is paid once per change.
- Loading a different file with the **same triangle count** kept the previous
  file's optical properties and component names. They are now refreshed after
  every file read.
- **CMake build** of the library as a subproject failed at device link
  (relocatable device code merged the per-file `__constant__` BVH arrays beyond
  the 64 KB limit) and missed the CUDA driver library needed by OptiX. The
  library now uses whole-program CUDA compilation and links `CUDA::cuda_driver`.
- Documentation of `g_srp_phi0`: the default is **1.0** (results per unit
  pressure), not 4.56e-6 N/m² as stated before.

### Added

- `SRPEngine::setMomentReference(x, y, z)`, `getMomentReference()`,
  `setMomentReferenceToCenterOfMass()`: total moment about any body-frame point.
  The default is still the body-frame origin.
- `SatelliteDataset::hasMassProperties()`, `totalMass()`, `centerOfMass()`,
  `inertiaTensor()`: mass properties from the optional `/inertia` group written
  by SatForm3D (tensor about the centre of mass, pose as stored in the file).
- `SatelliteDataset::geometryVersion()`.
- Version macros `SOLART_VERSION_*` in `SRPLibrary.h`.

### Unchanged

- All SRP kernels and formulas. For fixed geometry, results match 1.0.0 up to the
  run-to-run differences of GPU summation order (1e-17…1e-13 relative), and call
  times match within the run-to-run spread (about ±1.5 %).

## [1.0.0] — 2026-09-08

Code accompanying the article: P.R. Zapevalin, *Real-Time Solar Radiation
Pressure Modeling on GPU RT-Cores*, Aerospace Science and Technology, 2026,
113714, DOI 10.1016/j.ast.2026.113714.

[1.1.0]: https://github.com/paulzap/Real-Time-Solar-Radiation-Pressure-Modeling/releases/tag/v1.1.0
[1.0.0]: https://github.com/paulzap/Real-Time-Solar-Radiation-Pressure-Modeling/releases/tag/v1.0.0
