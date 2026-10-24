# Changelog

All notable changes to SolaRT (the solar radiation pressure library for GPU RT
cores, formerly "Solar Radiation Pressure Library — GPU & RT-Cores").
Versions follow [Semantic Versioning](https://semver.org/).

## [1.2.0] — unreleased

### Changed — articulation no longer rebuilds the geometry

- **A pose change costs well under a millisecond on every GPU/RTX method.**
  A movable component that is out of the pose stored in the file is kept as a
  separate rigid part with its own acceleration structure in its own frame
  (CUDA: one BVH per part and a part table; RTX: one GAS per part under an IAS);
  a pose change only updates the part transforms and the body-frame centroids
  and normals of the moved parts. Example (`satellite_62`, 842 662 triangles,
  both solar arrays sweeping, RTX 3060 Ti): `articulate()` of the two arrays
  54 ms → 3–10 µs; pose change on the GPU 683 / 47 / 615 / 55 ms → 0.53–0.59 ms
  (`CentroidGPU` / `CentroidRTX` / `PixelGridGPU` / `PixelGridRTX`).
- A component leaving or returning to its file pose changes the split and costs
  one rebuild (0.14–0.59 s, the price of every rotation in 1.1.0). **A model in
  the pose of the file is computed exactly as in 1.1.0** (same code, same
  results, same speed).
- Labels, bounce levels and the other visualisation data keep the triangle
  numbering of `triangles()`.
- `articulate()` / `reload()` only record the joint values; `triangles()` is
  rebuilt on its next call (bit-identical to the eager rebuild of 1.1.0). A
  reference to the triangle list taken before an articulation shows the old
  pose until `triangles()` is called again.

### Changed — the file decides what moves

- `SatelliteDataset` reads the SatForm3D joint attributes `joint_type`,
  `joint_axis`, `joint_min`, `joint_max`, `joint_default_value`. Only components
  declared `revolute` or `prismatic` can be articulated; a fixed component, the
  root, a joint of the other type, a value outside `[joint_min, joint_max]` or
  an unknown name throws `std::invalid_argument` (1.1.0 accepted any component).
  An axis not parallel to the declared one is used as given, with a warning.

### Added

- `SatelliteDataset::setJointValue(name, value)` — the declared joint (type and
  axis from the file, pivot at the origin of the local frame, limits checked),
  applied at once; `jointInfo(name)`, `movableComponents()`, `structureVersion()`;
  struct `JointInfo`.

### Accuracy

- Unarticulated models: identical to 1.1.0 (CPU bit for bit, GPU/RTX within the
  run-to-run spread of 1.1.0 itself).
- Articulated poses, against 1.1.0 rebuilding the geometry for the same pose:
  CPU bit for bit; `PixelGridGPU` within 1e-12; the other methods differ by
  single rays that graze a triangle edge, which is rounded differently in the
  part frame — at most 5e-6 of the force for `CentroidGPU` with reflections
  and 9e-6 for `PixelGridRTX` (one grid cell), far below each method's own
  error. Body-frame centroids and normals are bit-identical to `triangles()`.

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
