# Changelog

All notable changes to SolaRT (the solar radiation pressure library for GPU RT
cores, formerly "Solar Radiation Pressure Library — GPU & RT-Cores").
Versions follow [Semantic Versioning](https://semver.org/).

## [1.2.0] — 2026-10-29

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

### Added — SatForm3D 2.3 files (HDF5 schema 2.2)

- **The pose of the file**: a joint sits at `joint_value` if the file records
  one, otherwise at `joint_default_value` (1.1.0 always used zero);
  `clearJoint()` returns to it.
- **Coupled joints** (`coupling_driver_path`, `coupling_ratio`,
  `coupling_offset`): a driven joint follows its driver,
  q = ratio · q_driver + offset clipped to its limits, as in the generator;
  articulating a driven joint directly throws. In 1.1.0 the driven links stayed
  in place when the driver moved — also in earlier SatForm3D files, e.g. the
  folding wing of `sat_01`…`sat_10` (link1 = −link0, link2 = +link0).
- **Deployment states** (`deploy_states`): `deploymentStateNames()`,
  `setDeploymentState(name)`.
- **Shared meshes** (`/prototypes` with `instance_key`, `instance_material`).
- The additive tables `srp`, `symmetry`, `prototypes` are never taken for the
  geometry root.
- Checked on 100 generator models (all 11 bus shapes, 13 mission regimes,
  42 part types, 128–31 166 triangles): geometry equal to an independent rebuild
  by the generator's rules in the file pose, in every deployment state and for
  random driver values (204 cases, 5e-17 of the model size).

### Added — API

- `SatelliteDataset::setJointValue(name, value)` — the declared joint (type and
  axis from the file, pivot at the origin of the local frame, limits checked),
  applied at once; `jointInfo(name)`, `movableComponents()`, `structureVersion()`;
  struct `JointInfo` (also file value, driver, ratio, offset, deployment states).

### Added — animation

- `FrameRecorder`: writes a sequence of computed frames (pose of every mesh, sun
  direction, labels, force, moment, an optional body-to-view rotation such as
  the attitude) to a folder; `SM2D/animate_srp.py` plays it in real time as a
  self-contained HTML page (WebGL 2, works offline, needs only numpy): the
  geometry is stored once, a frame adds the mesh motions and one bit per
  triangle, gzip-compressed (`satellite_62`, 842 662 triangles, 200 frames:
  4.5 MB, 6 ms per frame change). Force and moment arrows; two looks —
  technical (lit / shadowed colours, axes) and space (star sky, the sun with
  its glow, materials from the component types, the library's shadows); two
  frames — body or the recording's view frame (e.g. inertial). Format in
  `docs/USAGE.md`. For display and visual checking, not for timing.
- `SatelliteDataset::meshMotions()` — every mesh's triangle range and its rigid
  motion from the file pose to the current pose (cheap, no triangle list);
  `SRPEngine::getSunDirection()`.
- Demo section 17: a fixed scenario instead of an orbit — the declared parts
  deploy, the sun goes once round the satellite, solar arrays (revolute joints
  with at least three quarters of a turn) follow it, other joints swing; 200 frames are
  computed with `computeViz()` and recorded. An array is turned square to the
  sun from the main normal of its faces (choosing the angle by the largest lit
  area made the arrays jitter by degrees and the moment saw-tooth from frame to
  frame).

### Fixed

- `CentroidGPU` crashed on models with 513–819 triangles: the BVH went to
  constant memory when it was below 64 KB, but the array there holds 1024 nodes
  (40 KB). Present since 1.0.0; 11 of the 100 SatForm3D 2.3 models hit it.

### Accuracy

- Unarticulated models: identical to 1.1.0 (CPU bit for bit, GPU/RTX within the
  run-to-run spread of 1.1.0 itself).
- Articulated poses, against 1.1.0 rebuilding the geometry for the same pose:
  CPU bit for bit; `PixelGridGPU` within 1e-12; the other methods differ by
  single rays that graze a triangle edge, which is rounded differently in the
  part frame — at most 5e-6 of the force for `CentroidGPU` with reflections
  and 9e-6 for `PixelGridRTX` (one grid cell), far below each method's own
  error. In folded poses (a stowed accordion wing) faces seen almost edge-on
  flip whole strips of grid cells: up to about 50 cells (1e-3 of the force) for
  the pixel-grid methods, against an error of the method itself of 0.2–0.3 %.
  Body-frame centroids and normals are bit-identical to `triangles()`.
- **The dataset of the article** (Zenodo 10.5281/zenodo.20261561, 100 models,
  up to 5.4 million triangles): every file loads; in the pose of the file all
  six methods give the 1.1.0 results — `CentroidCPU` bit for bit on all 100
  models, the GPU/RTX methods within 5e-13 of the force and 4e-12 of the
  moment, the same as two calls of 1.1.0 differ. Call times unchanged
  (`satellite_62`, `CentroidRTX`: 2.2–2.3 ms in both versions).
- SatForm3D 2.3 sets of 100 models with 0.5 m and 0.1 m meshes (up to 512 582
  triangles): all methods run; the force differs from the generator's `/srp`
  table by 0.2–0.7 % (median), by single rays of the one-point methods on
  coarse facets — on the finer mesh the outliers disappear.

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
