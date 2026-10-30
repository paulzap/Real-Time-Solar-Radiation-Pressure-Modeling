# Changelog

All notable changes to SolaRT (the solar radiation pressure library for GPU RT
cores, formerly "Solar Radiation Pressure Library — GPU & RT-Cores").
Versions follow [Semantic Versioning](https://semver.org/).

## [1.2.1] — 2026-10-30

This release corrects defects found by checking the library against exact answers.
The results of the article are not put in question: its experiments used random Sun
directions, and the changes below move them at most in the fifth digit of the force for
`CentroidGPU` and `CentroidRTX` and the fourth for `CentroidCPU`. Version 1.0.0 stays the code
of the article.

**Numbers on the 100 models of the article's dataset** (5 random Sun directions each, 0 and 3
reflections, against 1.2.0): `PixelGridGPU` and `PixelGridRTX` agree within run-to-run noise.
`CentroidRTX` moves by at most 9e-6 of the force and 1.9e-4 of the moment, `CentroidGPU` by at
most 6.6e-5 and 4.6e-3, `CentroidCPU` by at most 7.2e-4 and 6.9e-3.

**On the newer SatForm3D models with few large triangles** the changes are larger: half of the
models move by more than 0.1 %, a few by up to 15 % with `CentroidGPU` and 20 % with
`CentroidCPU`. Re-run results computed there. The three centroid methods now agree, but they
are no more accurate than before: one point per triangle is a coarse estimate on such models,
and a pixel-grid method should be used there.

### Fixed — wrong numbers

- **Pixel-grid methods lost rays with the Sun along a model axis.** A ray lying in the
  plane of a bounding-box face was rejected (a zero direction component with a negative
  sign gave NaN and -inf), so rays passed through the model along the seams of the BVH:
  -2.1 % on `satellite_3` with the Sun along +Y, the default direction, -29 % on a smaller
  model, -52 % on a cube with an exact answer. `PixelGridCPU` and `PixelGridGPU` affected;
  `PixelGridRTX` was right. Rays through the edge shared by two triangles are no longer lost
  to rounding either, here and in `CentroidCPU`.
- **`compute()` on GPU/RTX kept the old geometry when the whole model hung on one joint**
  (only the first rotation counted, up to +91 % and more, without a warning). Regression of
  1.2.0, no published or SatForm3D model is built this way.
- **A black surface was computed as a half-reflecting one.** A triangle with reflectance 0
  and specularity 0 got 0.5 / 0.5 in the centroid methods, a substitution left from the CSV
  input that no longer exists (a closed black cube: 1.42 against the exact 1.00). The optics
  are now used as given in all six methods, and pixel-grid methods accept a fully black model
  with reflections. No published dataset has such triangles.
- **`CentroidCPU` and `CentroidGPU` missed shadows from a triangle whose centre lies behind
  the point**, because of a filter on the centre that `CentroidRTX` does not have. Together with
  the fix of `CentroidRTX` below, the three now agree: on the newer models from up to 10.6 % of
  the force down to 6e-5.
- **`CentroidCPU` reflections now match `CentroidGPU` and `CentroidRTX`:** moment arm and next
  ray start at the actual hit point rather than the centre of the triangle (up to 2.7 % of
  the force on models with large triangles), the same surface offset, and a ray on a shared
  edge no longer gives its force to both triangles.
- **`g_srp_phi0` is now a pure factor.** On the CPU the chain of reflections was cut at an
  absolute power, so setting it to 4.56e-6 lost later reflections (+36 % on a small mirror
  chain). `PixelGridCPU` is affected the same way; both are fixed.
- **Small triangles were transparent** to `CentroidCPU` (2 × area below 1e-6 m²), `CentroidGPU`
  and `PixelGridGPU` (below 1e-8 m²). `PixelGridRTX` missed the top surface with a grid step
  below 1e-4 m (force 0), and `CentroidRTX` missed an occluder closer than 2e-4 m.
- `CentroidCPU` lost the reflected light of a file without `areas`.
- **`compute()` and `computeViz()` of `CentroidGPU` gave different numbers** (up to 0.97 %):
  the shadow and reflected rays of `computeViz()` had no offset from the surface.

### Fixed — crashes and building

- A clean CMake build failed with `atomicAdd` errors: the guard that chooses the CUDA
  architectures could never fire, and the code was built for sm_52. The default is now really
  75, 86 and 89, and an architecture below 60 stops with a clear message. `setup.ps1` passes
  the architecture it detects.
- `find_package(SRPLibrary)` failed on Windows because of backslashes in the generated
  `SRPLibraryConfig.cmake`, and could not link the CUDA libraries. The generated configs no
  longer call `cmake_minimum_required()`, which reset the policies of the user's project.
- A refused file or joint in `load()` or `load_single_file()` left the dataset inconsistent, and
  the next `reload()` crashed or computed a cut model. A refused load now changes nothing.
- `CentroidGPU` ended the process on a CUDA error; it now throws `std::runtime_error` like the
  other methods.
- A short `normals` array or a wrong vertex index in a file crashed; such a file, or one with a
  short `reflectance` or `specularity` array, is now refused at load time.
- `.ptx` files are also looked for next to the executable, as the documentation said.

### Changed

- The citation section of the README moved up, right under the introduction, and the quickstart
  uses names that exist in the dataset.
- The default `SURFACE_OFFSET` of `CentroidCPU` is 1e-4 m, as on GPU and RTX (was 2 mm).

### Added

- Every source file starts with the authorship line, the SPDX licence identifier and the paper
  to cite. `SOLART_CITATION` in `SRPLibrary.h` gives the reference as a string, and the demo
  programs print it when they start; the library itself prints nothing.

### Documentation

- α is the reflectance, not the absorptance; forces are in m² unless `g_srp_phi0` is set;
  `SRP_PYTHONHOME` is the variable that chooses Python; versions, the demo menu and the
  declarations of helper functions match the code; `emissivity` takes no part in the force;
  GPU and RTX results have a numerical floor of about 1e-7; pixel-grid labels are not a map of
  illumination; the grid error with the Sun exactly along an axis is described.

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

[1.2.1]: https://github.com/paulzap/Real-Time-Solar-Radiation-Pressure-Modeling/releases/tag/v1.2.1
[1.2.0]: https://github.com/paulzap/Real-Time-Solar-Radiation-Pressure-Modeling/releases/tag/v1.2.0
[1.1.0]: https://github.com/paulzap/Real-Time-Solar-Radiation-Pressure-Modeling/releases/tag/v1.1.0
[1.0.0]: https://github.com/paulzap/Real-Time-Solar-Radiation-Pressure-Modeling/releases/tag/v1.0.0
