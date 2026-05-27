# Solar Radiation Pressure Library — GPU & RT-Cores

[English](README.md) · [Русский](README.ru.md)

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus)](https://en.cppreference.com/w/cpp/20)
[![CUDA 12.4+](https://img.shields.io/badge/CUDA-12.4%2B-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![OptiX 9.1](https://img.shields.io/badge/OptiX-9.1-76B900?logo=nvidia)](https://developer.nvidia.com/rtx/ray-tracing/optix)
[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6?logo=windows)](https://www.microsoft.com)
[![Visual Studio](https://img.shields.io/badge/Visual%20Studio-2022-5C2D91?logo=visualstudio)](https://visualstudio.microsoft.com/)

> Accurate, GPU-accelerated solar radiation pressure (SRP) force and moment
> computation on articulated spacecraft geometry — six interchangeable
> algorithms, one header.

---

## Why

Solar radiation pressure is a non-conservative perturbation that, on
high-area-to-mass spacecraft, dominates secular orbit evolution beyond the drag
regime. Computing it correctly on a real CAD mesh requires self-shadowing,
multi-bounce specular reflections, and per-facet optical properties
(absorptance α, specularity μ, emissivity ε). This library does that, fast,
through a one-header C++ interface.

The same triangulation can be evaluated by **six algorithms × two paths**:

| Algorithm family | CPU            | CUDA           | OptiX RTX      |
| ---------------- | -------------- | -------------- | -------------- |
| **Centroid**     | `CentroidCPU`  | `CentroidGPU`  | `CentroidRTX`  |
| **PixelGrid**    | `PixelGridCPU` | `PixelGridGPU` | `PixelGridRTX` |

* **Centroid** — one shadow ray per triangle centroid. Linear cost in triangle
  count, ideal for moderate meshes and large-scale ensemble runs.
* **PixelGrid** — parallel rays on a 2-D grid perpendicular to the sun. Cost
  scales as 1/step² but resolves sub-triangle illumination and grazing facets
  far more accurately.

Each method has two implementations:

* `compute(method)` — fast path using lean *bench* kernels (no bounce tracking).
* `computeViz(method)` — full path with per-triangle `bounce_levels`,
  `incident_dirs`, `origin_pts` for visualization.

---

## Quickstart

```cpp
#include "SRPLibrary.h"

int main() {
    SRPEngine engine("data3d_hdf5_0.5");        // scans folder for .h5 files

    engine.setSunDirection(1.0, 0.0, 0.0);      // sun along +X (body frame)
    engine.setMaxReflections(2);                // 2 specular bounces
    engine.setGridStep(0.05);                   // 5 cm cell (PixelGrid* only)

    engine.dataset().articulate("SolarPanel_L", JointConfig::rotY(M_PI / 4));

    SRPResult r = engine.compute(SRPMethod::CentroidRTX);

    std::printf("F = [%.3e, %.3e, %.3e] N\n",
                r.total_force[0], r.total_force[1], r.total_force[2]);
    std::printf("M = [%.3e, %.3e, %.3e] N*m\n",
                r.total_moment[0], r.total_moment[1], r.total_moment[2]);

    // For coloured 3-D visualisation of shadows + reflections:
    engine.computeViz(SRPMethod::CentroidRTX);
    engine.visualizeLastResult();
}
```

`SRPLibrary.h` is the **only** header you need to distribute with the binary.

> **Important:** always build and run in **Release | x64** configuration.
> Debug CUDA builds use significantly more GPU registers and will fail with
> `CUDA error 701: too many resources requested for launch` on complex kernels.

---

## Algorithms at a glance

| Method          | Backend         | Precision               | When to use                                                |
| --------------- | --------------- | ----------------------- | ---------------------------------------------------------- |
| `CentroidCPU`   | CPU             | double                  | No GPU available; reference for unit-tests.                |
| `CentroidGPU`   | CUDA            | double                  | Any NVIDIA GPU; OptiX not installed.                       |
| `CentroidRTX`   | OptiX RT Cores  | float + double atomics  | **Default.** Fastest on Turing/Ampere/Ada.                 |
| `PixelGridCPU`  | CPU             | double                  | High accuracy on small meshes; debugging.                  |
| `PixelGridGPU`  | CUDA            | double                  | Large meshes without OptiX.                                |
| `PixelGridRTX`  | OptiX RT Cores  | float + double atomics  | **Highest accuracy** — RTX hardware + sub-triangle sampling. |

Hardware capabilities at runtime:

```cpp
#include "SM3D_hw_info.cpp"  // declarations exposed through this TU
bool   has_cuda = is_cuda_device_available();
bool   has_rtx  = is_rtx_device_available();
std::string why = rtx_unavailable_reason();   // empty if RTX is usable
print_hardware_info();                        // table to stdout
```

---

## Requirements

| Dependency                    | Version   | Notes                                                          |
| ----------------------------- | --------- | -------------------------------------------------------------- |
| Windows                       | 10/11     | x64                                                            |
| Visual Studio                 | 2022      | Toolset v143, C++20 (`stdcpplatest`)                           |
| NVIDIA CUDA Toolkit           | **12.4+** | `nvcc` on PATH; `setup.ps1` auto-patches the project file      |
| NVIDIA OptiX SDK              | 9.1.0     | RTX methods only; default path `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0` |
| NVIDIA GPU                    | SM ≥ 7.5  | Turing or newer (RTX 20xx / 30xx / 40xx / 50xx)               |
| vcpkg packages                | latest    | `hdf5:x64-windows` `highfive:x64-windows` `pybind11:x64-windows` |
| Python                        | **3.10+** | Only for `visualizeLastResult()`; auto-detected at runtime     |

---

## Dataset

Test geometry (triangulated spacecraft meshes in HDF5 format) is published
separately on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20261561.svg)](https://doi.org/10.5281/zenodo.20261561)

Download and extract `.h5` files into `SM2D/data3d_hdf5_0.5/`
(or pass any folder path to `SRPEngine`).

---

## Build

### Option A — automated setup (recommended)

Run `setup.ps1` from the repository root in **PowerShell**:

```powershell
cd e:\GPU_projects\SM3D_GPU_CC10
.\setup.ps1
```

The script auto-detects CUDA, OptiX, vcpkg, Python; patches the CUDA version
in `SM2D.vcxproj`; generates `SM2D\LocalPaths.props`; installs missing vcpkg
packages (with prompt); and sets `PYTHONHOME` in your user environment.

If any dependency is in a non-default location, pass it explicitly:

```powershell
.\setup.ps1 -OptixRoot "D:\SDK\OptiX 9.1.0" -VcpkgRoot "D:\tools\vcpkg" -PythonRoot "C:\Python312"
```

Use `-Force` to overwrite an existing `LocalPaths.props`.

### Option B — manual

```powershell
# 1. Install vcpkg dependencies
vcpkg install hdf5:x64-windows highfive:x64-windows pybind11:x64-windows

# 2. Copy and fill in the local-paths template
Copy-Item SM2D\LocalPaths.props.template SM2D\LocalPaths.props
# Edit SM2D\LocalPaths.props with your CUDA/OptiX/vcpkg/Python paths

# 3. If your CUDA version is not 12.4, edit SM2D\SM2D.vcxproj manually:
#    search for "CUDA 12.4.props" and "CUDA 12.4.targets" and replace the version number.
#    (VS CUDA plugin requires a literal version string — macros are NOT supported there.)
```

### Build in Visual Studio

1. Open `SM2D.sln` in Visual Studio 2022
2. Select configuration: **Release | x64** ← **required**
3. **Build → Rebuild Solution**

> **Do NOT use Debug configuration for GPU kernels.** Debug CUDA builds allocate
> more registers per thread and exceed the SM resource limit on complex kernels,
> causing `CUDA error 701: too many resources requested for launch`.

The pre-build step automatically compiles the four OptiX shader source files
(`optix_shaders_center.cu`, `optix_shaders_center_bench.cu`,
`optix_shaders_pixel_grid.cu`, `optix_shaders_pixel_grid_bench.cu`) into PTX
modules placed next to the executable.

### Running

```powershell
cd x64\Release
.\SM3D.exe ..\data3d_hdf5_0.5
```

---

## Project layout

```
SM2D/
├── SRPLibrary.h           # single public header — distribute this only
├── SRPLibrary.cpp         # SRPEngine implementation
├── SatelliteDataset.cpp   # HDF5 loader + joint articulation
├── ShadowAlgorithms.cpp   # SRP physics + CSV / Python visualisation helpers
├── ReflectionRayCasting.cpp
├── PixelGridRayCasting.cpp
├── gpu_optix_shared.cu              # OptiX context, GAS, common buffers
├── gpu_optix_center_raytracer*.cu   # Centroid OptiX host-side pipelines
├── gpu_optix_pixel_grid_raytracer*.cu
├── gpu_reflection_raytracer*.cu     # Centroid CUDA kernels (+ bench)
├── gpu_pixel_grid_raytracer*.cu     # PixelGrid CUDA kernels (+ bench)
├── optix_shaders_center*.cu         # PTX shader sources (compiled at build)
├── optix_shaders_pixel_grid*.cu
├── bench_methods.h                  # lean (no bounce-tracking) entry points
├── LocalPaths.props.template        # copy → LocalPaths.props and fill in
├── main.cpp                         # demo / smoke-test programme
└── SM2D.vcxproj                     # VS 2022 project
setup.ps1                            # one-shot dependency setup script
```

---

## Troubleshooting

### `error MSB4019: imported project "CUDA X.Y.props" not found`

Your installed CUDA version does not match the version hardcoded in
`SM2D.vcxproj`. The VS CUDA plugin requires a literal version string in the
import path and does not support MSBuild macros there.

**Fix:** Run `.\setup.ps1` — it patches `SM2D.vcxproj` automatically.
Or edit `SM2D.vcxproj` manually: find `CUDA 12.4.props` / `CUDA 12.4.targets`
and replace `12.4` with your installed version (e.g. `13.1`).

---

### `fatal error C1083: highfive/highfive.hpp: No such file or directory`
### `fatal error C1083: pybind11/embed.h: No such file or directory`

The vcpkg packages `highfive` and/or `pybind11` are not installed.

**Fix:**
```powershell
vcpkg install highfive:x64-windows pybind11:x64-windows
```
Or re-run `.\setup.ps1` — it checks all three required packages and installs
missing ones with your confirmation.

---

### `CUDA error: too many resources requested for launch (error 701)`

You are building or running the **Debug** configuration. Debug CUDA kernels
use significantly more registers per thread and exceed SM resource limits.

**Fix:** Switch to **Release | x64** in Visual Studio and rebuild.

---

### `Failed to import encodings module` / `Visualisation skipped`

The embedded Python interpreter cannot find the standard library. This happens
when `PYTHONHOME` is not set or points to the wrong Python installation.

**Fix:** Run `.\setup.ps1` — it locates Python and sets `PYTHONHOME` in your
user environment. Then **restart Visual Studio** (it inherits the environment
at launch time). Alternatively, set `PYTHONHOME` manually:
```powershell
[System.Environment]::SetEnvironmentVariable("PYTHONHOME",
    "C:\Users\<you>\AppData\Local\Programs\Python\Python312", "User")
```

---

### `CUDA error: PTX JIT compilation failed` / blank OptiX output

The `.ptx` shader files are missing from the executable directory.
Pre-build events compile them automatically; they may have been skipped if
the build was not a full Rebuild.

**Fix:** **Build → Rebuild Solution** (not just Build). Confirm the four
`.ptx` files appear alongside `SM3D.exe`:
```
optix_shaders_center.ptx
optix_shaders_center_bench.ptx
optix_shaders_pixel_grid.ptx
optix_shaders_pixel_grid_bench.ptx
```

---

### `No HDF5 files found in <path>`

The path passed to `SRPEngine(folder)` does not contain `.h5` / `.hdf5` files,
or the working directory is wrong.

**Fix:** Run from `x64\Release\` and pass a relative path:
```powershell
cd x64\Release
.\SM3D.exe ..\..\data3d_hdf5_0.5
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Contact

**Author:** Zapevalin P.R.  
**Email:** pav9981@yandex.ru  
**Issues:** open an issue on this repository.

---

## Citation

If you use this code in your research, please cite:

> P.R. Zapevalin, "Comparative analysis of spacecraft self-shadowing algorithms",
> *Advances in Space Research*, 2026. *(in press)*

```bibtex
@article{zapevalin2026srp,
  title   = {Comparative analysis of spacecraft self-shadowing algorithms},
  author  = {Zapevalin, P.R.},
  journal = {Advances in Space Research},
  year    = {2026},
  note    = {in press}
}
```
