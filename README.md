# Solar Radiation Pressure Library — GPU & RT-Cores

[English](README.md) · [Русский](README.ru.md)

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus)](https://en.cppreference.com/w/cpp/20)
[![CUDA 12.4+](https://img.shields.io/badge/CUDA-12.4%2B-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![OptiX 9.1](https://img.shields.io/badge/OptiX-9.1-76B900?logo=nvidia)](https://developer.nvidia.com/rtx/ray-tracing/optix)
[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6?logo=windows)](https://www.microsoft.com)
[![Linux](https://img.shields.io/badge/Linux-Ubuntu%2020.04%2B-E95420?logo=ubuntu)](https://ubuntu.com)
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
#include "SRPLibrary.h"
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

## Build & integrate

Four paths — choose the one that matches your situation.

---

### Option A — FetchContent (recommended, no pre-built binaries needed)

The simplest way to use the library in a CMake project. CMake downloads and
builds it automatically. No vcpkg, no prebuilt `.a` / `.lib` required — just
HDF5 on your system.

**Linux (Ubuntu / Debian):**
```bash
sudo apt install libhdf5-dev python3-dev   # only HDF5 and Python needed
```

**Your `CMakeLists.txt`:**
```cmake
cmake_minimum_required(VERSION 3.23)
project(my_project LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(SRP
    GIT_REPOSITORY https://github.com/paulzap/Real-Time-Solar-Radiation-Pressure-Modeling.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE)
# GPU methods are auto-disabled when CUDA / OptiX are not installed.
# Force CPU-only explicitly if you prefer:
# set(SM3D_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(SRP)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE srp)
```

**Configure and build:**
```bash
cmake -B build
cmake --build build
```

No extra flags needed. HighFive is fetched automatically if not found locally.
On a machine with CUDA + OptiX, all 6 methods are compiled in automatically.

---

### Option B — one-command script (builds prebuilt `dist/` folder)

The scripts check and install all dependencies automatically, then build
`srp.lib` / `libsrp.a` and copy it to `dist/` together with a
`SRPLibraryConfig.cmake` that handles all transitive dependencies.

**Windows (PowerShell):**
```powershell
.\setup.ps1 -Mode Library
```

**Linux / macOS (bash):**
```bash
chmod +x setup.sh && ./setup.sh
```

**Your `CMakeLists.txt`:**
```cmake
find_package(SRPLibrary REQUIRED CONFIG
    HINTS "/path/to/dist")
target_link_libraries(my_app PRIVATE SRPLibrary::srp)
```

Copy `dist/python/visualize3d.py` next to your executable for 3-D visualization.

---

### Option C — Visual Studio development setup (Windows)

```powershell
.\setup.ps1          # generates SM2D\LocalPaths.props
```

1. Open `SM2D.sln` in Visual Studio 2022
2. Select **Release | x64** ← **required**
3. **Build → Rebuild Solution**

> **Do NOT use Debug configuration for GPU kernels.** Debug CUDA builds exceed
> SM register limits: `CUDA error 701`.

---

### Option D — CMake directly (any platform)

```bash
# CPU-only, system HDF5
cmake -B build -DSM3D_ENABLE_CUDA=OFF -DSM3D_ENABLE_OPTIX=OFF
cmake --build build

# With vcpkg (GPU build)
cmake -B build \
      -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

See [docs/USAGE.md](docs/USAGE.md) for a complete integration guide.

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

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for a full list of known issues
(CUDA version mismatch, missing vcpkg packages, Debug vs Release, PYTHONHOME, PTX).

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

If you use this code in your research, please cite the accompanying paper.
Citation details will be added upon publication.
