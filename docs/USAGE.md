# Using the SRP Library

This guide covers building the library, integrating it into your project, and using the full API.

**Platforms:** Windows 10/11 · Linux (Ubuntu 20.04+) · macOS 13+ (CPU-only)  
**Languages:** C++20 required  
**Compilers:** MSVC 2022, GCC 12+, Clang 15+

---

## Table of Contents

1. [Dependencies](#1-dependencies)
2. [Build with CMake](#2-build-with-cmake)
   - [Visual Studio (recommended on Windows)](#visual-studio-open-folder)
   - [VS Code](#vs-code)
   - [Command line](#command-line)
3. [Integrate the library in your project](#3-integrate-in-your-project)
4. [API Reference](#4-api-reference)
   - [SRPEngine](#srpengine)
   - [SatelliteDataset](#satellitedataset)
   - [JointConfig](#jointconfig)
   - [SRPMethod](#srpmethod)
   - [SRPResult](#srpresult)
   - [Triangle](#triangle)
5. [Code Examples](#5-code-examples)
6. [Testing Functions](#6-testing-functions)
7. [Visualization](#7-visualization)

---

## 1. Dependencies

| Dependency | Required for | Where to get |
|---|---|---|
| HDF5 ≥ 1.12 | All builds | [hdfgroup.org](https://www.hdfgroup.org/downloads/hdf5/) or vcpkg |
| HighFive ≥ 2.7 | All builds | vcpkg (`highfive`) |
| CUDA Toolkit ≥ 11.8 | GPU methods | [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads) |
| OptiX SDK ≥ 8.0 | RTX methods | [developer.nvidia.com/optix](https://developer.nvidia.com/optix) |
| pybind11 ≥ 2.11 | Python visualization | vcpkg (`pybind11`) |
| Python ≥ 3.10 | Python visualization | [python.org](https://www.python.org/) |

**Install HDF5 + HighFive + pybind11 via vcpkg (recommended):**

```bash
# Windows
vcpkg install hdf5:x64-windows highfive:x64-windows pybind11:x64-windows

# Linux / macOS
vcpkg install hdf5 highfive pybind11
```

**Or via apt (Linux):**

```bash
sudo apt install libhdf5-dev
pip install pybind11   # or vcpkg
```

---

## 2. Build with CMake

### Quickest path — setup script (recommended)

The setup scripts handle everything: check dependencies, install missing
packages, run CMake, compile, and copy outputs to `dist/`.

**Windows:**
```powershell
.\setup.ps1 -Mode Library
# outputs: dist\include\SRPLibrary.h  dist\lib\srp.lib  dist\bin\hdf5.dll
```

**Linux / macOS:**
```bash
chmod +x setup.sh && ./setup.sh
# outputs: dist/include/SRPLibrary.h  dist/lib/libsrp.a
```

Override non-default paths:
```powershell
.\setup.ps1 -Mode Library -VcpkgRoot D:\tools\vcpkg -OptixRoot "D:\SDK\OptiX 9.1.0"
```
```bash
./setup.sh --vcpkg ~/vcpkg --optix /opt/optix
```

Use `--force` / `-Force` to rebuild from scratch.

After the script: link `dist/lib/srp.lib` (or `libsrp.a`) and include
`dist/include/` in your project — that's it.

---

### Manual CMake build

The `CMakeLists.txt` auto-detects what's available and enables only the supported methods:

| What's installed | Methods available |
|---|---|
| Nothing special (CPU only) | `CentroidCPU`, `PixelGridCPU` |
| CUDA Toolkit | + `CentroidGPU`, `PixelGridGPU` |
| CUDA + OptiX | + `CentroidRTX`, `PixelGridRTX` |

### Visual Studio (Open Folder)

This is the easiest path on Windows — no new `.sln` needed.

1. **File → Open → Folder** — select the repo root (the folder containing `CMakeLists.txt`)
2. VS detects CMake automatically and shows a **CMake Overview** page
3. Select the desired configuration in the toolbar: `x64-Release` (recommended — CUDA Debug has known issues)
4. **Build → Build All** (`Ctrl+Shift+B`)
5. Output: `srp.lib` and `srp_demo.exe` appear in the build directory (shown in the Output window)

> ⚠️ **Always build Release, not Debug.**  
> CUDA kernels in Debug mode can fail with error 701 (device-side assertion).  
> In VS: select `x64-Release` in the configuration dropdown.

If OptiX is installed in a non-default location, set the environment variable before opening VS:

```
OPTIX_INSTALL_DIR=C:\path\to\OptiX SDK 9.1.0
```

Or set it in CMake settings (Project → CMake Settings → CMake variables):
```
OPTIX_INSTALL_DIR=C:\path\to\OptiX SDK 9.1.0
```

### VS Code

1. Install extensions: **C/C++**, **CMake Tools**
2. Open the repo folder: `code e:\GPU_projects\SM3D_GPU_CC10`
3. CMake Tools auto-detects `CMakeLists.txt` — click **Configure** when prompted
4. Select kit: `Visual Studio Community 2022 Release - x86_amd64` (Windows) or `GCC 12` (Linux)
5. Set build type to **Release** in the status bar
6. Click **Build** (or `Ctrl+Shift+P` → "CMake: Build")

### Command line

```powershell
# Windows — CPU-only (no GPU needed)
cmake -B build_cpu -DSM3D_ENABLE_CUDA=OFF -DSM3D_ENABLE_OPTIX=OFF `
      -DCMAKE_TOOLCHAIN_FILE=C:/Users/you/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build_cpu --config Release

# Windows — full build (CUDA + OptiX)
cmake -B build_full `
      -DCMAKE_TOOLCHAIN_FILE=C:/Users/you/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build_full --config Release

# Linux — CPU-only
cmake -B build_cpu -DSM3D_ENABLE_CUDA=OFF \
      -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build_cpu

# Linux — full build
cmake -B build_full \
      -DOPTIX_INSTALL_DIR=/opt/optix \
      -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build_full
```

**Build outputs:**

| File | Description |
|---|---|
| `srp.lib` / `libsrp.a` | Static library to link into your project |
| `srp_demo` / `srp_demo.exe` | Demo executable |
| `*.ptx` | OptiX shader files — must be **in the same directory as your exe** at runtime |

---

## 3. Integrate in Your Project

### Option A — CMake `add_subdirectory` (simplest)

If your project also uses CMake, add the SRP library as a subdirectory:

```cmake
# In your CMakeLists.txt:
add_subdirectory(path/to/SM3D_GPU_CC10)   # adds the srp target

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE srp)
# No need to set include dirs — srp exports them via target_include_directories PUBLIC
```

### Option B — Link the prebuilt .lib in Visual Studio

1. Build the library first (see §2 above) to produce `srp.lib`
2. In your VS project:
   - **Project → Properties → C/C++ → Additional Include Directories** → add path to `SM2D/` folder
   - **Linker → Input → Additional Dependencies** → add `srp.lib`
   - **Linker → General → Additional Library Directories** → add path to `build_full/Release/`
3. Copy `*.ptx` files next to your `.exe` (only needed for OptiX/RTX methods)
4. Copy `hdf5.dll` / `hdf5_D.dll` next to your `.exe` (from vcpkg `bin/`)

### Option C — Link the prebuilt .lib in VS Code (CMake)

```cmake
find_library(SRP_LIB srp PATHS path/to/build/Release)
target_link_libraries(my_app PRIVATE ${SRP_LIB})
target_include_directories(my_app PRIVATE path/to/SM3D_GPU_CC10/SM2D)
```

### Required defines

When linking a prebuilt `.lib`, you must replicate the defines used during compilation:

```cmake
# If the .lib was built with CUDA support:
target_compile_definitions(my_app PRIVATE SM3D_HAS_CUDA)

# If also built with OptiX support:
target_compile_definitions(my_app PRIVATE SM3D_HAS_OPTIX)

# If also built with Python visualization:
target_compile_definitions(my_app PRIVATE SM3D_HAS_PYTHON)
```

---

## 4. API Reference

Include the single public header:

```cpp
#include "SRPLibrary.h"
```

---

### SRPEngine

The main entry point. Owns a `SatelliteDataset` and runs SRP calculations.

```cpp
// Construction
SRPEngine engine("path/to/h5_files");   // scans folder, loads file[0]
SRPEngine engine("satellite.h5");        // single file also works

// Sun direction (normalised automatically)
engine.setSunDirection(1.0, 0.0, 0.0);  // Sun along +X axis

// Reflections (inter-surface bounces, default 0)
engine.setMaxReflections(2);

// Pixel-grid cell size in metres (PixelGrid methods only, default 0.05)
engine.setGridStep(0.02);
double step = engine.getGridStep();

// Run calculation
SRPResult r = engine.compute(SRPMethod::CentroidRTX);   // fast (no bounce tracking)
SRPResult r = engine.computeViz(SRPMethod::CentroidRTX); // slower (+ bounce tracking)

// Access results
const std::vector<int>& labels = engine.getLabels();     // 1=illuminated, 0=shadowed

// After computeViz() only:
const std::vector<int>&                  bounces = engine.getBounceLevels(); // -1/0/1/2...
const std::vector<std::array<double,3>>& dirs    = engine.getIncidentDirs();
const std::vector<std::array<double,3>>& origins = engine.getOriginPts();

// Python visualization (requires SM3D_HAS_PYTHON build)
engine.visualizeLastResult();              // all triangles
engine.visualizeLastResult(3, true);       // every 3rd triangle, show normals

// Dataset access
SatelliteDataset& ds = engine.dataset();
```

---

### SatelliteDataset

Loads and transforms satellite geometry from HDF5 files.

```cpp
SatelliteDataset& ds = engine.dataset();

// File management
int  n     = ds.fileCount();
auto names = ds.fileNames();
auto cur   = ds.currentFile();
ds.load(0);              // load by index
ds.load("path/to.h5");   // load by path
ds.reload();             // re-apply current joints without re-reading HDF5

// Geometry access
const std::vector<Triangle>& tris = ds.triangles();
int ntri = ds.triangleCount();

// Component inspection
auto comp_names = ds.componentNames();       // {"Body", "SolarPanel_L", ...}
auto comp_infos = ds.componentInfos();       // name + type + triangle count + area
bool exists     = ds.hasComponent("S1_Mirror");
auto s1_tris    = ds.componentTriangles("S1_Mirror");  // vector<const Triangle*>

ds.printInfo();        // total triangles, area, bounding box
ds.printComponents();  // table of all components
ds.printJoints();      // currently active joints

// Articulation — revolute (rotation)
ds.articulate("SolarPanel_L", JointConfig::rotY(M_PI / 4));  // setJoint + reload
ds.setJoint("SolarPanel_L",   JointConfig::rotY(M_PI / 4));  // stage only (reload manually)
ds.reload();                                                   // commit staged joints

// Articulation — prismatic (translation)
ds.articulate("Boom", JointConfig::translateZ(2.5));

// Joint management
ds.clearJoint("SolarPanel_L");
ds.clearAllJoints();
bool has = ds.hasJoint("SolarPanel_L");
JointConfig jc = ds.getJoint("SolarPanel_L");
const auto& all = ds.joints();   // std::map<std::string, JointConfig>
```

---

### JointConfig

Describes a single revolute or prismatic articulation.

```cpp
// Revolute — rotation around a standard axis
JointConfig::rotZ(angle_rad)                    // rotate around Z
JointConfig::rotX(angle_rad)                    // rotate around X
JointConfig::rotY(angle_rad)                    // rotate around Y
JointConfig::rot(ax, ay, az, angle_rad)         // rotate around arbitrary axis

// Prismatic — translation along an axis
JointConfig::translateZ(dist_m)
JointConfig::translateX(dist_m)
JointConfig::translateY(dist_m)
JointConfig::translate(ax, ay, az, dist_m)

// Pivot control (append to any revolute factory)
JointConfig::rotZ(angle).atOrigin()             // pivot at (0, 0, 0)  — use for hinge joints
JointConfig::rotZ(angle).atPivot(x, y, z)       // explicit pivot in local frame
// default: pivot = auto centroid of the component mesh

// Examples
ds.articulate("Panel_L", JointConfig::rotY(M_PI / 2));
ds.articulate("Panel_L", JointConfig::rotY(M_PI / 2).atOrigin());  // hinge at origin
ds.articulate("Boom",    JointConfig::translateZ(3.0));
ds.articulate("Mirror",  JointConfig::rot(0, 1, 0, 0.3).atPivot(0.5, 0, 0));
```

---

### SRPMethod

```cpp
enum class SRPMethod {
    CentroidCPU,   // CPU,  one ray per triangle centroid
    CentroidGPU,   // CUDA, one ray per triangle centroid
    CentroidRTX,   // OptiX RT Cores, one ray per triangle centroid  [fastest on RTX GPU]
    PixelGridCPU,  // CPU,  parallel rays on 2-D grid (controlled by setGridStep)
    PixelGridGPU,  // CUDA, parallel rays on 2-D grid
    PixelGridRTX,  // OptiX RT Cores, parallel rays on 2-D grid     [most accurate + fast]
};
```

**Choosing a method:**

| Situation | Recommended |
|---|---|
| No GPU | `CentroidCPU` or `PixelGridCPU` |
| NVIDIA GPU, no OptiX | `CentroidGPU` or `PixelGridGPU` |
| RTX GPU (Turing/Ampere/Ada) | `CentroidRTX` (speed) or `PixelGridRTX` (accuracy) |
| Very small / grazing facets | `PixelGrid*` (centroid misses thin geometry) |
| Multi-bounce reflections | Any method — pass `setMaxReflections(n)` |

Calling an unavailable method throws `std::runtime_error` with a descriptive message.  
Use `is_cuda_device_available()` / `is_rtx_device_available()` to check at runtime (declared in the demo's `main.cpp`).

---

### SRPResult

```cpp
struct SRPResult {
    std::vector<int>       labels;        // per-triangle: 1 = illuminated, 0 = shadowed
    std::array<double, 3>  total_force;   // total SRP force  [Fx, Fy, Fz]  (N)
    std::array<double, 3>  total_moment;  // total SRP moment [Mx, My, Mz]  (N·m)
};

// Reading results
SRPResult r = engine.compute(SRPMethod::CentroidGPU);
std::cout << "Force:  " << r.total_force[0]  << " " << r.total_force[1]  << " " << r.total_force[2]  << " N\n";
std::cout << "Moment: " << r.total_moment[0] << " " << r.total_moment[1] << " " << r.total_moment[2] << " N·m\n";

int illuminated = std::count(r.labels.begin(), r.labels.end(), 1);
std::cout << illuminated << " / " << r.labels.size() << " triangles illuminated\n";
```

---

### Triangle

Read-only from `dataset().triangles()`. All coordinates are in body frame (metres).

```cpp
struct Triangle {
    std::string  ID;                                // unique identifier
    double       v1_x, v1_y, v1_z;                 // vertex 1
    double       v2_x, v2_y, v2_z;                 // vertex 2
    double       v3_x, v3_y, v3_z;                 // vertex 3
    double       normal_x, normal_y, normal_z;     // outward unit normal
    std::string  component_type;                    // HDF5 component_type attribute
    std::string  component_name;                    // HDF5 group name
    double       label;                             // 1 = illuminated, 0 = shadowed
    double       reflectance;   // α ∈ [0,1] — fraction of flux reflected
    double       specularity;   // μ ∈ [0,1] — of reflected, fraction specular
    double       emissivity;    // ε ∈ [0,1] — thermal emission efficiency
    double       area;          // m²
    double       centroid_x, centroid_y, centroid_z;
};
```

---

### Global scale factor

```cpp
extern double g_srp_phi0;   // default: 4.56e-6 N/m² at 1 AU

// Override before any compute() call:
g_srp_phi0 = 4.56e-6;   // SI (default)
g_srp_phi0 = 1.0;        // dimensionless / normalised output
```

---

## 5. Code Examples

### Minimal quickstart

```cpp
#include "SRPLibrary.h"
#include <iostream>

int main() {
    SRPEngine engine("data/h5_files");
    engine.setSunDirection(1, 0, 0);

    SRPResult r = engine.compute(SRPMethod::CentroidCPU);

    std::cout << "Force: "
              << r.total_force[0] << "  "
              << r.total_force[1] << "  "
              << r.total_force[2] << " N\n";
}
```

### Multi-bounce reflections

```cpp
engine.setSunDirection(0, 1, 0);
engine.setMaxReflections(2);

SRPResult r = engine.compute(SRPMethod::CentroidGPU);
```

### Articulate solar panels and compute

```cpp
engine.dataset().articulate("SolarPanel_L", JointConfig::rotY( M_PI / 4));
engine.dataset().articulate("SolarPanel_R", JointConfig::rotY(-M_PI / 4));
engine.setSunDirection(0, 0, 1);

SRPResult r = engine.compute(SRPMethod::PixelGridRTX);
engine.visualizeLastResult();
```

### Sweep sun direction

```cpp
#include <cmath>
#include <vector>

std::vector<SRPResult> results;
const int N = 36;
for (int i = 0; i < N; ++i) {
    double angle = i * 2.0 * M_PI / N;
    engine.setSunDirection(std::cos(angle), std::sin(angle), 0.0);
    results.push_back(engine.compute(SRPMethod::CentroidRTX));
}
```

### Sweep solar panel angle

```cpp
for (int deg = 0; deg <= 90; deg += 5) {
    double rad = deg * M_PI / 180.0;
    engine.dataset().articulate("SolarPanel_L", JointConfig::rotY(rad));
    SRPResult r = engine.compute(SRPMethod::CentroidGPU);
    std::cout << deg << "°  Fz=" << r.total_force[2] << " N\n";
}
```

### Compare methods on the same geometry

```cpp
engine.setSunDirection(1, 0, 0);
engine.setGridStep(0.02);

const SRPMethod methods[] = {
    SRPMethod::CentroidCPU,
    SRPMethod::CentroidGPU,
    SRPMethod::CentroidRTX,
    SRPMethod::PixelGridCPU,
    SRPMethod::PixelGridGPU,
    SRPMethod::PixelGridRTX,
};
const char* names[] = {
    "CentroidCPU", "CentroidGPU", "CentroidRTX",
    "PixelGridCPU","PixelGridGPU","PixelGridRTX",
};

for (int i = 0; i < 6; ++i) {
    try {
        SRPResult r = engine.compute(methods[i]);
        std::cout << names[i] << "  Fx=" << r.total_force[0] << " N\n";
    } catch (const std::runtime_error& e) {
        std::cout << names[i] << "  unavailable: " << e.what() << "\n";
    }
}
```

### Inspect bounce levels after computeViz

```cpp
SRPResult r = engine.computeViz(SRPMethod::CentroidRTX);
const auto& bounces = engine.getBounceLevels();
const auto& dirs    = engine.getIncidentDirs();

int direct = 0, reflected = 0, shadowed = 0;
for (int b : bounces) {
    if (b == 0)       ++direct;
    else if (b > 0)   ++reflected;
    else              ++shadowed;   // b == -1
}
std::cout << "Direct=" << direct
          << " Reflected=" << reflected
          << " Shadowed="  << shadowed << "\n";
```

---

## 6. Testing Functions

The demo executable (`srp_demo`) serves as the primary test harness. Run it and use the interactive menu:

```
srp_demo path/to/h5_files
```

Key menu items for testing:

| Item | Tests |
|---|---|
| **Hardware info** | Confirms CUDA + OptiX runtime are found |
| **CentroidCPU** | Baseline CPU result (reference) |
| **CentroidGPU** | GPU result — compare labels/forces against CPU |
| **CentroidRTX** | RTX result — compare labels/forces against GPU |
| **PixelGridCPU** | Pixel-grid CPU baseline |
| **PixelGridGPU** | Pixel-grid GPU — compare against CPU |
| **PixelGridRTX** | Pixel-grid RTX — compare against GPU |
| **Benchmark** | Times all available methods, prints ns/triangle |
| **Visualize** | Opens interactive 3D plot in browser |

**Validation rules of thumb:**

- `CentroidCPU` vs `CentroidGPU` vs `CentroidRTX`: force/moment should agree to < 0.01%  
- `PixelGrid*` vs `Centroid*`: expect differences on satellites with small/thin facets; differences decrease as `grid_step` decreases  
- After `setMaxReflections(0)` (default): zero entries ≥ 1 in `getBounceLevels()`  
- After `setMaxReflections(2)`: at least some entries = 1 or 2 for geometry with facing surfaces (solar panels)

**Smoke test from code:**

```cpp
#include "SRPLibrary.h"
#include <cassert>

void smoke_test(const std::string& data_path) {
    SRPEngine engine(data_path);
    engine.setSunDirection(0, 1, 0);

    // Basic sanity
    SRPResult r = engine.compute(SRPMethod::CentroidCPU);
    assert(!r.labels.empty());
    assert(r.labels.size() == engine.dataset().triangleCount());
    assert(std::isfinite(r.total_force[0]));

    // Label range
    for (int l : r.labels) assert(l == 0 || l == 1);

    // Some triangles must be illuminated
    int lit = std::count(r.labels.begin(), r.labels.end(), 1);
    assert(lit > 0);

    std::cout << "Smoke test passed: " << lit << "/" << r.labels.size()
              << " triangles illuminated\n";
}
```

---

## 7. Visualization

Visualization requires Python 3.10+ with the following packages (auto-installed on first run if pip is available):

```
numpy  pandas  plotly  matplotlib
```

```cpp
// After any compute() call:
engine.visualizeLastResult();         // open in browser (plotly HTML)
engine.visualizeLastResult(1, false); // all triangles, no normal arrows
engine.visualizeLastResult(3, true);  // every 3rd triangle, show normals

// For bounce-aware coloring (yellow/orange/grey by bounce level):
engine.computeViz(SRPMethod::CentroidRTX);   // must use computeViz, not compute
engine.visualizeLastResult();
```

**Color scheme:**
- 🟡 Yellow — directly illuminated (bounce = 0)  
- 🟠 Orange shades — reflected light (bounce = 1, 2, …)  
- ⬜ Grey — shadowed (bounce = -1)

**Manual Python path** (if auto-detection fails):

```bash
# Windows — set before launching the app:
set PYTHONHOME=C:\Users\you\AppData\Local\Programs\Python\Python312

# Linux / macOS:
export PYTHONHOME=/usr/local
```

Or in VS: **Project → Properties → Debugging → Environment** → add `PYTHONHOME=C:\...\Python312`

---

## Notes on build tiers

The same `SRPLibrary.h` header works for all build tiers. Calling a method that was not compiled in throws `std::runtime_error` — it does **not** crash silently:

```
CentroidGPU requires CUDA. Use CentroidCPU or rebuild with CUDA support.
CentroidRTX requires OptiX. Use CentroidCPU/CentroidGPU or rebuild with OptiX support.
```

Check availability at runtime:

```cpp
// Declared in main.cpp (or your own wrapper):
bool        is_cuda_device_available();
bool        is_rtx_device_available();
std::string rtx_unavailable_reason();

// Usage:
if (is_rtx_device_available()) {
    r = engine.compute(SRPMethod::CentroidRTX);
} else if (is_cuda_device_available()) {
    r = engine.compute(SRPMethod::CentroidGPU);
} else {
    r = engine.compute(SRPMethod::CentroidCPU);
}
```
