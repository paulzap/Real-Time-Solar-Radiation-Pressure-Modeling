# Using SolaRT (SRP Library)

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
   - [Animation](#animation)

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
#           dist\python\visualize3d.py  dist\SRPLibraryConfig.cmake
```

**Linux / macOS:**
```bash
chmod +x setup.sh && ./setup.sh
# outputs: dist/include/SRPLibrary.h  dist/lib/libsrp.a
#           dist/python/visualize3d.py  dist/SRPLibraryConfig.cmake
```

Override non-default paths:
```powershell
.\setup.ps1 -Mode Library -VcpkgRoot D:\tools\vcpkg -OptixRoot "D:\SDK\OptiX 9.1.0"
```
```bash
./setup.sh --vcpkg ~/vcpkg --optix /opt/optix
```

Use `--force` / `-Force` to rebuild from scratch.

After the script: use `find_package(SRPLibrary CONFIG HINTS "path/to/dist")` in your
CMakeLists.txt — it automatically handles all dependencies. See §3 below.

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

### Option A — `find_package` with prebuilt dist/ (recommended)

After running the setup script, `dist/` contains a ready-to-use `SRPLibraryConfig.cmake`
that automatically pulls in all required dependencies (HDF5, Python) based on what was
compiled in. No manual `-DCUDA` defines or `-lpython` flags needed:

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_project LANGUAGES CXX)

# Point CMake at the dist/ folder produced by setup.sh / setup.ps1
find_package(SRPLibrary REQUIRED CONFIG
    HINTS "/path/to/Real-Time-Solar-Radiation-Pressure-Modeling/dist")

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE SRPLibrary::srp)
```

`SRPLibrary::srp` is an imported target — include dirs, Python, and HDF5 are
all set automatically as transitive dependencies.

**Visualization** (`visualizeLastResult`): copy `dist/python/visualize3d.py`
to the directory where your executable runs. The library imports this script
at runtime via `sys.path`.

### Option B — CMake `add_subdirectory` (if you have the full source)

```cmake
add_subdirectory(path/to/Real-Time-Solar-Radiation-Pressure-Modeling)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE srp)
```

### Option C — Link the prebuilt lib in Visual Studio

1. Build the library first (see §2 above) to produce `srp.lib`
2. In your VS project:
   - **Project → Properties → C/C++ → Additional Include Directories** → add path to `dist/include/`
   - **Linker → Input → Additional Dependencies** → add `srp.lib`
   - **Linker → General → Additional Library Directories** → add path to `dist/lib/`
3. Copy `dist/bin/*.ptx` next to your `.exe` (OptiX/RTX methods only)
4. Copy `dist/bin/hdf5.dll` next to your `.exe`
5. Copy `dist/python/visualize3d.py` to your working directory

> **Note:** if `srp.lib` was built with Python support (default), you must also link
> `python3X.lib`. The easiest way is to use Option A (`SRPLibraryConfig.cmake`) — it
> handles this automatically. For manual VS linking, add the Python `.lib` to
> Additional Dependencies and its `libs/` folder to Additional Library Directories.

### Required compile defines (manual linking only)

When **not** using `SRPLibraryConfig.cmake`, replicate the defines used during compilation:

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
std::array<double,3> s = engine.getSunDirection();   // unit vector towards the Sun, body frame

// Reflections (inter-surface bounces, default 0)
engine.setMaxReflections(2);

// Pixel-grid cell size in metres (PixelGrid methods only, default 0.05).
// Choose it from the size of the model: the default suits models of a few
// metres and up; a 0.3 m CubeSat needs about 0.001 m (L / 300 is a good start).
engine.setGridStep(0.02);
double step = engine.getGridStep();

// Moment reference point, body frame [m] (default: body-frame origin).
// total_moment of every result is taken about this point.
engine.setMomentReference(0.40, -0.27, 0.70);
engine.setMomentReferenceToCenterOfMass();   // CoM from the HDF5 /inertia group
auto ref = engine.getMomentReference();

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
const std::vector<Triangle>& tris = ds.triangles();  // body frame, current pose (rebuilt on demand)
int ntri = ds.triangleCount();
std::uint64_t ver = ds.geometryVersion();    // changes on every load/reload/articulate
std::uint64_t sv  = ds.structureVersion();   // changes only when a file is read

// Mass properties (only files with an /inertia group, e.g. SatForm3D output;
// values are for the pose stored in the file, articulation does not update them)
if (ds.hasMassProperties()) {
    double m = ds.totalMass();                       // kg
    std::array<double,3> com = ds.centerOfMass();    // m, body frame
    std::array<double,9> I   = ds.inertiaTensor();   // kg*m^2 about the CoM, row-major
}

// Component inspection
auto comp_names = ds.componentNames();       // {"Body", "SolarPanel_L", ...}
auto comp_infos = ds.componentInfos();       // name + type + triangle count + area
bool exists     = ds.hasComponent("S1_Mirror");
auto s1_tris    = ds.componentTriangles("S1_Mirror");  // vector<const Triangle*>

ds.printInfo();        // total triangles, area, bounding box
ds.printComponents();  // table of all components
ds.printJoints();      // currently active joints

// Joints declared in the file (SatForm3D attributes joint_type, joint_axis,
// joint_min, joint_max). The file decides which components move: only those
// declared "revolute" or "prismatic" can be articulated.
for (const auto& name : ds.movableComponents()) {
    JointInfo ji = ds.jointInfo(name);   // type, axis (local frame), limits
}

// Articulation with the declared joint: type and axis from the file, pivot at
// the origin of the component's local frame, value checked against the limits
ds.setJointValue("SolarArrayGroup_L", 0.7);   // rad (revolute) or m (prismatic), applied at once

// The pose stored in the file: each joint at joint_value (if recorded) or
// joint_default_value; clearJoint() / clearAllJoints() + reload() return to it.
// Coupled joints (SatForm3D 2.3: coupling_driver_path / _ratio / _offset)
// follow their driver, q = ratio * q_driver + offset clipped to the limits;
// movableComponents() lists only the drivers, jointInfo() describes both.
for (const auto& st : ds.deploymentStateNames()) { /* e.g. "deployed", "stowed" */ }
ds.setDeploymentState("deployed");   // every joint that declares the state

// Articulation with an explicit JointConfig (same joint written out)
ds.articulate("SolarArrayGroup_L", JointConfig::rotX(0.7).atOrigin());  // setJoint + reload
ds.setJoint("SolarArrayGroup_L",   JointConfig::rotX(0.7).atOrigin());  // stage only
ds.reload();                                                             // commit staged joints
ds.articulate("DeployableBoom", JointConfig::translate(0, -1, 0, 1.5));

// Rejected with std::invalid_argument: a component declared fixed, the root,
// a joint of the other type, a value outside [joint_min, joint_max], an
// unknown name. An axis not parallel to the declared one is used as given
// (one warning per component).

// Cost of a pose change. A movable component that is out of the pose stored
// in the file is a separate rigid part: the GPU/RTX methods keep one
// acceleration structure per part and a pose change only moves the parts
// (about 0.6 ms for 800k triangles). A component leaving or returning to its
// file pose changes the split and costs one rebuild (0.15-0.6 s); a model in
// the pose of the file is computed exactly as in 1.1.0. articulate() itself
// only records the joint values: triangles() is rebuilt on its next call (CPU
// methods and visualisation call it). A reference to the triangle list taken
// before an articulation shows the old pose until triangles() is called again.

// Joint management
ds.clearJoint("SolarPanel_L");
ds.clearAllJoints();
bool has = ds.hasJoint("SolarPanel_L");
JointConfig jc = ds.getJoint("SolarPanel_L");
const auto& all = ds.joints();   // std::map<std::string, JointConfig>

// Rigid motion of every mesh from the file pose to the current pose (display,
// animation; cheap, builds no triangle list). triangles() is the concatenation
// of the meshes in this order.
for (const auto& m : ds.meshMotions()) {
    // m.name, m.first_triangle, m.triangle_count,
    // m.motion: row-major 3x4 [R | t], p_now = R * p_file + t (identity if not moved)
}
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
// default: pivot = auto centroid of the component mesh. A joint declared in a
// SatForm3D file acts at the local origin, so write .atOrigin() for it, or use
// ds.setJointValue(name, value), which does this for you.

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
extern double g_srp_phi0;   // default: 1.0 (results per unit pressure: m², m³)

// Override before any compute() call:
g_srp_phi0 = 4.56e-6;   // SI: solar pressure at 1 AU, N/m²
g_srp_phi0 = 1.0;        // dimensionless / normalised output (default)
```

`g_srp_phi0` is one process-wide variable. In multi-threaded code, or when the
pressure changes along the orbit, leave it at 1.0 and multiply the returned
force and moment by the current pressure P = 4.56e-6 · (1 AU / r)² yourself.

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

### Animation

`FrameRecorder` writes a sequence of computed frames; `SM2D/animate_srp.py`
(copied next to the demo by CMake) plays it in real time as a self-contained
HTML page with play / pause, a time slider, a speed choice and a camera that
turns with the mouse (wheel: zoom, right button: move). The page draws with
WebGL 2 and works offline; the script needs only Python 3 with `numpy`. It is
meant for display and visual checking: write frames in a separate run, not in
runs that measure time.

```cpp
FrameRecorder rec("results_3d/anim_sat");          // folder, created if needed
for (int k = 0; k < 200; ++k) {
    engine.dataset().setJointValue("SolarArrayGroup_L", gamma[k]);   // pose
    engine.setSunDirection(s[k][0], s[k][1], s[k][2]);               // body frame
    SRPResult r = engine.computeViz(SRPMethod::CentroidRTX);         // labels needed
    rec.addFrame(engine, r, t[k], "gamma = ...", A_bo[k]);   // A_bo: body -> view (optional)
}
rec.finish();                                       // also done by the destructor
```

```
python animate_srp.py results_3d/anim_sat            # writes and opens animation.html
python animate_srp.py results_3d/anim_sat --every 2 --fps 10 --no-open
```

`compute()` on the GPU does not return labels (fast path), so record the
results of `computeViz()`; `addFrame()` throws on a result without labels.
The last argument is the rotation from the body frame to a "view" frame
(row-major). The page can draw either frame: **body** (the satellite stands
still, the sun moves round it) or **view** (the satellite turns as the
attitude law turns it). The star sky of the space look is fixed in the view
frame, so pass the attitude body → inertial frame for a true sky.
Demo section 17 is a complete example with a fixed scenario (deployment, the
sun going once round the satellite, solar arrays following it); it records
body → inertial for an Earth-pointing satellite, so in the view frame the sun
stays still and the body turns under the arrays.

On the page:

- **force and moment vectors** (check boxes): arrows from the moment
  reference point, scaled to the largest value of the recording, drawn over
  the model; their lengths are in the information line, in m² and m³ per unit
  pressure when `g_srp_phi0` = 1, otherwise in N and N·m;
- **look**: *technical* — white background, gold = illuminated, gray =
  shadowed, body axes, the sun's path; *space* — a star sky with the Milky Way,
  the sun with its glow and rays far away, and materials guessed from the
  component types (solar cells, gold foil, white paint, metal); the shadows
  are still those computed by the library. Command line: `--style space`,
  `--frame body|view`.

The page stores the geometry once (16-bit coordinates over the model's
bounding box) and, per frame, the motion of every mesh and one bit per
triangle, all gzip-compressed and unpacked by the browser itself (a current
Chrome, Edge or Firefox): `satellite_62` (842 662 triangles) with 200 frames
gives a 4.5 MB page that changes frames in about 6 ms; small models give
0.1–0.2 MB. For long recordings of large models use `--every`.

**Recording format** (version 1, little-endian), three files in the folder:

| File | Content |
|---|---|
| `scene.json` | `format` = "SolaRT frames", `format_version`, `solart_version`, `model`, `triangles` (N), `frames` (F), `record_bytes`, `pressure_scale` (`g_srp_phi0`), `moment_reference` (m, body frame), `meshes` (list of `name`, `type` = HDF5 `component_type`, `first`, `count`: triangle ranges in `triangles()` order), `notes` (one text per frame) |
| `geometry.bin` | N × 9 float32: vertices of every triangle at the file pose, body frame, m |
| `frames.bin` | F records of `record_bytes` bytes each |

A record: 19 float64 — time, sun direction (3, unit vector towards the sun,
body frame), force (3), moment (3), body-to-view rotation (9, row-major); then
M × 12 float32 — the motion of each mesh, row-major 3×4 `[R | t]`,
`p = R·p_file + t`; then N uint8 — labels (1 illuminated, 0 shadowed). The
vertices of frame k are `R_m · p_file + t_m` for the mesh m of the triangle,
then rotated by the body-to-view matrix.

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
