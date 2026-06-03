#include "SRPLibrary.h"
#include "SatelliteDataset.h"
#include "ShadowAlgorithms.h"
#ifdef SM3D_HAS_OPTIX
#  include "gpu_optix_raytracer.h"
#endif
#include "bench_methods.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#ifdef SM3D_HAS_PYTHON
#  include <pybind11/embed.h>
   namespace py = pybind11;
#endif

// -----------------------------------------------------------------------------
// Private implementation (PImpl)
// -----------------------------------------------------------------------------
struct SRPEngine::Impl {
    SatelliteDataset ds;                 // owns geometry + articulation state
    std::array<double, 3> sun_dir = { 0.0, 1.0, 0.0 };
    int    max_reflections = 0;          // bounces for all methods
    double grid_step       = 0.05;       // pixel-grid cell size [m] (PixelGrid* only)
    SRPResult last_result;

    explicit Impl(const std::string& folder_path) : ds(folder_path) {
        if (ds.fileCount() == 0)
            throw std::runtime_error("No HDF5 files found in " + folder_path);
        ds.load(0);
        if (ds.triangleCount() == 0)
            throw std::runtime_error("No triangles loaded from " + ds.currentFile());
    }

    void setSunDirection(double x, double y, double z) {
        double len = std::sqrt(x*x + y*y + z*z);
        if (len < 1e-12)
            throw std::runtime_error("Sun direction vector is zero");
        sun_dir = { x/len, y/len, z/len };
    }

    void setGridStep(double step) {
        if (!(step > 0.0))
            throw std::runtime_error("Grid step must be > 0");
        grid_step = step;
    }

    // Fast path: GPU/RTX branches use the lean "bench" kernels (no bounce tracking).
    // CPU branches always populate bounce globals internally (cheap on CPU).
    SRPResult compute(SRPMethod method) {
        const auto& triangles = ds.triangles();
        std::vector<double> sun_vec = { sun_dir[0], sun_dir[1], sun_dir[2] };

        switch (method) {
        case SRPMethod::CentroidCPU:
            last_result = calculate_labels_ray_casting_reflections(
                triangles, sun_vec, max_reflections, /*verbose=*/false);
            break;
        case SRPMethod::CentroidGPU:
#ifdef SM3D_HAS_CUDA
            last_result = calculate_labels_reflections_gpu_bench(
                triangles, sun_vec, max_reflections, /*verbose=*/false);
#else
            throw std::runtime_error("CentroidGPU requires CUDA. Use CentroidCPU or rebuild with CUDA support.");
#endif
            break;
        case SRPMethod::CentroidRTX:
#ifdef SM3D_HAS_OPTIX
            last_result = calculate_labels_ray_casting_rtx_bench(
                triangles, sun_vec, max_reflections);
#else
            throw std::runtime_error("CentroidRTX requires OptiX. Use CentroidCPU/CentroidGPU or rebuild with OptiX support.");
#endif
            break;
        case SRPMethod::PixelGridCPU:
            last_result = calculate_labels_pixel_grid(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
            break;
        case SRPMethod::PixelGridGPU:
#ifdef SM3D_HAS_CUDA
            last_result = calculate_labels_pixel_grid_gpu_bench(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
#else
            throw std::runtime_error("PixelGridGPU requires CUDA. Use PixelGridCPU or rebuild with CUDA support.");
#endif
            break;
        case SRPMethod::PixelGridRTX:
#ifdef SM3D_HAS_OPTIX
            last_result = calculate_labels_pixel_grid_rtx_bench(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
#else
            throw std::runtime_error("PixelGridRTX requires OptiX. Use PixelGridCPU/PixelGridGPU or rebuild with OptiX support.");
#endif
            break;
        default:
            throw std::runtime_error("Unknown SRP method");
        }
        return last_result;
    }

    // Visualization path: full bounce tracking (non-bench kernels).
    // After return, getBounceLevels() / getIncidentDirs() / getOriginPts() are valid.
    SRPResult computeViz(SRPMethod method) {
        const auto& triangles = ds.triangles();
        std::vector<double> sun_vec = { sun_dir[0], sun_dir[1], sun_dir[2] };

        switch (method) {
        case SRPMethod::CentroidCPU:
            last_result = calculate_labels_ray_casting_reflections(
                triangles, sun_vec, max_reflections, /*verbose=*/false);
            break;
        case SRPMethod::CentroidGPU:
#ifdef SM3D_HAS_CUDA
            last_result = calculate_labels_ray_casting_reflections_gpu(
                triangles, sun_vec, max_reflections, /*verbose=*/false);
#else
            throw std::runtime_error("CentroidGPU requires CUDA. Use CentroidCPU or rebuild with CUDA support.");
#endif
            break;
        case SRPMethod::CentroidRTX:
#ifdef SM3D_HAS_OPTIX
            last_result = calculate_labels_ray_casting_rtx(
                triangles, sun_vec, max_reflections);
#else
            throw std::runtime_error("CentroidRTX requires OptiX. Use CentroidCPU/CentroidGPU or rebuild with OptiX support.");
#endif
            break;
        case SRPMethod::PixelGridCPU:
            last_result = calculate_labels_pixel_grid(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
            break;
        case SRPMethod::PixelGridGPU:
#ifdef SM3D_HAS_CUDA
            last_result = calculate_labels_pixel_grid_gpu(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
#else
            throw std::runtime_error("PixelGridGPU requires CUDA. Use PixelGridCPU or rebuild with CUDA support.");
#endif
            break;
        case SRPMethod::PixelGridRTX:
#ifdef SM3D_HAS_OPTIX
            last_result = calculate_labels_pixel_grid_rtx(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
#else
            throw std::runtime_error("PixelGridRTX requires OptiX. Use PixelGridCPU/PixelGridGPU or rebuild with OptiX support.");
#endif
            break;
        default:
            throw std::runtime_error("Unknown SRP method");
        }
        return last_result;
    }

    const std::vector<int>& getLabels() const { return last_result.labels; }

    void visualizeLastResult(int every, bool show_normals) const {
        if (last_result.labels.empty()) {
            std::cerr << "No result to visualize. Run compute() first.\n";
            return;
        }
#ifdef SM3D_HAS_PYTHON
        const std::string tmp_csv = "tmp_srp_result.csv";
        // pybind11 requires a live Python interpreter before any py::module_::import().
        // The embedded interpreter does NOT inherit sys.path from the system
        // Python - without explicit Lib / site-packages entries it cannot find
        // numpy / plotly / matplotlib even if they are installed system-wide.
        //
        // Python version used at runtime must match the pybind11 ABI (python312.lib).
        // PYTHONHOME is auto-detected (newest 64-bit CPython on PATH / LOCALAPPDATA).
        // To force a specific installation set SRP_PYTHONHOME before launching.
        // PYTHONHOME must be set before Py_Initialize() (before scoped_interpreter) to
        // suppress "Could not find platform independent libraries <prefix>".
        if (!Py_IsInitialized()) {
            // ── Resolve PYTHONHOME ──────────────────────────────────────────────
            // PYTHONHOME env var always takes precedence (set in system env or in VS
            // Project → Properties → Debugging → Environment → PYTHONHOME=...).
            // If absent we auto-detect the Python installation.
            // Must be set BEFORE Py_Initialize / scoped_interpreter to prevent
            // "Could not find platform independent libraries <prefix>" warnings.

#ifdef _WIN32
            // Do NOT trust the system PYTHONHOME — it may point to a stale Python
            // installation (e.g. Python37-32) that is incompatible with the pybind11
            // version compiled into this binary, causing "SRE module mismatch" crashes.
            // We always auto-detect the newest 64-bit Python and set PYTHONHOME ourselves.
            // To force a specific Python, set SRP_PYTHONHOME (not PYTHONHOME) before launch.
            std::string python_root;

            // 1. SRP_PYTHONHOME — explicit per-app override, separate from system PYTHONHOME
            {
                char* buf = nullptr; size_t len = 0;
                _dupenv_s(&buf, &len, "SRP_PYTHONHOME");
                if (buf && len > 0) python_root = std::string(buf);
                free(buf);
            }

            if (python_root.empty()) {
                // 2. Scan %LOCALAPPDATA%\Programs\Python\ for the newest 64-bit CPython.
                //    Skip names with '-' (e.g. Python37-32 = 32-bit build).
                //    Compare by integer version, not string ("Python37" < "Python312").
                char* la_buf = nullptr; size_t la_len = 0;
                _dupenv_s(&la_buf, &la_len, "LOCALAPPDATA");
                if (la_buf && la_len > 0) {
                    std::string py_base = std::string(la_buf) + "\\Programs\\Python";
                    int best_ver = 0;
                    try {
                        for (const auto& e : std::filesystem::directory_iterator(py_base)) {
                            if (!e.is_directory()) continue;
                            std::string name = e.path().filename().string();
                            if (name.rfind("Python3", 0) != 0) continue; // must start with Python3
                            if (name.find('-') != std::string::npos) continue; // skip 32-bit
                            if (!std::filesystem::exists(e.path().string() + "\\include\\Python.h")) continue;
                            try {
                                int ver = std::stoi(name.substr(7)); // "312" → 312, "310" → 310
                                if (ver > best_ver) { best_ver = ver; python_root = e.path().string(); }
                            } catch (...) {}
                        }
                    } catch (...) {}
                }
                free(la_buf);
            }

            // 3. Fallback: common system-wide install paths
            if (python_root.empty()) {
                for (const char* p : {"C:\\Python313","C:\\Python312","C:\\Python311","C:\\Python310"}) {
                    if (std::filesystem::exists(std::string(p) + "\\include\\Python.h")) {
                        python_root = p; break;
                    }
                }
            }
            if (python_root.empty())
                python_root = "C:\\Users\\zapevalin\\AppData\\Local\\Programs\\Python\\Python312";

            // Always override PYTHONHOME — pybind11 needs it to match the compiled-in version
            _putenv_s("PYTHONHOME", python_root.c_str());
#else // Linux / macOS
            const char* phenv = std::getenv("PYTHONHOME");
            const bool had_pythonhome = (phenv != nullptr);
            std::string python_root = had_pythonhome ? std::string(phenv) : "";

            if (!had_pythonhome) {
                // Auto-detect: try common prefix directories, pick first that has Python.h.
                // Covers system Python (/usr), Homebrew (/opt/homebrew, /usr/local),
                // Conda (/opt/conda, ~/miniconda3, ~/anaconda3).
                std::string home;
                if (const char* h = std::getenv("HOME")) home = h;
                const std::vector<std::string> candidates = {
                    "/usr", "/usr/local", "/opt/homebrew", "/opt/conda",
                    home + "/miniconda3", home + "/anaconda3",
                    home + "/.local",
                };
                for (const auto& prefix : candidates) {
                    // Python header may be python3.X or python3
                    bool found = false;
                    try {
                        for (const auto& entry : std::filesystem::directory_iterator(prefix + "/include")) {
                            if (!entry.is_directory()) continue;
                            std::string name = entry.path().filename().string();
                            if (name.rfind("python3", 0) == 0 &&
                                std::filesystem::exists(entry.path() / "Python.h")) {
                                python_root = prefix;
                                found = true;
                                break;
                            }
                        }
                    } catch (...) {}
                    if (found) break;
                }
                if (!python_root.empty())
                    setenv("PYTHONHOME", python_root.c_str(), /*overwrite=*/1);
            }
#endif // _WIN32
            static py::scoped_interpreter guard{};

            // ── Extend sys.path so stdlib / site-packages are importable ───────
            py::module_ sys = py::module_::import("sys");
            py::list path = sys.attr("path");
            path.attr("append")(".");   // always add working dir (visualize3d.py lives here)

#ifdef _WIN32
            // Windows Python layout: <root>\Lib, <root>\Lib\site-packages, <root>\DLLs
            if (!python_root.empty()) {
                for (const auto& p : {
                    python_root + "\\Lib",
                    python_root + "\\Lib\\site-packages",
                    python_root + "\\DLLs",
                }) path.attr("append")(p.c_str());
            }
#else
            // Linux/macOS: derive versioned lib dirs from sys.prefix + sys.version_info.
            // This is more reliable than hardcoding version numbers.
            if (!python_root.empty()) {
                try {
                    py::object vi = sys.attr("version_info");
                    int maj = vi.attr("major").cast<int>();
                    int min = vi.attr("minor").cast<int>();
                    std::string ver = std::to_string(maj) + "." + std::to_string(min);
                    for (const auto& p : {
                        python_root + "/lib/python" + ver,
                        python_root + "/lib/python" + ver + "/site-packages",
                        python_root + "/lib/python" + ver + "/dist-packages",
                        python_root + "/lib/python" + ver + "/lib-dynload",
                    }) path.attr("append")(p.c_str());
                } catch (...) {}
            }
#endif

            // ── Auto-install missing visualization packages ───────────────────
            // Required by visualize3d.py: numpy, pandas, plotly, matplotlib.
            // Checked once at first initialization; packages are installed into
            // the same Python environment via its own pip (sys.executable -m pip).
            // If pip fails (e.g. corporate proxy, read-only env), a warning is
            // printed and visualization proceeds — it will fail later with a
            // clear ImportError rather than a silent crash.
            try {
                py::module_ importlib_util = py::module_::import("importlib.util");
                py::module_ subprocess_mod = py::module_::import("subprocess");
                const char* required[][2] = {
                    { "numpy",      "numpy"      },
                    { "pandas",     "pandas"     },
                    { "plotly",     "plotly"     },
                    { "matplotlib", "matplotlib" },
                };
                bool any_installed = false;
                for (const auto& pkg : required) {
                    if (importlib_util.attr("find_spec")(pkg[1]).is_none()) {
                        std::cout << "  [setup] Package '" << pkg[0]
                                  << "' not found — installing via pip...\n";
                        std::cout.flush();
                        try {
                            py::list cmd;
                            cmd.append(sys.attr("executable"));
                            cmd.append(py::str("-m"));
                            cmd.append(py::str("pip"));
                            cmd.append(py::str("install"));
                            cmd.append(py::str(pkg[0]));
                            subprocess_mod.attr("check_call")(cmd);
                            std::cout << "  [setup] '" << pkg[0] << "' installed.\n";
                            any_installed = true;
                        } catch (...) {
                            std::cerr << "  [setup] Warning: could not install '"
                                      << pkg[0] << "'.\n"
                                      << "    Run manually: pip install " << pkg[0] << "\n";
                        }
                    }
                }
                if (any_installed) {
                    // Reload site-packages so newly installed packages are importable
                    // in this session without restarting the interpreter.
                    py::module_ importlib = py::module_::import("importlib");
                    py::module_ site = py::module_::import("site");
                    site.attr("main")();
                }
            } catch (const std::exception& ex) {
                std::cerr << "  [setup] Package check skipped: " << ex.what() << "\n";
            }
        }

        std::vector<Triangle> triangles_labeled = ds.triangles();
        for (size_t i = 0; i < triangles_labeled.size(); ++i)
            triangles_labeled[i].label = static_cast<double>(last_result.labels[i]);

        // Use bounce-aware CSV writer when bounce data is available (computeViz path).
        const auto& bounce_levels = get_last_bounce_levels();
        if (!bounce_levels.empty() && bounce_levels.size() == triangles_labeled.size()) {
            save_results_with_bounces(triangles_labeled, last_result.labels, last_result.labels,
                bounce_levels, get_last_incident_dirs(), get_last_origin_pts(), tmp_csv);
        } else {
            save_results(triangles_labeled, last_result.labels, last_result.labels, tmp_csv, false);
        }
        visualize_triangles("results_3d/" + tmp_csv, every, show_normals,
            last_result.total_force, last_result.total_moment);
#else
        (void)every; (void)show_normals;
        std::cerr << "visualizeLastResult() is not available: library was built without Python support.\n"
                     "    Rebuild with SM3D_ENABLE_PYTHON=ON and pybind11 installed.\n";
#endif // SM3D_HAS_PYTHON
    }
};

// -----------------------------------------------------------------------------
// SRPEngine method implementations
// -----------------------------------------------------------------------------
SRPEngine::SRPEngine(const std::string& folder_path)
    : pImpl(std::make_unique<Impl>(folder_path)) {}

SRPEngine::~SRPEngine() = default;

SRPEngine::SRPEngine(SRPEngine&&) noexcept = default;
SRPEngine& SRPEngine::operator=(SRPEngine&&) noexcept = default;

SatelliteDataset&       SRPEngine::dataset()       { return pImpl->ds; }
const SatelliteDataset& SRPEngine::dataset() const { return pImpl->ds; }

void SRPEngine::setMaxReflections(int bounces) { pImpl->max_reflections = bounces; }

void   SRPEngine::setGridStep(double step) { pImpl->setGridStep(step); }
double SRPEngine::getGridStep() const      { return pImpl->grid_step; }

void SRPEngine::setSunDirection(double x, double y, double z) {
    pImpl->setSunDirection(x, y, z);
}

SRPResult SRPEngine::compute(SRPMethod method)    { return pImpl->compute(method); }
SRPResult SRPEngine::computeViz(SRPMethod method) { return pImpl->computeViz(method); }

const std::vector<int>&                          SRPEngine::getLabels()       const { return pImpl->getLabels(); }
const std::vector<int>&                          SRPEngine::getBounceLevels() const { return get_last_bounce_levels(); }
const std::vector<std::array<double,3>>&         SRPEngine::getIncidentDirs() const { return get_last_incident_dirs(); }
const std::vector<std::array<double,3>>&         SRPEngine::getOriginPts()    const { return get_last_origin_pts(); }

void SRPEngine::visualizeLastResult(int every, bool show_normals) const {
    pImpl->visualizeLastResult(every, show_normals);
}
