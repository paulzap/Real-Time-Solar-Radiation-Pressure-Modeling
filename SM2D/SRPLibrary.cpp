#include "SRPLibrary.h"
#include "SatelliteDataset.h"
#include "ShadowAlgorithms.h"
#include "gpu_optix_raytracer.h"
#include "bench_methods.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <pybind11/embed.h>

namespace py = pybind11;

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
            last_result = calculate_labels_reflections_gpu_bench(
                triangles, sun_vec, max_reflections, /*verbose=*/false);
            break;
        case SRPMethod::CentroidRTX:
            last_result = calculate_labels_ray_casting_rtx_bench(
                triangles, sun_vec, max_reflections);
            break;
        case SRPMethod::PixelGridCPU:
            last_result = calculate_labels_pixel_grid(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
            break;
        case SRPMethod::PixelGridGPU:
            last_result = calculate_labels_pixel_grid_gpu_bench(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
            break;
        case SRPMethod::PixelGridRTX:
            last_result = calculate_labels_pixel_grid_rtx_bench(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
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
            last_result = calculate_labels_ray_casting_reflections_gpu(
                triangles, sun_vec, max_reflections, /*verbose=*/false);
            break;
        case SRPMethod::CentroidRTX:
            last_result = calculate_labels_ray_casting_rtx(
                triangles, sun_vec, max_reflections);
            break;
        case SRPMethod::PixelGridCPU:
            last_result = calculate_labels_pixel_grid(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
            break;
        case SRPMethod::PixelGridGPU:
            last_result = calculate_labels_pixel_grid_gpu(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
            break;
        case SRPMethod::PixelGridRTX:
            last_result = calculate_labels_pixel_grid_rtx(
                triangles, sun_vec, grid_step, max_reflections, /*verbose=*/false);
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
        const std::string tmp_csv = "tmp_srp_result.csv";

        // pybind11 requires a live Python interpreter before any py::module_::import().
        // The embedded interpreter does NOT inherit sys.path from the system
        // Python - without explicit Lib / site-packages entries it cannot find
        // numpy / plotly / matplotlib even if they are installed system-wide.
        //
        // Portability: PYTHONHOME env var takes precedence over the compile-time default.
        // On a new machine set PYTHONHOME in system environment, or in VS:
        //   Project -> Properties -> Debugging -> Environment -> PYTHONHOME=C:\...\Python3XX
        // PYTHONHOME must be set before Py_Initialize() (before scoped_interpreter) to
        // suppress "Could not find platform independent libraries <prefix>".
        if (!Py_IsInitialized()) {
            char*  python_home_buf = nullptr;
            size_t python_home_len = 0;
            _dupenv_s(&python_home_buf, &python_home_len, "PYTHONHOME");

            // Portable fallback: PYTHONHOME > %LOCALAPPDATA%\Programs\Python\Python312 > C:\Python312
            // On a new machine: set PYTHONHOME in system environment or in VS:
            //   Project → Properties → Debugging → Environment → PYTHONHOME=C:\...\Python3XX
            std::string python_root;
            if (python_home_buf && python_home_len > 0) {
                python_root = std::string(python_home_buf);
            } else {
                // Auto-detect Python: scan %LOCALAPPDATA%\Programs\Python\Python3XX
                // and pick the highest version that has include\Python.h.
                // Fallback chain: scan > C:\Python3XX > C:\Python312
                python_root = "";
                char*  localappdata_buf = nullptr;
                size_t localappdata_len = 0;
                _dupenv_s(&localappdata_buf, &localappdata_len, "LOCALAPPDATA");
                if (localappdata_buf && localappdata_len > 0) {
                    std::string py_base = std::string(localappdata_buf) + "\\Programs\\Python";
                    try {
                        for (const auto& entry : std::filesystem::directory_iterator(py_base)) {
                            if (!entry.is_directory()) continue;
                            std::string candidate = entry.path().string();
                            if (std::filesystem::exists(candidate + "\\include\\Python.h")) {
                                // Pick highest version (lexicographic order works: Python310 < Python312 < Python313)
                                if (python_root.empty() || candidate > python_root)
                                    python_root = candidate;
                            }
                        }
                    } catch (...) {}
                }
                free(localappdata_buf);
                // Fallback: try common system-wide paths
                if (python_root.empty()) {
                    for (const char* p : {"C:\\Python313","C:\\Python312","C:\\Python311","C:\\Python310"}) {
                        if (std::filesystem::exists(std::string(p) + "\\include\\Python.h")) {
                            python_root = p; break;
                        }
                    }
                }
                if (python_root.empty())
                    python_root = "C:\\Python312";  // last resort
            }
            const bool had_pythonhome = (python_home_buf != nullptr);
            free(python_home_buf);

            if (!had_pythonhome) {
                _putenv_s("PYTHONHOME", python_root.c_str());
            }
            static py::scoped_interpreter guard{};

            py::module_ sys = py::module_::import("sys");
            py::list path = sys.attr("path");
            // Build sys.path entries dynamically from python_root so they follow
            // whatever Python installation is configured via PYTHONHOME.
            const std::string lib_paths[] = {
                python_root + "\\Lib",
                python_root + "\\Lib\\site-packages",
                python_root + "\\DLLs",
                ".",          // working directory - visualize3d.py lives next to the .exe
            };
            for (const auto& p : lib_paths) path.attr("append")(p.c_str());

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
