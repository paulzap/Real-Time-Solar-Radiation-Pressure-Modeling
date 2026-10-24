#include "SRPLibrary.h"
#include "ShadowAlgorithms.h"
#include "bench_methods.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <string>
#include <functional>
#include <cmath>
#include <cstdlib>
#ifdef SM3D_HAS_CUDA
#  include <nvml.h>
#endif

// Declared in SM3D_hw_info.cpp
void        print_hardware_info();
bool        is_cuda_device_available();
bool        is_rtx_device_available();
std::string rtx_unavailable_reason();

namespace fs = std::filesystem;

// =============================================================================
// NVML GPU SM clock monitoring (§12)
// =============================================================================
#ifdef SM3D_HAS_CUDA
namespace {
    struct NvmlContext {
        nvmlDevice_t device = nullptr;
        bool ok = false;
        NvmlContext() {
            if (nvmlInit_v2() == NVML_SUCCESS)
                ok = (nvmlDeviceGetHandleByIndex_v2(0, &device) == NVML_SUCCESS);
        }
        ~NvmlContext() { if (ok) nvmlShutdown(); }
    };

    int gpu_sm_clock_mhz() {          // returns -1 if NVML unavailable
        static NvmlContext ctx;
        unsigned int clk = 0;
        return (ctx.ok && nvmlDeviceGetClockInfo(ctx.device, NVML_CLOCK_SM, &clk)
                       == NVML_SUCCESS) ? static_cast<int>(clk) : -1;
    }

    std::string clock_tag() {         // "[GPU: 1905 MHz]" or "[GPU: N/A]"
        int c = gpu_sm_clock_mhz();
        return c < 0 ? "[GPU: N/A]" : "[GPU: " + std::to_string(c) + " MHz]";
    }
}
#else
namespace {
    // NVML not available in CPU-only builds; clock display silently returns empty.
    inline std::string clock_tag() { return ""; }
}
#endif // SM3D_HAS_CUDA

// Toggle: true = show SM clock at warm-up start/end and on each result line.
// Default OFF so there is zero NVML overhead during normal timed sessions.
static bool g_show_gpu_clock = false;

// Warm-up settings for GPU benchmarks (Section 10 and Main calculation).
// Without warm-up, WDDM keeps GPU at P8 (~135–300 MHz) between runs, inflating
// latency 5–10×. Disable only to intentionally measure cold-start behaviour.
static bool   g_warmup_enabled   = true;
static int    g_warmup_min_calls  = 10;
static double g_warmup_min_ms     = 500.0;

// =============================================================================
// Console helpers
// =============================================================================

static void code(const std::string& s) {
    std::cout << "\n  >> " << s << "\n";
}

static void section(int n, const std::string& title) {
    std::cout << "\n";
    std::cout << "=================================================================\n";
    std::cout << "  SECTION " << n << " - " << title << "\n";
    std::cout << "=================================================================\n";
}

static void printResult(const SRPResult& res, double ms = -1.0)
{
    std::cout << std::fixed << std::setprecision(6);
    if (ms >= 0.0)
        std::cout << "  Time   : " << std::setprecision(3) << ms << " ms\n";
    std::cout << std::setprecision(6);
    std::cout << "  Force  : Fx=" << std::setw(12) << res.total_force[0]
              <<           " Fy=" << std::setw(12) << res.total_force[1]
              <<           " Fz=" << std::setw(12) << res.total_force[2] << "  N\n";
    std::cout << "  Moment : Mx=" << std::setw(12) << res.total_moment[0]
              <<           " My=" << std::setw(12) << res.total_moment[1]
              <<           " Mz=" << std::setw(12) << res.total_moment[2] << "  N*m\n";
    int lit = static_cast<int>(std::count(res.labels.begin(), res.labels.end(), 1));
    std::cout << "  Lit    : " << lit << " / " << res.labels.size() << " triangles\n";
}

static SRPResult runMethod(SRPEngine& engine, SRPMethod method, const std::string& enumStr)
{
    code("engine.compute(SRPMethod::" + enumStr + ");");
    auto t0 = std::chrono::high_resolution_clock::now();
    SRPResult res = engine.compute(method);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    printResult(res, ms);
    return res;
}

static SRPResult runMethodViz(SRPEngine& engine, SRPMethod method, const std::string& enumStr)
{
    code("engine.computeViz(SRPMethod::" + enumStr + ");");
    auto t0 = std::chrono::high_resolution_clock::now();
    SRPResult res = engine.computeViz(method);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    printResult(res, ms);
    return res;
}

// =============================================================================
// Declared joints: the HDF5 file decides which components move (joint_type
// "revolute" / "prismatic") and about / along which axis.
// =============================================================================

// First component the loaded file declares with this joint type, or "".
static std::string firstDeclared(const SatelliteDataset& ds, const std::string& type)
{
    for (const auto& name : ds.movableComponents())
        if (ds.jointInfo(name).type == type) return name;
    return "";
}

// Value moved inside the declared limits.
static double insideLimits(const JointInfo& ji, double v)
{
    return ji.has_limits ? std::min(std::max(v, ji.min_value), ji.max_value) : v;
}

// The declared joint of comp at the given value, as a JointConfig.
static JointConfig declaredJoint(const SatelliteDataset& ds, const std::string& comp, double v)
{
    const JointInfo ji = ds.jointInfo(comp);
    const auto& a = ji.axis;
    return ji.type == "prismatic" ? JointConfig::translate(a[0], a[1], a[2], v)
                                  : JointConfig::rot(a[0], a[1], a[2], v).atOrigin();
}

static void printDeclared(const SatelliteDataset& ds)
{
    code("ds.movableComponents();  ds.jointInfo(name);");
    const auto names = ds.movableComponents();
    if (names.empty()) { std::cout << "  (the file declares no movable components)\n"; return; }
    for (const auto& n : names) {
        const JointInfo ji = ds.jointInfo(n);
        std::cout << "    " << std::left << std::setw(28) << n << std::setw(10) << ji.type
                  << "axis (" << ji.axis[0] << ", " << ji.axis[1] << ", " << ji.axis[2] << ")";
        if (ji.has_limits) std::cout << "  range [" << ji.min_value << ", " << ji.max_value << "]";
        std::cout << "\n";
    }
}

// =============================================================================
// Section functions - each is fully standalone
// =============================================================================

static void section1_dataset_construction(SRPEngine& engine)
{
    section(1, "Dataset construction");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);

    code("SRPEngine engine(folder_path);");
    std::cout << "  Engine already constructed; dataset loaded from folder.\n";

    code("SatelliteDataset& ds = engine.dataset();");

    code("ds.fileCount();");
    std::cout << "  Files in folder : " << ds.fileCount() << "\n";

    code("ds.fileNames();");
    for (int i = 0; i < ds.fileCount(); ++i)
        std::cout << "    [" << i << "] "
                  << fs::path(ds.fileNames()[i]).filename().string() << "\n";

    code("ds.currentFile();");
    std::cout << "  Current file    : "
              << fs::path(ds.currentFile()).filename().string() << "\n";
}

static void section2_geometry_inspection(SRPEngine& engine)
{
    section(2, "Geometry inspection");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);

    code("ds.triangleCount();");
    std::cout << "  Total triangles : " << ds.triangleCount() << "\n";

    code("ds.printInfo();");
    ds.printInfo();

    code("ds.printComponents();");
    ds.printComponents();

    auto compNames = ds.componentNames();
    code("ds.componentNames();");
    std::cout << "  Component list (" << compNames.size() << "):\n";
    for (const auto& n : compNames)
        std::cout << "    " << n << "\n";

    code("ds.componentInfos();");
    auto infos = ds.componentInfos();
    std::cout << "  Largest component by area:\n";
    auto maxIt = std::max_element(infos.begin(), infos.end(),
        [](const SatelliteDataset::ComponentInfo& a,
           const SatelliteDataset::ComponentInfo& b)
        { return a.total_area < b.total_area; });
    if (maxIt != infos.end())
        std::cout << "    " << maxIt->name << "  type=" << maxIt->type
                  << "  tris=" << maxIt->triangle_count
                  << "  area=" << std::fixed << std::setprecision(4)
                  << maxIt->total_area << " m2\n";

    std::string probe = compNames.empty() ? "Bus" : compNames[0];

    code("ds.hasComponent(\"" + probe + "\");");
    std::cout << "  hasComponent(\"" << probe << "\") = "
              << (ds.hasComponent(probe) ? "true" : "false") << "\n";

    code("ds.componentTriangles(\"" + probe + "\");");
    {
        auto ptrs = ds.componentTriangles(probe);
        std::cout << "  Triangles in \"" << probe << "\": " << ptrs.size() << "\n";
        if (!ptrs.empty()) {
            const Triangle& t0 = *ptrs[0];
            std::cout << "    ID         : " << t0.ID << "\n";
            std::cout << "    component  : " << t0.component_name << " (" << t0.component_type << ")\n";
            std::cout << "    centroid   : ("
                      << std::fixed << std::setprecision(4)
                      << t0.centroid_x << ", " << t0.centroid_y << ", " << t0.centroid_z << ")\n";
            std::cout << "    normal     : ("
                      << t0.normal_x << ", " << t0.normal_y << ", " << t0.normal_z << ")\n";
            std::cout << "    area       : " << t0.area << " m2\n";
            std::cout << "    reflectance: " << t0.reflectance << "\n";
            std::cout << "    specularity: " << t0.specularity << "\n";
            std::cout << "    emissivity : " << t0.emissivity  << "\n";
            std::cout << "    label      : " << t0.label << "  (1=lit, 0=shadow)\n";
        }
    }

    code("ds.triangles();  // full vector - iterate to sum area");
    {
        const auto& allTris = ds.triangles();
        double totalArea = 0.0;
        for (const auto& tri : allTris) totalArea += tri.area;
        std::cout << "  Total surface area : "
                  << std::fixed << std::setprecision(4) << totalArea << " m2\n";
    }
}

static void section3_loading_files(SRPEngine& engine)
{
    section(3, "Loading files");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();

    code("ds.load(0);");
    ds.load(0);
    std::cout << "  " << fs::path(ds.currentFile()).filename().string()
              << "  (" << ds.triangleCount() << " triangles)\n";

    if (ds.fileCount() > 1) {
        code("ds.load(1);");
        ds.load(1);
        std::cout << "  " << fs::path(ds.currentFile()).filename().string()
                  << "  (" << ds.triangleCount() << " triangles)\n";
        ds.load(0);
    }

    code("ds.load(ds.fileNames()[0]);  // load by explicit path");
    ds.load(ds.fileNames()[0]);
    std::cout << "  " << fs::path(ds.currentFile()).filename().string()
              << "  (" << ds.triangleCount() << " triangles)\n";

    code("ds.reload();  // re-read same file, re-apply all staged joints");
    ds.reload();
    std::cout << "  After reload: " << ds.triangleCount() << " triangles\n";

    code("ds.load_single_file(path);  // legacy interface");
    ds.load_single_file(ds.fileNames()[0]);
    std::cout << "  Legacy load: " << ds.get_triangles().size() << " triangles\n";

    ds.load(0);
}

static void section4_revolute_joints(SRPEngine& engine)
{
    section(4, "What is a joint?  (+ revolute joints)");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);

    std::cout << "\n"
        "  A JOINT describes a single degree of freedom of one satellite component\n"
        "  relative to its parent body.  Two types are supported:\n"
        "\n"
        "    REVOLUTE  - rotation around an axis.\n"
        "                Example: a solar panel rotating to track the sun.\n"
        "                Value   = angle in radians.\n"
        "\n"
        "    PRISMATIC - translation along an axis.\n"
        "                Example: a telescopic antenna boom extending.\n"
        "                Value   = distance in metres.\n"
        "\n"
        "  The HDF5 file declares which components move: joint_type, joint_axis\n"
        "  (in the component's local frame) and the range joint_min..joint_max.\n"
        "  Other components are fixed and cannot be articulated.\n"
        "\n"
        "  Workflow:\n"
        "    ds.setJointValue(name, value) - the declared joint, applied at once.\n"
        "    ds.setJoint(name, config)     - stage a joint (no geometry change yet).\n"
        "    ds.reload()                   - apply all staged joints.\n"
        "    ds.articulate(name, config)   - setJoint + reload in one call.\n"
        "  A declared joint acts at the origin of the local frame: write it as\n"
        "  JointConfig::rot(axis, angle).atOrigin(). Without .atOrigin() the\n"
        "  component turns about the centroid of its vertices.\n";

    printDeclared(ds);
    const std::string comp = firstDeclared(ds, "revolute");
    if (comp.empty()) { std::cout << "  No revolute joint in this file.\n"; return; }
    const JointInfo ji = ds.jointInfo(comp);
    const auto& a = ji.axis;
    std::cout << "\n  Demonstrating on component: \"" << comp << "\"\n";

    const double a1 = insideLimits(ji, M_PI / 4);
    code("ds.setJointValue(\"" + comp + "\", " + std::to_string(a1) + ");");
    ds.setJointValue(comp, a1);
    std::cout << "  declared joint at " << a1 * 180.0 / M_PI << " deg: "
              << ds.triangleCount() << " triangles\n";

    const double a2 = insideLimits(ji, M_PI / 6);
    code("ds.setJoint(\"" + comp + "\", JointConfig::rot(axis, " + std::to_string(a2) +
         ").atOrigin());  ds.reload();");
    ds.setJoint(comp, JointConfig::rot(a[0], a[1], a[2], a2).atOrigin());
    ds.reload();
    std::cout << "  same joint, JointConfig form: " << ds.triangleCount() << " triangles\n";

    const double a3 = insideLimits(ji, M_PI / 2);
    code("ds.articulate(\"" + comp + "\", JointConfig::rot(axis, " + std::to_string(a3) +
         ").atOrigin());  // setJoint+reload");
    ds.articulate(comp, JointConfig::rot(a[0], a[1], a[2], a3).atOrigin());
    std::cout << "  articulate: " << ds.triangleCount() << " triangles\n";

    std::string fixed_comp;
    const auto movable = ds.movableComponents();
    for (const auto& n : ds.componentNames())
        if (std::find(movable.begin(), movable.end(), n) == movable.end()) { fixed_comp = n; break; }
    if (!fixed_comp.empty()) {
        code("ds.articulate(\"" + fixed_comp + "\", JointConfig::rotZ(0.1));  // declared fixed");
        try {
            ds.articulate(fixed_comp, JointConfig::rotZ(0.1));
        } catch (const std::invalid_argument& e) {
            std::cout << "  rejected: " << e.what() << "\n";
        }
    }

    code("ds.getJoint(\"" + comp + "\");");
    std::cout << "  getJoint.value = " << std::fixed << std::setprecision(4)
              << ds.getJoint(comp).value << " rad\n";

    code("ds.clearJoint(\"" + comp + "\");  ds.reload();");
    ds.clearJoint(comp);
    ds.reload();
    std::cout << "  clearJoint - nominal: " << ds.triangleCount() << " triangles\n";
}

static void section5_prismatic_joints(SRPEngine& engine)
{
    section(5, "Prismatic joints (translation)");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);

    const std::string comp = firstDeclared(ds, "prismatic");
    if (comp.empty()) { std::cout << "  No prismatic joint in this file.\n"; return; }
    const JointInfo ji = ds.jointInfo(comp);
    const auto& a = ji.axis;
    std::cout << "  Demonstrating on component: \"" << comp << "\"\n\n";

    const double d1 = insideLimits(ji, 0.25);
    code("ds.setJointValue(\"" + comp + "\", " + std::to_string(d1) + ");");
    ds.setJointValue(comp, d1);
    std::cout << "  extended by " << d1 << " m\n";

    const double d2 = insideLimits(ji, 1.0);
    code("ds.articulate(\"" + comp + "\", JointConfig::translate(axis, " + std::to_string(d2) + "));");
    ds.articulate(comp, JointConfig::translate(a[0], a[1], a[2], d2));
    std::cout << "  extended by " << d2 << " m\n";

    code("ds.clearJoint(\"" + comp + "\");  ds.reload();");
    ds.clearJoint(comp);
    ds.reload();
    std::cout << "  Cleared - nominal: " << ds.triangleCount() << " triangles\n";
}

static void section6_multiple_joints(SRPEngine& engine)
{
    section(6, "Multiple joints in one reload");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);
    const auto movable = ds.movableComponents();
    if (movable.empty()) { std::cout << "  The file declares no movable components.\n"; return; }

    for (size_t k = 0; k < movable.size() && k < 2; ++k) {
        const std::string& comp = movable[k];
        const JointInfo ji = ds.jointInfo(comp);
        const double v = insideLimits(ji, ji.type == "prismatic" ? 0.3 : M_PI / 6);
        code("ds.setJoint(\"" + comp + "\", <declared joint at " + std::to_string(v) + ">);");
        ds.setJoint(comp, declaredJoint(ds, comp, v));
    }
    code("ds.reload();  // all staged joints applied in one pass");
    ds.reload();
    std::cout << "  Joints applied: " << ds.triangleCount() << " triangles\n";

    code("ds.printJoints();");
    ds.printJoints();

    code("ds.hasJoint(\"" + movable[0] + "\");");
    std::cout << "  hasJoint(\"" << movable[0] << "\") = "
              << (ds.hasJoint(movable[0]) ? "true" : "false") << "\n";

    code("ds.joints();  // full map<name, JointConfig>");
    std::cout << "  All active joints (" << ds.joints().size() << "):\n";
    for (const auto& [name, cfg] : ds.joints())
        std::cout << "    " << std::left << std::setw(24) << name
                  << (cfg.type == JointConfig::Type::REVOLUTE ? "revolute " : "prismatic")
                  << "  value=" << cfg.value << "\n";

    code("ds.clearAllJoints();  ds.reload();");
    ds.clearAllJoints();
    ds.reload();
    std::cout << "  clearAllJoints - nominal: " << ds.triangleCount() << " triangles\n";
}

static void section7_legacy_load(SRPEngine& engine)
{
    section(7, "Legacy load_single_file with Z-angle overrides");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);

    // The legacy overrides are rotations about z: only for joints declared about z.
    std::string comp;
    for (const auto& n : ds.movableComponents()) {
        const JointInfo ji = ds.jointInfo(n);
        if (ji.type == "revolute" && std::fabs(std::fabs(ji.axis[2]) - 1.0) < 1e-12) { comp = n; break; }
    }
    if (!comp.empty()) {
        const double v = insideLimits(ds.jointInfo(comp), M_PI / 3.0);
        code("ds.load_single_file(path, {{\"" + comp + "\", " + std::to_string(v) + "}});");
        std::map<std::string, double> overrides;
        overrides[comp] = v;
        ds.load_single_file(ds.fileNames()[0], overrides);
        std::cout << "  Z-override on \"" << comp
                  << "\": " << ds.get_triangles().size() << " triangles\n";
    } else {
        code("ds.load_single_file(path);  // no joint declared about z in this file");
        ds.load_single_file(ds.fileNames()[0]);
        std::cout << "  " << ds.get_triangles().size() << " triangles\n";
    }

    ds.clearAllJoints();
    ds.load(0);
}

static void section8_display_helpers(SRPEngine& engine)
{
    section(8, "Display helpers");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);

    code("ds.printInfo();");
    ds.printInfo();

    code("ds.printComponents();");
    ds.printComponents();

    code("ds.printJoints();  // empty after clearAllJoints");
    ds.printJoints();
}

static void section9_engine_parameters(SRPEngine& engine)
{
    section(9, "Engine parameters (sun direction, reflections)");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);

    double sun_theta = 60.0, sun_phi = 45.0;
    double t_rad = sun_theta * M_PI / 180.0;
    double p_rad = sun_phi   * M_PI / 180.0;
    double sx = std::sin(t_rad) * std::cos(p_rad);
    double sy = std::sin(t_rad) * std::sin(p_rad);
    double sz = std::cos(t_rad);

    code("engine.setSunDirection(0, 1, 0);  // from +Y axis");
    engine.setSunDirection(0, 1, 0);

    code("engine.setSunDirection(1, 0, 0);  // from +X axis");
    engine.setSunDirection(1, 0, 0);

    code("engine.setSunDirection(sx, sy, sz);  // arbitrary direction");
    engine.setSunDirection(sx, sy, sz);
    std::cout << "  Sun direction: (" << std::fixed << std::setprecision(4)
              << sx << ", " << sy << ", " << sz << ")\n";

    code("engine.setMaxReflections(2);");
    engine.setMaxReflections(2);
    std::cout << "  Max reflection bounces: 2\n";
}

static void section10_srp_computation(SRPEngine& engine)
{
    section(10, "SRP computation methods");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);

    // ── Interactive parameters ────────────────────────────────────────────────
    // Default sun direction: (0, 1, 0) - matches the CC12 SM3D_benchmark.cpp
    // default for apples-to-apples comparison.
    double sx = 0.0, sy = 1.0, sz = 0.0;
    int    max_reflections = 2;
    double grid_step       = 0.005;

    std::cout << "\n";
    std::cout << "  Default benchmark parameters (press Enter to keep, or type new value):\n";
    std::cout << "    Sun vector : (" << sx << ", " << sy << ", " << sz << ")\n";
    std::cout << "    Max reflections: " << max_reflections << "\n";
    std::cout << "    Grid step (PixelGrid only): " << grid_step << " m\n";
    std::cout << "\n";

    auto prompt_double = [](const char* label, double def) -> double {
        std::cout << "  " << label << " [" << def << "]: ";
        std::string ln;
        if (!std::getline(std::cin, ln)) return def;
        while (!ln.empty() && std::isspace(static_cast<unsigned char>(ln.front()))) ln.erase(ln.begin());
        while (!ln.empty() && std::isspace(static_cast<unsigned char>(ln.back())))  ln.pop_back();
        if (ln.empty()) return def;
        try { return std::stod(ln); } catch (...) { return def; }
    };
    auto prompt_int = [](const char* label, int def) -> int {
        std::cout << "  " << label << " [" << def << "]: ";
        std::string ln;
        if (!std::getline(std::cin, ln)) return def;
        while (!ln.empty() && std::isspace(static_cast<unsigned char>(ln.front()))) ln.erase(ln.begin());
        while (!ln.empty() && std::isspace(static_cast<unsigned char>(ln.back())))  ln.pop_back();
        if (ln.empty()) return def;
        try { return std::stoi(ln); } catch (...) { return def; }
    };

    // Sun vector entry: three numbers in one line (space-separated), or Enter for default.
    {
        std::cout << "  Sun vector (Sx Sy Sz, space-separated) [" << sx << " " << sy << " " << sz << "]: ";
        std::string ln;
        if (std::getline(std::cin, ln)) {
            while (!ln.empty() && std::isspace(static_cast<unsigned char>(ln.front()))) ln.erase(ln.begin());
            if (!ln.empty()) {
                std::istringstream iss(ln);
                double a, b, c;
                if (iss >> a >> b >> c) {
                    double mag = std::sqrt(a*a + b*b + c*c);
                    if (mag > 1e-12) { sx = a; sy = b; sz = c; }
                }
            }
        }
    }
    max_reflections = std::max(0, prompt_int("Number of reflections", max_reflections));
    grid_step       = std::max(1e-9, prompt_double("Grid step (m) for PixelGrid", grid_step));

    engine.setSunDirection(sx, sy, sz);
    engine.setMaxReflections(max_reflections);
    engine.setGridStep(grid_step);

    // CC12-style benchmark: direct calls to bench-functions (no SRPEngine wrapper),
    // 10 timed runs averaged. Mirrors SM3D_benchmark.cpp:run_benchmark_mode.
    constexpr int N_RUNS = 10;

    const auto& triangles = ds.triangles();
    const std::vector<double> sun_vec = { sx, sy, sz };

    std::cout << "\n";
    std::cout << "  Six methods (Centroid + PixelGrid, 3 backends each).\n";
    std::cout << "  Triangles : " << triangles.size() << "\n";
    std::cout << "  Sun vector : (" << sx << ", " << sy << ", " << sz << ")\n";
    std::cout << "  Reflections: " << max_reflections << "\n";
    std::cout << "  Grid step  : " << grid_step << " m  (PixelGrid only)\n";
    if (g_warmup_enabled)
        std::cout << "  Protocol  : warm-up (>= " << (int)g_warmup_min_ms << " ms / >= "
                  << g_warmup_min_calls << " calls) + " << N_RUNS << " timed runs, averaged.\n\n";
    else
        std::cout << "  Protocol  : warm-up DISABLED + " << N_RUNS
                  << " timed runs (cold-start, GPU may be at P8).\n\n";

    using BenchFn = std::function<SRPResult(const std::vector<Triangle>&, const std::vector<double>&)>;

    // NVIDIA GPU P-state drops to base clock within ~1 s of idle, slashing
    // performance ~5× (3 ms → 15 ms). Warm-up parameters are controlled via
    // the W menu option. CPU methods never need a warm-up.
    auto run_bench = [&](const char* display_name, BenchFn func, bool is_gpu) {
        if (is_gpu) {
            // Capture clock before warm-up (NVML calls are outside timed region)
            std::string clk_before = g_show_gpu_clock ? clock_tag() : "";
            if (g_warmup_enabled) {
                auto   tw0    = std::chrono::high_resolution_clock::now();
                int    wcalls = 0;
                double wms    = 0.0;
                do {
                    func(triangles, sun_vec);
                    ++wcalls;
                    wms = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::high_resolution_clock::now() - tw0).count() / 1000.0;
                } while (wcalls < g_warmup_min_calls || wms < g_warmup_min_ms);
                if (g_show_gpu_clock)
                    std::cout << "  [warm-up] " << display_name << "  "
                              << clk_before << " -> " << clock_tag() << "\n";
            } else if (g_show_gpu_clock) {
                std::cout << "  [warm-up disabled]  " << display_name
                          << "  " << clk_before << "\n";
            }
        }
        double total_ms = 0.0;
        SRPResult last;
        for (int i = 0; i < N_RUNS; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            last = func(triangles, sun_vec);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        }
        double avg_ms = total_ms / N_RUNS;
        std::cout << "  " << std::left << std::setw(18) << display_name
                  << ": avg = " << std::fixed << std::setprecision(3) << avg_ms << " ms";
        if (g_show_gpu_clock) std::cout << "  " << clock_tag();
        std::cout << "\n";
        std::cout << std::setprecision(6);
        std::cout << "    Force  = [" << last.total_force[0]  << ", "
                                      << last.total_force[1]  << ", "
                                      << last.total_force[2]  << "] N\n";
        std::cout << "    Moment = [" << last.total_moment[0] << ", "
                                      << last.total_moment[1] << ", "
                                      << last.total_moment[2] << "] N*m\n\n";
    };

#ifdef SM3D_HAS_OPTIX
    run_bench("CentroidRTX",
        [&](const auto& t, const auto& s) {
            return calculate_labels_ray_casting_rtx_bench(t, s, max_reflections);
        }, /*is_gpu=*/true);
#endif // SM3D_HAS_OPTIX

#ifdef SM3D_HAS_CUDA
    run_bench("CentroidGPU",
        [&](const auto& t, const auto& s) {
            return calculate_labels_reflections_gpu_bench(t, s, max_reflections, false);
        }, /*is_gpu=*/true);
#endif // SM3D_HAS_CUDA

    run_bench("CentroidCPU",
        [&](const auto& t, const auto& s) {
            return calculate_labels_ray_casting_reflections(t, s, max_reflections, false);
        }, /*is_gpu=*/false);

#ifdef SM3D_HAS_OPTIX
    run_bench("PixelGridRTX",
        [&](const auto& t, const auto& s) {
            return calculate_labels_pixel_grid_rtx_bench(t, s, grid_step, max_reflections, false);
        }, /*is_gpu=*/true);
#endif // SM3D_HAS_OPTIX

#ifdef SM3D_HAS_CUDA
    run_bench("PixelGridGPU",
        [&](const auto& t, const auto& s) {
            return calculate_labels_pixel_grid_gpu_bench(t, s, grid_step, max_reflections, false);
        }, /*is_gpu=*/true);
#endif // SM3D_HAS_CUDA

    run_bench("PixelGridCPU",
        [&](const auto& t, const auto& s) {
            return calculate_labels_pixel_grid(t, s, grid_step, max_reflections, false);
        }, /*is_gpu=*/false);
}

static void section11_labels_readback(SRPEngine& engine)
{
    section(11, "Labels read-back after compute");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);
    engine.setSunDirection(0.5, 0.7, 0.5);
    engine.setMaxReflections(0);

    code("SRPResult last = engine.compute(SRPMethod::CentroidRTX);");
    SRPResult last = engine.compute(SRPMethod::CentroidRTX);

    code("last.labels  (via SRPResult struct)");
    {
        int lit  = static_cast<int>(std::count(last.labels.begin(), last.labels.end(), 1));
        int dark = static_cast<int>(last.labels.size()) - lit;
        std::cout << "  lit=" << lit << "  dark=" << dark << "\n";
    }

    code("engine.getLabels()  (same data, no copy)");
    {
        const std::vector<int>& lbl = engine.getLabels();
        int lit  = static_cast<int>(std::count(lbl.begin(), lbl.end(), 1));
        int dark = static_cast<int>(lbl.size()) - lit;
        std::cout << "  lit=" << lit << "  dark=" << dark << "\n";
    }

    code("per-component illumination ratio");
    {
        // compute() (bench) returns empty labels — per-triangle data requires computeViz().
        // Run computeViz here so the table is populated.
        SRPResult viz = engine.computeViz(SRPMethod::CentroidRTX);
        const std::vector<int>& lbl = engine.getLabels();

        if (lbl.empty()) {
            std::cout << "  (labels not available — computeViz() returned no data)\n";
        } else {
            std::cout << "\n  " << std::left << std::setw(26) << "Component"
                      << std::setw(10) << "Lit" << "Total\n";
            std::cout << "  " << std::string(42, '-') << "\n";
            for (const auto& ci : ds.componentInfos()) {
                int comp_lit = 0;
                for (size_t idx = 0; idx < ds.triangles().size(); ++idx)
                    if (ds.triangles()[idx].component_name == ci.name && lbl[idx] == 1)
                        ++comp_lit;
                std::cout << "  " << std::setw(26) << ci.name
                          << std::setw(10) << comp_lit << ci.triangle_count << "\n";
            }
        }
    }
}

static void section12_articulate_recompute(SRPEngine& engine)
{
    section(12, "Articulate, recompute, compare");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);
    engine.setSunDirection(0.5, 0.7, 0.5);
    engine.setMaxReflections(0);
    const std::string comp = firstDeclared(ds, "revolute");

    if (!comp.empty()) {
        const double v = insideLimits(ds.jointInfo(comp), M_PI / 2);
        code("ds.setJointValue(\"" + comp + "\", " + std::to_string(v) + ");");
        ds.setJointValue(comp, v);
        std::cout << "  Component \"" << comp << "\" turned by " << v * 180.0 / M_PI
                  << " deg about its declared axis.\n";

        code("engine.compute(SRPMethod::CentroidRTX);  // rotated geometry");
        SRPResult after = engine.compute(SRPMethod::CentroidRTX);
        std::cout << "  After rotation:\n";
        printResult(after);

        code("ds.clearAllJoints();  ds.reload();");
        ds.clearAllJoints();
        ds.reload();

        code("engine.compute(SRPMethod::CentroidRTX);  // nominal geometry");
        SRPResult nominal = engine.compute(SRPMethod::CentroidRTX);
        std::cout << "  Nominal:\n";
        printResult(nominal);

        std::cout << "\n  Force delta:  dFx="
                  << std::fixed << std::setprecision(6)
                  << (after.total_force[0] - nominal.total_force[0]) << "  dFy="
                  << (after.total_force[1] - nominal.total_force[1]) << "  dFz="
                  << (after.total_force[2] - nominal.total_force[2]) << "  N\n";
        std::cout << "  Moment delta: dMx="
                  << (after.total_moment[0] - nominal.total_moment[0]) << "  dMy="
                  << (after.total_moment[1] - nominal.total_moment[1]) << "  dMz="
                  << (after.total_moment[2] - nominal.total_moment[2]) << "  N*m\n";
    }
}

static void section13_azimuth_sweep(SRPEngine& engine)
{
    section(13, "Sun azimuth sweep (full force + moment vectors)");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);
    engine.setMaxReflections(0);

    code("for az in [0,45,90,135,180]: engine.setSunDirection(...); engine.compute(CentroidRTX);");
    std::cout << "\n";
    std::cout << "  " << std::left
              << std::setw(9)  << "Az(deg)"
              << std::setw(7)  << "Lit"
              << std::setw(13) << "Fx(N)"
              << std::setw(13) << "Fy(N)"
              << std::setw(13) << "Fz(N)"
              << std::setw(13) << "Mx(N*m)"
              << std::setw(13) << "My(N*m)"
              <<                  "Mz(N*m)" << "\n";
    std::cout << "  " << std::string(94, '-') << "\n";

    for (int az = 0; az <= 180; az += 45) {
        double phi_r = az * M_PI / 180.0;
        engine.setSunDirection(std::cos(phi_r), std::sin(phi_r), 0.0);
        SRPResult sr = engine.compute(SRPMethod::CentroidRTX);
        int nlit = static_cast<int>(std::count(sr.labels.begin(), sr.labels.end(), 1));
        std::cout << std::fixed << std::setprecision(4)
                  << "  " << std::left
                  << std::setw(9)  << az
                  << std::setw(7)  << nlit
                  << std::setw(13) << sr.total_force[0]
                  << std::setw(13) << sr.total_force[1]
                  << std::setw(13) << sr.total_force[2]
                  << std::setw(13) << sr.total_moment[0]
                  << std::setw(13) << sr.total_moment[1]
                  <<                  sr.total_moment[2] << "\n";
    }
}

static void section14_visualize(SRPEngine& engine)
{
    section(14, "Visualisation (RTX, no reflections)");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);
    engine.setSunDirection(0.5, 0.7, 0.5);
    engine.setMaxReflections(0);

    code("engine.compute(SRPMethod::CentroidRTX);");
    engine.compute(SRPMethod::CentroidRTX);

    code("engine.visualizeLastResult(1, false);");
    try {
        engine.visualizeLastResult(1, false);
    }
    catch (const std::exception& e) {
        std::cerr << "  Visualisation skipped: " << e.what() << "\n";
        std::cerr << "  (Install Python + matplotlib/plotly to enable.)\n";
    }
}

static void section15_viz_sweep(SRPEngine& engine)
{
    section(15, "SRP visualization sweep - all methods, reflections 0..3");

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);
    engine.setSunDirection(0.5, 0.7, 0.5);

    std::cout << "\n"
        "  Runs engine.computeViz() for every combination of method and\n"
        "  reflection count 0..3.  computeViz() populates bounce globals\n"
        "  (getBounceLevels / getIncidentDirs / getOriginPts) and then\n"
        "  calls visualizeLastResult() to open the visualizer.\n\n"
        "  Outer loop : reflections = 0, 1, 2, 3\n"
        "  Inner loop : CentroidRTX  ->  ReflectionsGPU  ->  ReflectionsCPU\n\n";

    const SRPMethod methods[] = {
        SRPMethod::CentroidRTX,
        SRPMethod::CentroidGPU,
        SRPMethod::CentroidCPU
    };
    const char* methodNames[] = {
        "CentroidRTX",
        "CentroidGPU",
        "CentroidCPU"
    };

    for (int nrefl = 0; nrefl <= 3; ++nrefl) {
        std::cout << "\n--- Reflections: " << nrefl << " ---\n";
        engine.setMaxReflections(nrefl);

        for (int mi = 0; mi < 3; ++mi) {
            std::cout << "\n  [ " << methodNames[mi] << " ]\n";
            SRPResult res = runMethodViz(engine, methods[mi], methodNames[mi]);

            // Print bounce summary
            const auto& blevels = engine.getBounceLevels();
            if (!blevels.empty()) {
                int direct = 0, refl1 = 0, refl2plus = 0, nhit = 0;
                for (int b : blevels) {
                    if      (b == 0)  ++direct;
                    else if (b == 1)  ++refl1;
                    else if (b >= 2)  ++refl2plus;
                    else if (b == -1) ++nhit;
                }
                std::cout << "  Bounce  : direct=" << direct
                          << "  refl1=" << refl1
                          << "  refl2+=" << refl2plus
                          << "  unhit=" << nhit << "\n";
            }

            try {
                engine.visualizeLastResult(1, false);
            }
            catch (const std::exception& e) {
                std::cerr << "  Visualisation skipped: " << e.what() << "\n";
            }
        }
    }
}

static void section16_hardware_info(SRPEngine& /*engine*/)
{
    section(16, "Hardware info (GPU, CUDA, OptiX/RTX support)");
    print_hardware_info();
}

// =============================================================================
// Section 17 - animation. A fixed scenario instead of an orbit: the movable
// parts declared in the file deploy, then the sun goes once round the
// satellite while the solar arrays follow it and the other joints swing.
// Every frame is computed and written with FrameRecorder; animate_srp.py
// turns the recording into an HTML page. For display and visual checking.
// =============================================================================

// v turned about the unit axis a by phi (right hand).
static std::array<double, 3> rotateAbout(const std::array<double, 3>& a,
                                         const std::array<double, 3>& v, double phi)
{
    const double c = std::cos(phi), s = std::sin(phi);
    const double d = a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
    const std::array<double, 3> x = { a[1] * v[2] - a[2] * v[1],
                                      a[2] * v[0] - a[0] * v[2],
                                      a[0] * v[1] - a[1] * v[0] };
    std::array<double, 3> r;
    for (int i = 0; i < 3; ++i) r[i] = v[i] * c + x[i] * s + a[i] * d * (1.0 - c);
    return r;
}

// A revolute joint that follows the sun: its axis in the body frame and the
// triangles it turns (normal at the file pose, area).
struct SunTracker {
    std::string name;
    JointInfo   info;
    double      value = 0.0;
    std::array<double, 3> axis{};
    std::vector<std::array<double, 4>> faces;
};

// Axis and moving triangles of a joint, found from two trial poses with the
// public API only (meshMotions); the dataset is left at the file pose.
static SunTracker probeTracker(SatelliteDataset& ds, const std::string& name)
{
    SunTracker t;
    t.name  = name;
    t.info  = ds.jointInfo(name);
    t.value = t.info.file_value;
    const double d  = (!t.info.has_limits || t.value + 0.6 <= t.info.max_value) ? 0.3 : -0.3;
    const double p1 = t.value + d, p2 = t.value + 2 * d;
    ds.setJointValue(name, p1);
    const auto m1 = ds.meshMotions();
    ds.setJointValue(name, p2);
    const auto m2 = ds.meshMotions();
    ds.clearJoint(name);
    ds.reload();

    const std::vector<Triangle>& tris = ds.triangles();   // file pose
    bool have_axis = false;
    for (size_t k = 0; k < m1.size(); ++k) {
        if (m1[k].motion == m2[k].motion) continue;        // not turned by this joint
        if (!have_axis) {
            // R2 * R1^T is the turn by p2 - p1 about the body-frame axis
            double R[3][3] = {};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int c = 0; c < 3; ++c)
                        R[i][j] += m2[k].motion[i * 4 + c] * m1[k].motion[j * 4 + c];
            std::array<double, 3> a = { R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1] };
            const double len = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
            const double sgn = (std::sin(p2 - p1) > 0) ? 1.0 : -1.0;
            for (auto& x : a) x *= sgn / len;
            t.axis = a;
            have_axis = true;
        }
        for (int i = m1[k].first_triangle; i < m1[k].first_triangle + m1[k].triangle_count; ++i)
            t.faces.push_back({ tris[i].normal_x, tris[i].normal_y, tris[i].normal_z, tris[i].area });
    }
    return t;
}

// Area of the tracker's triangles facing the sun s when the joint is at gamma.
static double sunFacingArea(const SunTracker& t, const std::array<double, 3>& s, double gamma)
{
    const auto sl = rotateAbout(t.axis, s, -(gamma - t.info.file_value));   // sun seen at the file pose
    double area = 0.0;
    for (const auto& f : t.faces) {
        const double c = f[0] * sl[0] + f[1] * sl[1] + f[2] * sl[2];
        if (c > 0.0) area += f[3] * c;
    }
    return area;
}

// Joint value turning the most area to the sun, on a 1-degree grid. After the
// first frame only near the previous value, so the array turns smoothly and
// does not jump to its other side. A full-turn range wraps; a shorter one
// stops at its limit and swings back when the far side faces the sun better.
static double trackSun(SunTracker& t, const std::array<double, 3>& s, bool first)
{
    const double lo = t.info.has_limits ? t.info.min_value : -M_PI;
    const double hi = t.info.has_limits ? t.info.max_value :  M_PI;
    const bool   full = hi - lo >= 2 * M_PI - 0.01;
    const double step = M_PI / 180.0;
    auto search = [&](double from, int n, double& best, double& best_area) {
        for (int i = 0; i <= n; ++i) {
            double v = from + i * step;
            if (full && v > hi) v -= 2 * M_PI;
            if (full && v < lo) v += 2 * M_PI;
            v = std::min(std::max(v, lo), hi);
            const double area = sunFacingArea(t, s, v);
            if (area > best_area + 1e-12) { best_area = area; best = v; }
        }
    };
    double best = t.value, best_area = -1.0;
    if (first) search(lo, static_cast<int>((hi - lo) / step), best, best_area);
    else       search(t.value - 30 * step, 60, best, best_area);
    if (!first && !full) {
        double g = best, g_area = -1.0;
        search(lo, static_cast<int>((hi - lo) / step), g, g_area);
        if (g_area > 1.02 * best_area) best = g;
    }
    t.value = best;
    return best;
}

static void section17_animation(SRPEngine& engine)
{
    section(17, "Animation: moving sun, solar arrays following it, deployment");

    const int    kDeployFrames = 40;              // parts deploy, sun fixed
    const int    kOrbitFrames  = 160;             // sun goes once round the satellite
    const double kFrameTime    = 36.0;            // s per frame (orbit of 96 min)
    const double kSunElevation = 20.0 * M_PI / 180.0;

    SatelliteDataset& ds = engine.dataset();
    ds.clearAllJoints();
    ds.load(0);
    engine.setMaxReflections(0);

    // Role of every declared joint in the scenario.
    std::vector<std::pair<std::string, std::array<double, 2>>> deploy;   // from, to
    std::vector<std::pair<std::string, std::array<double, 2>>> swing;    // centre, amplitude
    std::vector<SunTracker> trackers;
    for (const auto& name : ds.movableComponents()) {
        const JointInfo ji = ds.jointInfo(name);
        const auto st = ji.deploy_states.find("stowed"), dp = ji.deploy_states.find("deployed");
        const double span = ji.max_value - ji.min_value;
        if (st != ji.deploy_states.end() && dp != ji.deploy_states.end()) {
            deploy.push_back({ name, { insideLimits(ji, st->second), insideLimits(ji, dp->second) } });
        } else if (ji.type == "prismatic") {
            deploy.push_back({ name, { ji.has_limits ? ji.min_value : ji.file_value,
                                       ji.has_limits ? ji.max_value : ji.file_value + 1.0 } });
        } else if (!ji.has_limits || span >= 1.5 * M_PI) {
            trackers.push_back(probeTracker(ds, name));
        } else {
            const double room = std::min(ji.max_value - ji.file_value, ji.file_value - ji.min_value);
            swing.push_back({ name, room > 0.05 ? std::array<double, 2>{ ji.file_value, 0.6 * room }
                                                : std::array<double, 2>{ 0.5 * (ji.min_value + ji.max_value),
                                                                         0.3 * span } });
        }
    }
    auto clean = [](double x) { return std::fabs(x) < 1e-9 ? 0.0 : x; };
    std::cout << "  Scenario for " << fs::path(ds.currentFile()).filename().string() << ":\n"
              << std::fixed << std::setprecision(3);
    for (const auto& d : deploy)
        std::cout << "    deploy  " << std::left << std::setw(40) << d.first
                  << d.second[0] << " -> " << d.second[1] << "\n";
    for (const auto& t : trackers)
        std::cout << "    follow  " << std::left << std::setw(40) << t.name << "axis ("
                  << clean(t.axis[0]) << ", " << clean(t.axis[1]) << ", " << clean(t.axis[2])
                  << "), " << t.faces.size() << " triangles\n";
    for (const auto& s : swing)
        std::cout << "    swing   " << std::left << std::setw(40) << s.first
                  << "amplitude " << s.second[1] << " rad\n";

    // The sun circles the axis of the largest solar array (body z if none),
    // kSunElevation above the plane the array turns in.
    auto area = [](const SunTracker& t) {
        double s = 0.0;
        for (const auto& f : t.faces) s += f[3];
        return s;
    };
    std::stable_sort(trackers.begin(), trackers.end(),
                     [&](const SunTracker& x, const SunTracker& y) { return area(x) > area(y); });
    const std::array<double, 3> a = trackers.empty() ? std::array<double, 3>{ 0, 0, 1 } : trackers[0].axis;
    const std::array<double, 3> ref = (std::fabs(a[2]) < 0.9) ? std::array<double, 3>{ 0, 0, 1 }
                                                               : std::array<double, 3>{ 1, 0, 0 };
    std::array<double, 3> e1 = { a[1] * ref[2] - a[2] * ref[1], a[2] * ref[0] - a[0] * ref[2],
                                 a[0] * ref[1] - a[1] * ref[0] };
    const double l1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    for (auto& x : e1) x /= l1;
    const std::array<double, 3> e2 = { a[1] * e1[2] - a[2] * e1[1], a[2] * e1[0] - a[0] * e1[2],
                                       a[0] * e1[1] - a[1] * e1[0] };

    SRPMethod method = SRPMethod::CentroidCPU;
    std::string method_name = "CentroidCPU";
    if (is_rtx_device_available())       { method = SRPMethod::CentroidRTX; method_name = "CentroidRTX"; }
    else if (is_cuda_device_available()) { method = SRPMethod::CentroidGPU; method_name = "CentroidGPU"; }

    const std::string out_dir = "results_3d/anim_" + fs::path(ds.currentFile()).stem().string();
    std::cout << "\n  Method " << method_name << ", " << kDeployFrames + kOrbitFrames
              << " frames -> " << out_dir << "\n";
    code("FrameRecorder rec(dir);  ... ds.setJointValue(...); engine.setSunDirection(...);");
    code("SRPResult r = engine.computeViz(method);  rec.addFrame(engine, r, t, note);");

    FrameRecorder rec(out_dir);
    std::vector<double> ms;
    const double theta0 = 30.0 * M_PI / 180.0;
    const int frames = kDeployFrames + kOrbitFrames;
    for (int f = 0; f < frames; ++f) {
        const bool   deploying = f < kDeployFrames;
        const double u = deploying ? double(f) / (kDeployFrames - 1) : 1.0;
        const double w = u * u * (3.0 - 2.0 * u);                          // smooth start and stop
        const double orbit = deploying ? 0.0 : double(f - kDeployFrames) / kOrbitFrames;
        const double theta = theta0 + 2 * M_PI * orbit;

        for (const auto& d : deploy)
            ds.setJointValue(d.first, d.second[0] + (d.second[1] - d.second[0]) * w);
        for (const auto& s : swing) {
            if (deploying) break;
            const JointInfo ji = ds.jointInfo(s.first);
            const double v = ji.file_value + (s.second[0] - ji.file_value) * std::min(1.0, 8.0 * orbit)
                           + s.second[1] * std::sin(2 * M_PI * orbit);   // eases in from the file value
            ds.setJointValue(s.first, insideLimits(ji, v));
        }
        std::array<double, 3> sun;
        for (int i = 0; i < 3; ++i)
            sun[i] = std::cos(kSunElevation) * (std::cos(theta) * e1[i] + std::sin(theta) * e2[i])
                   + std::sin(kSunElevation) * a[i];
        for (auto& t : trackers)
            ds.setJointValue(t.name, trackSun(t, sun, f == 0));
        engine.setSunDirection(sun[0], sun[1], sun[2]);

        const auto t0 = std::chrono::high_resolution_clock::now();
        const SRPResult r = engine.computeViz(method);   // with labels (the fast path has none)
        const auto t1 = std::chrono::high_resolution_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        std::ostringstream note;
        note << std::fixed << std::setprecision(0);
        if (deploying) note << "deployment " << 100.0 * w << "%";
        else           note << "sun " << 360.0 * orbit << " deg round the satellite";
        if (!trackers.empty()) note << ", array " << trackers[0].value * 180.0 / M_PI << " deg";
        rec.addFrame(engine, r, f * kFrameTime, note.str());
    }
    rec.finish();
    ds.clearAllJoints();
    ds.reload();

    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    std::cout << std::fixed << std::setprecision(2)
              << "  Computed " << frames << " frames: median " << sorted[sorted.size() / 2]
              << " ms, longest " << sorted.back()
              << " ms (computeViz; a part leaving or returning to its file pose rebuilds once)\n";

    const std::string cmd = "python animate_srp.py \"" + out_dir + "\"";
    std::cout << "  Player: " << cmd << "\n";
    std::string skip;                                  // SOLART_NO_PLAYER=1: do not start it
#ifdef _WIN32
    char* buf = nullptr; size_t len = 0;
    if (_dupenv_s(&buf, &len, "SOLART_NO_PLAYER") == 0 && buf) skip = buf;
    free(buf);
#else
    if (const char* v = std::getenv("SOLART_NO_PLAYER")) skip = v;
#endif
    if (fs::exists("animate_srp.py") && (skip.empty() || skip == "0")) {
        if (std::system(cmd.c_str()) != 0)
            std::cout << "  (the player did not run: needs Python 3 with numpy)\n";
    }
}

// =============================================================================
// Main calculation - interactive; user selects method or auto-picks fastest
// =============================================================================

static void main_calculation(SRPEngine& engine)
{
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  MAIN CALCULATION\n";
    std::cout << "================================================================\n\n";

    // ── a) Sun vector ────────────────────────────────────────────────────────
    double sx = 0.0, sy = 1.0, sz = 0.0;
    while (true) {
        std::cout << "  a) Sun vector  (Sx Sy Sz, space-separated): ";
        std::string ln;
        if (!std::getline(std::cin, ln)) return;
        std::istringstream iss(ln);
        if (iss >> sx >> sy >> sz) {
            double mag = std::sqrt(sx*sx + sy*sy + sz*sz);
            if (mag < 1e-12) {
                std::cout << "     [!] Zero vector - please re-enter.\n";
                continue;
            }
            break;
        }
        std::cout << "     [!] Could not parse three numbers - please re-enter.\n";
    }

    // ── b) Number of reflections ─────────────────────────────────────────────
    int nrefl = 0;
    while (true) {
        std::cout << "  b) Number of reflections (0, 1, 2, ...): ";
        std::string ln;
        if (!std::getline(std::cin, ln)) return;
        try {
            int v = std::stoi(ln);
            if (v < 0) { std::cout << "     [!] Must be >= 0.\n"; continue; }
            nrefl = v;
            break;
        } catch (...) {
            std::cout << "     [!] Enter a non-negative integer.\n";
        }
    }

    // ── c) Visualization ─────────────────────────────────────────────────────
    bool do_viz = false;
    while (true) {
        std::cout << "  c) Visualization needed? (Y / N): ";
        std::string ln;
        if (!std::getline(std::cin, ln)) return;
        // trim
        while (!ln.empty() && std::isspace(static_cast<unsigned char>(ln.front()))) ln.erase(ln.begin());
        while (!ln.empty() && std::isspace(static_cast<unsigned char>(ln.back())))  ln.pop_back();
        if (ln.empty()) continue;
        char c = static_cast<char>(std::tolower(static_cast<unsigned char>(ln[0])));
        if (c == 'y') { do_viz = true;  break; }
        if (c == 'n') { do_viz = false; break; }
        std::cout << "     [!] Enter Y or N.\n";
    }

    // ── d) Method selection ──────────────────────────────────────────────────
    bool rtx_ok = is_rtx_device_available();
    bool gpu_ok = is_cuda_device_available();

    std::cout << "\n";
    std::cout << "  d) Calculation method:\n";
    std::cout << "       A  Auto-select fastest available  (Centroid family)\n";
    std::cout << "       --- Centroid (one shadow ray per triangle centroid) ---\n";
    std::cout << "       1  CentroidRTX     (OptiX RT Cores)  "
              << (rtx_ok ? "[available]"
                         : "[unavailable - " + rtx_unavailable_reason() + "]") << "\n";
    std::cout << "       2  CentroidGPU     (CUDA cores)      "
              << (gpu_ok ? "[available]" : "[unavailable - no CUDA GPU]") << "\n";
    std::cout << "       3  CentroidCPU     (CPU)             [always available]\n";
    std::cout << "       --- Pixel-grid (parallel rays on a 2-D grid) ---\n";
    std::cout << "       4  PixelGridRTX    (OptiX RT Cores)  "
              << (rtx_ok ? "[available]"
                         : "[unavailable - " + rtx_unavailable_reason() + "]") << "\n";
    std::cout << "       5  PixelGridGPU    (CUDA cores)      "
              << (gpu_ok ? "[available]" : "[unavailable - no CUDA GPU]") << "\n";
    std::cout << "       6  PixelGridCPU    (CPU)             [always available]\n";

    SRPMethod   method       = SRPMethod::CentroidCPU;
    const char* method_label = "CentroidCPU (CPU)";
    char        mchoice      = 'a';

    while (true) {
        std::cout << "     Choice (A / 1..6): ";
        std::string ln;
        if (!std::getline(std::cin, ln)) return;
        while (!ln.empty() && std::isspace(static_cast<unsigned char>(ln.front()))) ln.erase(ln.begin());
        while (!ln.empty() && std::isspace(static_cast<unsigned char>(ln.back())))  ln.pop_back();
        if (ln.empty()) continue;
        mchoice = static_cast<char>(std::tolower(static_cast<unsigned char>(ln[0])));
        if (mchoice == 'a' || (mchoice >= '1' && mchoice <= '6')) break;
        std::cout << "     [!] Enter A, or a digit from 1 to 6.\n";
    }

    // Resolve auto → fastest available centroid method
    if (mchoice == 'a') {
        if      (rtx_ok) { mchoice = '1'; std::cout << "  [auto] RTX available - CentroidRTX selected.\n"; }
        else if (gpu_ok) { mchoice = '2'; std::cout << "  [auto] RTX not available - CentroidGPU selected.\n"; }
        else             { mchoice = '3'; std::cout << "  [auto] No GPU found - CentroidCPU fallback selected.\n"; }
    } else {
        // Manual pick - fall back gracefully if hardware not present
        if ((mchoice == '1' || mchoice == '4') && !rtx_ok) {
            std::cout << "  [!] RTX not available: " << rtx_unavailable_reason() << "\n";
            // 1->2 (Centroid RTX -> GPU), 4->5 (PixelGrid RTX -> GPU), then to CPU
            char alt_gpu = (mchoice == '1') ? '2' : '5';
            char alt_cpu = (mchoice == '1') ? '3' : '6';
            mchoice = gpu_ok ? alt_gpu : alt_cpu;
            std::cout << "  Falling back to " << (gpu_ok ? "GPU CUDA" : "CPU") << " method.\n";
        } else if ((mchoice == '2' || mchoice == '5') && !gpu_ok) {
            std::cout << "  [!] No CUDA GPU found - falling back to CPU method.\n";
            mchoice = (mchoice == '2') ? '3' : '6';
        }
    }

    switch (mchoice) {
    case '1': method = SRPMethod::CentroidRTX;
              method_label = "CentroidRTX (OptiX RT Cores)";  break;
    case '2': method = SRPMethod::CentroidGPU;
              method_label = "CentroidGPU (CUDA cores)";      break;
    case '3': method = SRPMethod::CentroidCPU;
              method_label = "CentroidCPU (CPU)";             break;
    case '4': method = SRPMethod::PixelGridRTX;
              method_label = "PixelGridRTX (OptiX RT Cores)"; break;
    case '5': method = SRPMethod::PixelGridGPU;
              method_label = "PixelGridGPU (CUDA cores)";     break;
    default:  method = SRPMethod::PixelGridCPU;
              method_label = "PixelGridCPU (CPU)";            break;
    }

    // ── d2) Grid step (only for PixelGrid methods) ───────────────────────────
    const bool is_pixel = (mchoice == '4' || mchoice == '5' || mchoice == '6');
    if (is_pixel) {
        std::cout << "  d2) Grid step in metres (current = "
                  << engine.getGridStep() << "; press Enter to keep): ";
        std::string ln;
        if (!std::getline(std::cin, ln)) return;
        while (!ln.empty() && std::isspace(static_cast<unsigned char>(ln.front()))) ln.erase(ln.begin());
        while (!ln.empty() && std::isspace(static_cast<unsigned char>(ln.back())))  ln.pop_back();
        if (!ln.empty()) {
            try {
                double gs = std::stod(ln);
                if (gs > 0.0) engine.setGridStep(gs);
                else std::cout << "     [!] Must be > 0, keeping current.\n";
            } catch (...) {
                std::cout << "     [!] Could not parse a number, keeping current.\n";
            }
        }
    }

    // Note: viz path uses bounce-tracking; bench path skips it for speed.
    std::cout << "  Method     : " << method_label << "\n";
    std::cout << "  Mode       : " << (do_viz ? "viz (non-bench, bounce tracking)" : "bench (fast, no bounce tracking)") << "\n";
    std::cout << "  Sun vector : (" << std::fixed << std::setprecision(6)
              << sx << ", " << sy << ", " << sz << ")\n";
    std::cout << "  Reflections: " << nrefl << "\n";
    if (is_pixel) std::cout << "  Grid step  : " << engine.getGridStep() << " m\n";
    std::cout << "\n";

    // ── Compute ──────────────────────────────────────────────────────────────
    // Do NOT reload the dataset here - it is already in memory from startup
    // (or from the last section that ran).  Reloading an HDF5 with 748k
    // triangles takes several seconds and is pure wasted I/O for this path.
    engine.setSunDirection(sx, sy, sz);
    engine.setMaxReflections(nrefl);

    // Time-based warm-up for GPU/RTX methods. Without this the very first
    // call shows GPU P-state ramp-up cost (~120-250 ms cold-start) and any
    // call following an idle period of 1+ s reads at base clock (~13-15 ms).
    // The warm-up loop sustains GPU work for ≥500 ms before the timed call,
    // guaranteeing steady boost-state clocks - what an orbit integrator
    // actually sees in continuous use. CPU methods get no warm-up: no GPU
    // P-state to ramp.
    const bool is_gpu_method =
        (method == SRPMethod::CentroidGPU || method == SRPMethod::CentroidRTX ||
         method == SRPMethod::PixelGridGPU || method == SRPMethod::PixelGridRTX);
    if (is_gpu_method) {
        if (g_warmup_enabled) {
            std::cout << "  [warm-up: >= " << (int)g_warmup_min_ms << " ms / >= "
                      << g_warmup_min_calls << " calls] ...\n";
            auto   tw0    = std::chrono::high_resolution_clock::now();
            int    wcalls = 0;
            double wms    = 0.0;
            do {
                if (do_viz) engine.computeViz(method);
                else        engine.compute(method);
                ++wcalls;
                wms = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::high_resolution_clock::now() - tw0).count() / 1000.0;
            } while (wcalls < g_warmup_min_calls || wms < g_warmup_min_ms);
        } else {
            std::cout << "  [warm-up disabled — cold-start measurement]\n";
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    SRPResult res;
    if (do_viz)
        res = engine.computeViz(method);
    else
        res = engine.compute(method);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    // ── Results ───────────────────────────────────────────────────────────────
    printResult(res, ms);

    // ── Visualize ─────────────────────────────────────────────────────────────
    if (do_viz) {
        try {
            engine.visualizeLastResult(1, false);
        } catch (const std::exception& e) {
            std::cerr << "  Visualisation skipped: " << e.what() << "\n";
            std::cerr << "  (Install Python + plotly/matplotlib to enable.)\n";
        }
    }
}

// =============================================================================
// Menu
// =============================================================================

using SectionFn = void(*)(SRPEngine&);

struct SectionEntry {
    int         number;
    const char* title;
    SectionFn   fn;
};

static const SectionEntry kSections[] = {
    {  1, "Dataset construction",                    section1_dataset_construction },
    {  2, "Geometry inspection",                     section2_geometry_inspection  },
    {  3, "Loading files",                           section3_loading_files        },
    {  4, "Revolute joints",                         section4_revolute_joints      },
    {  5, "Prismatic joints (translation)",          section5_prismatic_joints     },
    {  6, "Multiple joints in one reload",           section6_multiple_joints      },
    {  7, "Legacy load_single_file",                 section7_legacy_load          },
    {  8, "Display helpers",                         section8_display_helpers      },
    {  9, "Engine parameters (sun / reflections)",   section9_engine_parameters    },
    { 10, "SRP computation methods",                 section10_srp_computation     },
    { 11, "Labels read-back after compute",          section11_labels_readback     },
    { 12, "Articulate, recompute, compare",          section12_articulate_recompute},
    { 13, "Sun azimuth sweep",                       section13_azimuth_sweep       },
    { 14, "Visualisation (RTX, no reflections)",     section14_visualize           },
    { 15, "SRP visualization sweep (0..3 refl.)",    section15_viz_sweep           },
    { 16, "Hardware info (GPU / CUDA / OptiX)",      section16_hardware_info       },
    { 17, "Animation (moving sun and parts)",        section17_animation           },
};
static constexpr int kNumSections = static_cast<int>(sizeof(kSections) / sizeof(kSections[0]));

static void printMenu()
{
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  MENU - SRP Library Demo\n";
    std::cout << "================================================================\n";
    for (int i = 0; i < kNumSections; ++i)
        std::cout << "  " << std::setw(3) << kSections[i].number
                  << ".  " << kSections[i].title << "\n";
    std::cout << "\n";
    std::cout << "----------------------------------------------------------------\n";
    std::cout << "   C.  Main calculation  (enter sun vector → auto method → F, M)\n";
    std::cout << "   P.  Set phi0 (current: " << std::scientific << std::setprecision(4)
              << g_srp_phi0 << std::defaultfloat << " N/m²)  [4.56e-6 at 1 AU]\n";
    std::cout << "   G.  GPU clock display: " << (g_show_gpu_clock ? "ON " : "OFF")
              << "  (NVML SM frequency at warm-up + result)\n";
    std::cout << "   W.  Warm-up: " << (g_warmup_enabled ? "ON " : "OFF");
    if (g_warmup_enabled)
        std::cout << "  | min calls: " << std::setw(3) << g_warmup_min_calls
                  << "  | min time: " << (int)g_warmup_min_ms << " ms";
    else
        std::cout << "  (cold-start mode — GPU stays at P8)";
    std::cout << "\n";
    std::cout << "----------------------------------------------------------------\n";
    std::cout << "   A.  Run all sections in order\n";
    std::cout << "   Q.  Quit\n";
    std::cout << "----------------------------------------------------------------\n";
    std::cout << "  Choice: ";
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char* argv[])
{
    try {
        std::string folder_path = "data3d_hdf5_0.5";
        if (argc >= 2) {
            folder_path = argv[1];
            if (!fs::exists(folder_path) || !fs::is_directory(folder_path))
                throw std::runtime_error("Folder not found: " + folder_path);
        } else {
            if (!fs::exists(folder_path) || !fs::is_directory(folder_path))
                throw std::runtime_error("Default folder '" + folder_path + "' does not exist.");
        }
        fs::create_directories("results_3d");

        std::cout << "Loading dataset from: " << folder_path << "\n";
        SRPEngine engine(folder_path);
        std::cout << "Loaded " << engine.dataset().triangleCount()
                  << " triangles from "
                  << fs::path(engine.dataset().currentFile()).filename().string() << "\n";

        std::string line;
        while (true) {
            printMenu();
            if (!std::getline(std::cin, line)) break;   // EOF

            // Trim whitespace
            while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
                line.erase(line.begin());
            while (!line.empty() && (line.back() == ' '  || line.back() == '\r' ||
                                     line.back() == '\n' || line.back() == '\t'))
                line.pop_back();

            if (line.empty()) continue;

            char first = static_cast<char>(std::tolower(static_cast<unsigned char>(line[0])));

            if (first == 'q') {
                std::cout << "\n  Bye.\n";
                break;
            }

            if (first == 'a') {
                for (int i = 0; i < kNumSections; ++i) {
                    try {
                        kSections[i].fn(engine);
                    } catch (const std::exception& e) {
                        std::cerr << "\n  Section " << kSections[i].number
                                  << " error: " << e.what() << "\n";
                    }
                }
                std::cout << "\n====== All sections completed. ======\n";
                continue;
            }

            if (first == 'c') {
                try {
                    main_calculation(engine);
                } catch (const std::exception& e) {
                    std::cerr << "\n  Calculation error: " << e.what() << "\n";
                }
                continue;
            }

            if (first == 'g') {
                g_show_gpu_clock = !g_show_gpu_clock;
                std::cout << "  GPU clock display: " << (g_show_gpu_clock ? "ON" : "OFF") << "\n";
                continue;
            }

            if (first == 'w') {
                // Helper: trim whitespace from both ends
                auto trim = [](std::string& s) {
                    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
                    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))  s.pop_back();
                };

                std::cout << "  Warm-up settings  (current: "
                          << (g_warmup_enabled ? "ON" : "OFF")
                          << ", min calls=" << g_warmup_min_calls
                          << ", min time=" << (int)g_warmup_min_ms << " ms)\n";

                // Enable / disable
                {
                    std::cout << "  Enable warm-up? (Y/N) [" << (g_warmup_enabled ? "Y" : "N") << "]: ";
                    std::string ln;
                    if (std::getline(std::cin, ln)) {
                        trim(ln);
                        if (!ln.empty()) {
                            char c = static_cast<char>(std::tolower(static_cast<unsigned char>(ln[0])));
                            if (c == 'y') g_warmup_enabled = true;
                            if (c == 'n') g_warmup_enabled = false;
                        }
                    }
                }

                if (g_warmup_enabled) {
                    // Min calls
                    {
                        std::cout << "  Min calls (>= 1) [" << g_warmup_min_calls << "]: ";
                        std::string ln;
                        if (std::getline(std::cin, ln)) {
                            trim(ln);
                            if (!ln.empty()) {
                                try {
                                    int v = std::stoi(ln);
                                    if (v >= 1) g_warmup_min_calls = v;
                                    else std::cout << "  [!] Must be >= 1, keeping " << g_warmup_min_calls << ".\n";
                                } catch (...) {
                                    std::cout << "  [!] Invalid, keeping " << g_warmup_min_calls << ".\n";
                                }
                            }
                        }
                    }
                    // Min time
                    {
                        std::cout << "  Min time ms (>= 0) [" << (int)g_warmup_min_ms << "]: ";
                        std::string ln;
                        if (std::getline(std::cin, ln)) {
                            trim(ln);
                            if (!ln.empty()) {
                                try {
                                    double v = std::stod(ln);
                                    if (v >= 0.0) g_warmup_min_ms = v;
                                    else std::cout << "  [!] Must be >= 0, keeping " << (int)g_warmup_min_ms << ".\n";
                                } catch (...) {
                                    std::cout << "  [!] Invalid, keeping " << (int)g_warmup_min_ms << ".\n";
                                }
                            }
                        }
                    }
                }

                std::cout << "  Warm-up: " << (g_warmup_enabled ? "ON" : "OFF");
                if (g_warmup_enabled)
                    std::cout << "  | min calls: " << g_warmup_min_calls
                              << "  | min time: " << (int)g_warmup_min_ms << " ms";
                std::cout << "\n";
                continue;
            }

            if (first == 'p') {
                std::cout << "  Solar radiation pressure constant phi0 [N/m²].\n"
                          << "    1 AU (Earth orbit) ≈ 4.56e-6 N/m²\n"
                          << "    Use 1.0 for dimensionless / normalised output.\n"
                          << "  Current: " << std::scientific << std::setprecision(4)
                          << g_srp_phi0 << std::defaultfloat << " N/m²\n"
                          << "  New phi0: ";
                std::string phi0_line;
                if (std::getline(std::cin, phi0_line)) {
                    try {
                        double new_phi0 = std::stod(phi0_line);
                        if (new_phi0 <= 0.0) {
                            std::cout << "  phi0 must be positive. Keeping "
                                      << std::scientific << std::setprecision(4)
                                      << g_srp_phi0 << std::defaultfloat << "\n";
                        } else {
                            g_srp_phi0 = new_phi0;
                            std::cout << "  phi0 set to " << std::scientific
                                      << std::setprecision(4) << g_srp_phi0
                                      << std::defaultfloat << " N/m²\n";
                        }
                    } catch (...) {
                        std::cout << "  Invalid value. Keeping current phi0.\n";
                    }
                }
                continue;
            }

            // Parse number
            try {
                int choice = std::stoi(line);
                bool found = false;
                for (int i = 0; i < kNumSections; ++i) {
                    if (kSections[i].number == choice) {
                        try {
                            kSections[i].fn(engine);
                        } catch (const std::exception& e) {
                            std::cerr << "\n  Section error: " << e.what() << "\n";
                        }
                        found = true;
                        break;
                    }
                }
                if (!found)
                    std::cout << "  Unknown section: " << choice
                              << "  (enter 1-" << kNumSections << ", A, or Q)\n";
            } catch (const std::invalid_argument&) {
                std::cout << "  Invalid input. Enter a section number, A, or Q.\n";
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
