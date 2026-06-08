#pragma once

// =============================================================================
// SRPLibrary.h  -  single public header for the SM3D SRP computation library.
//
// The user only needs this one file.  All types used in the public API are
// defined here directly; no additional headers need to be distributed.
//
// Typical usage:
//   #include "SRPLibrary.h"
//
//   SRPEngine engine("path/to/h5_files");
//   engine.setSunDirection(1, 0, 0);
//   engine.dataset().articulate("SolarPanel_L", JointConfig::rotY(M_PI/4));
//
//   // Centroid method (default, one shadow ray per triangle):
//   SRPResult r = engine.compute();                            // CentroidRTX (default)
//   SRPResult r = engine.compute(SRPMethod::CentroidRTX);     // explicit
//
//   // Pixel-grid method (parallel rays on a 2-D grid perpendicular to the sun):
//   engine.setGridStep(0.05);                                  // cell size [m]
//   SRPResult r = engine.compute(SRPMethod::PixelGridRTX);
// =============================================================================

#include <vector>
#include <string>
#include <array>
#include <map>
#include <memory>
#include <iosfwd>
#include <iostream>
#include <filesystem>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Triangle - flat-shaded triangle in body frame with optical properties
// =============================================================================
struct Triangle {
    std::string ID;
    double v1_x, v1_y, v1_z;            // Vertex 1 (body frame, metres)
    double v2_x, v2_y, v2_z;            // Vertex 2
    double v3_x, v3_y, v3_z;            // Vertex 3
    double normal_x, normal_y, normal_z; // Outward unit normal (body frame)
    std::string component_type;          // HDF5 component_type attribute
    std::string component_name;          // HDF5 group name  e.g. "S1_Mirror"
    double label = 1.0;                  // 1 = illuminated, 0 = shadowed

    // Optical surface properties (per SatForm3D physical model):
    //   reflectance rho in [0,1]  - fraction of incident flux reflected
    //   specularity s   in [0,1]  - of reflected, fraction that is specular
    //   emissivity  eps in [0,1]  - thermal emission efficiency
    //   SRP: specular = rho*s,  diffuse = rho*(1-s),  absorbed = 1-rho
    double reflectance = 0.0;
    double specularity = 0.0;
    double emissivity  = 0.0;

    double area = 0.0;  // m^2 - invariant under rigid transforms
    double centroid_x = 0.0, centroid_y = 0.0, centroid_z = 0.0;  // body frame
};

// =============================================================================
// SRPResult - output of any compute() call
// =============================================================================
struct SRPResult {
    std::vector<int>       labels;        // per-triangle: 1 = illuminated, 0 = shadowed
    std::array<double, 3>  total_force;   // total SRP force  [Fx, Fy, Fz]  (N)
    std::array<double, 3>  total_moment;  // total SRP moment [Mx, My, Mz]  (N*m)
};

// =============================================================================
// Solar radiation pressure scale factor (phi0) [N/m²].
// Default: 4.56e-6 N/m² (approximate value at 1 AU).
// Set this before calling any compute method to get forces in SI units.
// All SRP force/moment results are multiplied by this value.
// Use 1.0 for dimensionless / normalised output.
// =============================================================================
extern double g_srp_phi0;

// =============================================================================
// JointConfig - describes one articulation applied to a named component.
//
// REVOLUTE  - rotates the component around a given axis.
// PRISMATIC - translates the component along a given axis.
//
// Pivot point (REVOLUTE only):
//   use_auto_pivot = true  (default) -> pivot is the mesh centroid in local
//     frame.  Correct for test/snapshot components whose vertices are stored
//     in body frame and T_local_offset = Identity.
//   use_auto_pivot = false -> pivot is (px, py, pz) in local frame.
//     Use .atOrigin() or .atPivot(x,y,z) to set it.
//     Use (0,0,0) for properly articulated components (solar panels, booms)
//     whose mesh vertices are already centred at the joint origin.
//
// Quick-create factories:
//   JointConfig::rotZ(angle_rad)
//   JointConfig::rotX(angle_rad)
//   JointConfig::rotY(angle_rad)
//   JointConfig::rot(ax, ay, az, angle_rad)
//   JointConfig::translateZ(dist_m)
//   JointConfig::translateX(dist_m)
//   JointConfig::translateY(dist_m)
//   JointConfig::translate(ax, ay, az, dist_m)
//
// Pivot modifiers (return modified copy):
//   .atOrigin()           -> pivot = (0,0,0) - for hinge-jointed components
//   .atPivot(x, y, z)    -> explicit pivot in local frame
// =============================================================================
struct JointConfig {
    enum class Type { REVOLUTE, PRISMATIC };

    Type   type  = Type::REVOLUTE;
    double ax = 0.0, ay = 0.0, az = 1.0; // axis (normalised internally)
    double value = 0.0;                   // rad (REVOLUTE) or m (PRISMATIC)

    bool   use_auto_pivot = true;
    double px = 0.0, py = 0.0, pz = 0.0; // explicit pivot (local frame)

    // ---- Static factories -----------------------------------------------
    static JointConfig rotZ(double angle_rad)
        { return {Type::REVOLUTE, 0,0,1, angle_rad}; }
    static JointConfig rotX(double angle_rad)
        { return {Type::REVOLUTE, 1,0,0, angle_rad}; }
    static JointConfig rotY(double angle_rad)
        { return {Type::REVOLUTE, 0,1,0, angle_rad}; }
    static JointConfig rot(double ax_, double ay_, double az_, double angle_rad)
        { return {Type::REVOLUTE, ax_,ay_,az_, angle_rad}; }
    static JointConfig translateZ(double dist_m)
        { return {Type::PRISMATIC, 0,0,1, dist_m}; }
    static JointConfig translateX(double dist_m)
        { return {Type::PRISMATIC, 1,0,0, dist_m}; }
    static JointConfig translateY(double dist_m)
        { return {Type::PRISMATIC, 0,1,0, dist_m}; }
    static JointConfig translate(double ax_, double ay_, double az_, double dist_m)
        { return {Type::PRISMATIC, ax_,ay_,az_, dist_m}; }

    // ---- Pivot modifiers (return copy) ----------------------------------
    JointConfig atOrigin() const {
        JointConfig c = *this;
        c.use_auto_pivot = false;
        c.px = c.py = c.pz = 0.0;
        return c;
    }
    JointConfig atPivot(double x, double y, double z) const {
        JointConfig c = *this;
        c.use_auto_pivot = false;
        c.px = x; c.py = y; c.pz = z;
        return c;
    }
};

// =============================================================================
// SatelliteDataset - HDF5 satellite geometry loader with scene-graph
// transforms and general articulation support.
//
// Each component in the HDF5 file has a T_local_offset matrix (its static
// pose in the parent frame) and optionally joint parameters.  The loader
// builds body-frame vertex positions via the chain:
//
//   T_body = T_body(parent) * T_local_offset * T_joint(value)
//
// Typical workflow:
//   SatelliteDataset ds;
//   ds.setFolder("path/to/h5_files");
//   ds.setJoint("S2_Bounce1", JointConfig::rotZ(M_PI/2));
//   ds.load(0);
//   ds.printComponents();
//   ds.articulate("S1_Mirror", JointConfig::rotX(0.5));  // setJoint + reload
//   const auto& tris = ds.triangles();
// =============================================================================
class SatelliteDataset {
public:
    // ---- Construction ---------------------------------------------------
    SatelliteDataset() = default;
    explicit SatelliteDataset(const std::string& folder_path);

    // ---- Folder / file list ---------------------------------------------
    void        setFolder(const std::string& folder_path);
    int         fileCount()  const;
    const std::vector<std::string>& fileNames() const;
    std::string currentFile() const;

    // ---- Loading --------------------------------------------------------
    void load(const std::string& file_path);
    void load(int index);
    void reload();

    // ---- Triangle access ------------------------------------------------
    const std::vector<Triangle>& triangles()     const;
    int                          triangleCount() const;

    // ---- Component inspection -------------------------------------------
    struct ComponentInfo {
        std::string name;
        std::string type;
        int         triangle_count;
        double      total_area;     // m^2
    };
    std::vector<std::string>     componentNames()  const;
    std::vector<ComponentInfo>   componentInfos()  const;
    bool                         hasComponent(const std::string& name) const;
    std::vector<const Triangle*> componentTriangles(const std::string& name) const;

    // ---- Articulation - general joints ----------------------------------
    void        setJoint(const std::string& component_name, const JointConfig& jc);
    void        clearJoint(const std::string& component_name);
    void        clearAllJoints();
    JointConfig getJoint(const std::string& component_name) const;
    bool        hasJoint(const std::string& component_name) const;
    const std::map<std::string, JointConfig>& joints() const;
    void        articulate(const std::string& component_name, const JointConfig& jc);

    // ---- Display helpers ------------------------------------------------
    void printInfo(std::ostream& out = std::cout)       const;
    void printComponents(std::ostream& out = std::cout) const;
    void printJoints(std::ostream& out = std::cout)     const;

    // ---- Legacy interface (backward-compatible) -------------------------
    void   setJointZ(const std::string& name, double angle_rad);
    void   rotateZ  (const std::string& name, double angle_rad);
    double getJointZ(const std::string& name) const;
    void   load_single_file(const std::string& file_path,
                            const std::map<std::string, double>& z_joint_overrides = {});
    const std::vector<Triangle>&    get_triangles()  const;
    const std::vector<std::string>& get_file_names() const;

private:
    using Mat4 = std::array<double, 16>;

    // ---------------------------------------------------------------------
    // CachedNode: a single component from the HDF5 scene graph, parsed once
    // and held in memory. reload() / articulate() rebuild the triangle list
    // from this cache WITHOUT re-reading the HDF5 file.
    // ---------------------------------------------------------------------
    struct CachedNode {
        std::string                       name;             // group name (e.g. "S1_Mirror")
        std::string                       component_type;   // HDF5 attribute
        Mat4                              T_local_offset{}; // static pose in parent frame
        int                               parent_index = -1;// index into cached_nodes_ (-1 = root)

        bool                              has_mesh = false;
        std::vector<std::vector<double>>  vertices_local;   // raw vertices in component local frame
        std::vector<std::vector<double>>  normals_local;    // per-triangle local-frame normals
        std::vector<std::vector<int>>     tri_indices;      // triangle vertex indices
        std::vector<double>               areas;
        std::vector<double>               reflectance;
        std::vector<double>               specularity;
        std::vector<double>               emissivity;
    };

    std::vector<Triangle>              triangles_;
    std::vector<std::string>           file_names_;
    std::string                        current_file_;
    std::map<std::string, JointConfig> joints_;

    // Parsed scene graph (filled by load_h5_into_cache, consumed by rebuildTriangles).
    std::vector<CachedNode>            cached_nodes_;

    void scanFolder(const std::string& folder_path);

    static Mat4 identity4();
    static Mat4 mul4(const Mat4& A, const Mat4& B);
    static Mat4 makeTrans(double tx, double ty, double tz);
    static Mat4 makeRotAxis(double ax, double ay, double az, double angle_rad);
    static Mat4 makeRotAxisAroundPoint(double ax, double ay, double az,
                                       double angle_rad,
                                       double cx, double cy, double cz);
    static void transformPoint (const Mat4& T, double ix, double iy, double iz,
                                double& ox, double& oy, double& oz);
    static void transformNormal(const Mat4& T, double ix, double iy, double iz,
                                double& ox, double& oy, double& oz);
    static Mat4 buildJointTransform(
        const JointConfig& jc,
        const std::vector<std::vector<double>>& vertices);

    // Read HDF5 file into cached_nodes_ (slow, ~once per file).
    void load_h5_into_cache(const std::string& file_path);

    // Rebuild triangles_ from cached_nodes_ applying current joints_.
    // Fast: pure matrix math, no file I/O. Called by load() and reload().
    void rebuildTriangles(const std::map<std::string, JointConfig>& joint_overrides);
};

// =============================================================================
// SRPMethod - selects the shadow/SRP computation algorithm.
//
// Two algorithm families:
//   * Centroid  - one shadow ray per triangle centroid (fast, ~1 ray/triangle).
//   * PixelGrid - parallel rays on a 2-D grid perpendicular to the sun
//                 (accurate on small/grazing facets; cost scales with grid_step).
//
// =============================================================================
// Hardware detection helpers (defined in SM3D_hw_info.cpp, linked into libsrp)
// =============================================================================

/// Returns true if at least one CUDA-capable GPU is present and functional.
bool is_cuda_device_available();

/// Returns true if at least one OptiX-capable (RT-Core) GPU is present.
/// Requires SM_7.5+ (Turing / Ampere / Ada) and OptiX initialisation to succeed.
bool is_rtx_device_available();

/// Returns a human-readable explanation why RTX is unavailable, or "" if it is.
std::string rtx_unavailable_reason();

/// Prints a table of CUDA/OptiX device properties to stdout.
void print_hardware_info();

// =============================================================================
// Each family has three backends:
//   * CPU - portable, double precision, no GPU required.
//   * GPU - CUDA cores, double precision accumulators (no OptiX required).
//   * RTX - OptiX hardware ray tracing on RT Cores (fastest on RTX GPUs).
//
// Recommended on an RTX-capable NVIDIA GPU: CentroidRTX (fastest) or
// PixelGridRTX (most accurate). Fall back to *GPU if OptiX is unavailable,
// or *CPU if no GPU is present at all.
// =============================================================================
enum class SRPMethod {
    CentroidCPU,    // CPU  centroid ray casting (+ multi-bounce reflections)
    CentroidGPU,    // CUDA centroid ray casting (+ multi-bounce reflections)
    CentroidRTX,    // RTX  centroid ray casting (+ multi-bounce reflections)  [recommended]
    PixelGridCPU,   // CPU  parallel-ray pixel grid (+ multi-bounce reflections)
    PixelGridGPU,   // CUDA parallel-ray pixel grid (+ multi-bounce reflections)
    PixelGridRTX    // RTX  parallel-ray pixel grid (+ multi-bounce reflections)
};

// =============================================================================
// SRPEngine - main entry point for SRP calculations.
//
// The engine owns a SatelliteDataset that can be freely accessed and modified
// between compute() calls.  Any geometry change (articulation, file switch)
// takes effect on the next compute() call automatically.
//
// Typical usage:
//   SRPEngine engine("path/to/h5_files");   // scans folder, loads file[0]
//   engine.setSunDirection(1, 0, 0);
//   engine.dataset().articulate("SolarPanel", JointConfig::rotY(M_PI/4));
//   SRPResult r = engine.compute(SRPMethod::CentroidRTX);
//   engine.visualizeLastResult();
// =============================================================================
class SRPEngine {
public:
    // Constructor: scans folder_path for .h5 files and loads the first one.
    explicit SRPEngine(const std::string& folder_path);

    ~SRPEngine();
    SRPEngine(const SRPEngine&) = delete;
    SRPEngine& operator=(const SRPEngine&) = delete;
    SRPEngine(SRPEngine&&) noexcept;
    SRPEngine& operator=(SRPEngine&&) noexcept;

    // -------------------------------------------------------------------------
    // Dataset access - full SatelliteDataset API available through this ref.
    //
    // Load / switch files:
    //   engine.dataset().load(0)                - load by index
    //   engine.dataset().load("file.h5")        - load by path
    //   engine.dataset().reload()               - reload, re-apply joints
    //   engine.dataset().fileCount()
    //   engine.dataset().fileNames()
    //   engine.dataset().currentFile()
    //
    // Inspect geometry:
    //   engine.dataset().triangles()            - full triangle list
    //   engine.dataset().triangleCount()
    //   engine.dataset().componentNames()
    //   engine.dataset().componentInfos()       - name, type, tri count, area
    //   engine.dataset().componentTriangles("Name")
    //   engine.dataset().hasComponent("Name")
    //   engine.dataset().printInfo()
    //   engine.dataset().printComponents()
    //   engine.dataset().printJoints()
    //
    // Articulate (revolute joints - rotation):
    //   engine.dataset().setJoint("Name", JointConfig::rotZ(angle_rad))
    //   engine.dataset().setJoint("Name", JointConfig::rotX(angle_rad))
    //   engine.dataset().setJoint("Name", JointConfig::rotY(angle_rad))
    //   engine.dataset().setJoint("Name", JointConfig::rot(ax,ay,az,angle_rad))
    //   engine.dataset().reload()               - commit staged joints
    //   engine.dataset().articulate("Name", JointConfig::rotZ(angle_rad))
    //                                           - setJoint + reload in one call
    //
    // Articulate (prismatic joints - translation):
    //   engine.dataset().setJoint("Name", JointConfig::translateZ(dist_m))
    //   engine.dataset().setJoint("Name", JointConfig::translateX(dist_m))
    //   engine.dataset().setJoint("Name", JointConfig::translateY(dist_m))
    //   engine.dataset().setJoint("Name", JointConfig::translate(ax,ay,az,dist_m))
    //
    // Pivot control (REVOLUTE only, append to any factory):
    //   JointConfig::rotZ(a).atOrigin()         - pivot at (0,0,0)
    //   JointConfig::rotZ(a).atPivot(x,y,z)    - explicit pivot
    //   (default: pivot = auto centroid of component mesh)
    //
    // Joint management:
    //   engine.dataset().clearJoint("Name")
    //   engine.dataset().clearAllJoints()
    //   engine.dataset().hasJoint("Name")
    //   engine.dataset().getJoint("Name")       - returns JointConfig
    //   engine.dataset().joints()               - map<name, JointConfig>
    //
    // Legacy interface (still supported):
    //   engine.dataset().setJointZ("Name", angle_rad)
    //   engine.dataset().rotateZ("Name", angle_rad)   - setJointZ + reload
    //   engine.dataset().getJointZ("Name")
    //   engine.dataset().load_single_file(path, {{"Name", angle_rad}})
    //   engine.dataset().get_triangles()
    //   engine.dataset().get_file_names()
    // -------------------------------------------------------------------------
          SatelliteDataset& dataset();
    const SatelliteDataset& dataset() const;

    // -------------------------------------------------------------------------
    // Calculation parameters
    // -------------------------------------------------------------------------
    void setMaxReflections(int bounces);  // inter-reflection bounces for all methods (default 0)

    // Pixel-grid cell size in metres. Used only by PixelGrid* methods (ignored by Centroid*).
    // Smaller step  -> higher accuracy and more rays (cost ~ 1/step^2).
    // Default: 0.05 m. Must be > 0.
    void   setGridStep(double step);
    double getGridStep() const;

    // Sun direction - normalised automatically; can be called at any time.
    void setSunDirection(double x, double y, double z);

    // Run the chosen method using the current dataset geometry and sun direction.
    // Fast path: GPU/RTX branches use the lean "bench" kernels (no bounce tracking).
    SRPResult compute(SRPMethod method = SRPMethod::CentroidRTX);

    // Run the visualization path - same algorithm family, but with full bounce tracking.
    // After this call, getBounceLevels() / getIncidentDirs() / getOriginPts() are valid.
    SRPResult computeViz(SRPMethod method = SRPMethod::CentroidRTX);

    // Shadow labels from the last compute() call (no copy - reference into engine).
    const std::vector<int>& getLabels() const;

    // Per-triangle bounce level from the last computeViz() call.
    // -1 = never hit, 0 = directly illuminated, 1+ = reflection bounce level.
    const std::vector<int>& getBounceLevels() const;

    // Per-triangle incident direction from the last computeViz() call.
    const std::vector<std::array<double,3>>& getIncidentDirs() const;

    // Per-triangle ray origin from the last computeViz() call.
    // NaN for directly illuminated triangles.
    const std::vector<std::array<double,3>>& getOriginPts() const;

    // Launch Python visualisation of the last result.
    //   every        - render every N-th triangle (1 = all triangles)
    //   show_normals - overlay surface normal arrows
    // Requires Python with matplotlib/plotly in PATH.
    void visualizeLastResult(int every = 1, bool show_normals = false) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
