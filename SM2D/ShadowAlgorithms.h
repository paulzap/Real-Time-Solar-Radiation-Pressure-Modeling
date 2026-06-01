#pragma once
// ---------------------------------------------------------------------------
// ShadowAlgorithms.h -- internal header for the SRP library.
// SRPResult and Triangle are defined in SRPLibrary.h (the single public header).
// SatelliteDataset.h is a thin shim that re-exports SRPLibrary.h, kept here
// for compatibility with verbatim CC12 source files (.cpp / .cu) that include it.
// ---------------------------------------------------------------------------
#include <vector>
#include <string>
#include <array>
#include "SatelliteDataset.h"


// Structure for AABB (Axis-Aligned Bounding Box)
struct AABB {
    double min_x, min_y, min_z;
    double max_x, max_y, max_z;
};

// BVH Node structure
struct BVHNode {
    AABB bounds;
    size_t triangle_idx; // Triangle index if leaf node
    size_t left, right;  // Indices of child nodes if not leaf
    bool is_leaf;
};

// Solar radiation pressure scale factor [N/m^2].
// Default: 4.56e-6 N/m^2 (approximate value at 1 AU).
// All SRP force/moment results are multiplied by this value.
extern double g_srp_phi0;

// Compute SRP forces and moments from shadow labels.
// sun_x/y/z must form a normalized unit vector pointing toward the sun.
SRPResult compute_srp_forces(
    const std::vector<Triangle>& triangles,
    std::vector<int> labels,
    double sun_x, double sun_y, double sun_z);


// ---------------------------------------------------------------------------
// Centroid ray casting (one shadow ray per triangle centroid).
// ---------------------------------------------------------------------------

// primary_emitter_name: when non-empty, only polygons whose component_name matches
// this string are treated as direct-illumination emitters; all others have their
// initial labels forced to 0 and can only receive reflected light.
// Pass "" (default) for the normal behaviour where any lit polygon emits.
SRPResult calculate_labels_ray_casting_reflections(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    int max_reflections,
    bool verbose = true,
    const std::string& primary_emitter_name = "");

#ifdef SM3D_HAS_CUDA
SRPResult calculate_labels_ray_casting_reflections_gpu(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    int max_reflections,
    bool verbose = true,
    const std::string& primary_emitter_name = "");
#endif

#ifdef SM3D_HAS_OPTIX
// RTX (OptiX hardware) centroid ray casting + multi-bounce reflections.
SRPResult calculate_labels_ray_casting_rtx(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    int max_reflections = 0,
    const std::string& primary_emitter_name = "");
#endif


// ---------------------------------------------------------------------------
// Pixel-grid ray casting (parallel rays on a 2-D grid perpendicular to the sun).
// grid_step controls cell size (smaller = higher accuracy, more rays).
// ---------------------------------------------------------------------------

SRPResult calculate_labels_pixel_grid(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections = 0,
    bool verbose = true,
    const std::string& primary_emitter_name = "");

#ifdef SM3D_HAS_CUDA
SRPResult calculate_labels_pixel_grid_gpu(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections = 0,
    bool verbose = true,
    const std::string& primary_emitter_name = "");
#endif

#ifdef SM3D_HAS_OPTIX
SRPResult calculate_labels_pixel_grid_rtx(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections = 0,
    bool verbose = true,
    const std::string& primary_emitter_name = "");
#endif


// ---------------------------------------------------------------------------
// Bounce visualization globals
// Populated by all "interactive" (non-bench) calculate_labels_* functions.
// ---------------------------------------------------------------------------

// Get per-polygon reflection bounce levels from the last call.
// -1 = never hit, 0 = direct, 1+ = reflection bounce level.
const std::vector<int>& get_last_bounce_levels();

// Get per-polygon incident ray direction from the last call.
// Direction FROM which light arrived (anti-parallel to the ray).
const std::vector<std::array<double,3>>& get_last_incident_dirs();

// Get per-polygon ray origin (centroid of emitting triangle) from the last call.
// NaN for direct-sun polygons (sun is at infinity).
const std::vector<std::array<double,3>>& get_last_origin_pts();

// Populate globals from a plain labels vector.
// Sets bounce=0 for illuminated, bounce=-1 for shadowed; clears incident dirs to zero.
void set_bounce_levels_from_labels(const std::vector<int>& labels);

// Move fully-computed bounce data into the globals.
// origin_pts is optional; defaults to all-NaN when empty.
void set_bounce_globals(std::vector<int> bounce_levels,
                        std::vector<std::array<double,3>> incident_dirs,
                        std::vector<std::array<double,3>> origin_pts = {});


// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------
void visualize_triangles(
    const std::string& csv_file,
    int every,
    bool show_normals,
    std::array<double, 3> force  = {0.0, 0.0, 0.0},
    std::array<double, 3> moment = {0.0, 0.0, 0.0});

void save_results(
    const std::vector<Triangle>& triangles,
    const std::vector<int>& labels,
    const std::vector<int>& reference_labels,
    const std::string& filename,
    bool pixel_grid_mode = false);

void save_results_with_bounces(
    const std::vector<Triangle>& triangles,
    const std::vector<int>& labels,
    const std::vector<int>& reference_labels,
    const std::vector<int>& bounce_levels,
    const std::vector<std::array<double,3>>& incident_dirs,
    const std::vector<std::array<double,3>>& origin_pts,
    const std::string& filename);
