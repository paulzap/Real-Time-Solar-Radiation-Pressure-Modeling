// ---------------------------------------------------------------------------
// ReflectionRayCasting.cpp
// Clone of calculate_labels_ray_casting with multi-bounce specular reflections.
//
// Physics:
//   Reflection direction:  r = 2(n . s)n - s
//       n = unit surface normal, s = unit direction toward the light source.
//   Reflected intensity:   I_refl = I_incident * alpha * mu
//       alpha = reflectivity (0.5), mu = specular fraction (0.5).
//   Each illuminated polygon spawns one specular reflected ray from its center.
//   The ray is traced against the BVH; if it hits a front-facing polygon,
//   the SRP contribution from that reflected light is accumulated,
//   and a new reflected ray is spawned from the hit polygon for the next bounce.
//   Diffuse reflection is neglected (scattered, no directional ray).
// ---------------------------------------------------------------------------

#include "ShadowAlgorithms.h"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <functional>

// Global storage for per-polygon reflection bounce levels from the last reflection call
// -1 = never hit, 0 = direct illumination, 1+ = reflection bounce level
static std::vector<int> g_last_bounce_levels;

// Global storage for incident ray direction on first-bounce (bounce=1) polygons.
// Each entry is {dir_x, dir_y, dir_z} = direction FROM which the reflected light arrived.
// For polygons not hit by bounce=1, values are {0,0,0}.
static std::vector<std::array<double,3>> g_last_incident_dirs;

// Global storage for the ray origin point (centroid of the emitting triangle) for each
// reflected polygon. NaN for direct sun polygons (no specific origin; sun is at infinity).
// Empty when origin tracking is not available (GPU paths).
static std::vector<std::array<double,3>> g_last_origin_pts;

// Getter functions
const std::vector<int>& get_last_bounce_levels() {
    return g_last_bounce_levels;
}

const std::vector<std::array<double,3>>& get_last_incident_dirs() {
    return g_last_incident_dirs;
}

const std::vector<std::array<double,3>>& get_last_origin_pts() {
    return g_last_origin_pts;
}

// Called by the GPU path (which has no per-bounce tracking) to ensure the globals
// are always fresh and correctly sized before save_results_with_bounces() is used.
// Polygons with label==1 get bounce=0 (direct), all others get bounce=-1 (not hit).
// Incident directions are cleared to zero (no reflection arrows on GPU path).
void set_bounce_levels_from_labels(const std::vector<int>& labels) {
    const size_t N = labels.size();
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    g_last_bounce_levels.resize(N);
    g_last_incident_dirs.assign(N, {0.0, 0.0, 0.0});
    g_last_origin_pts.assign(N, {NaN, NaN, NaN});
    for (size_t i = 0; i < N; ++i) {
        g_last_bounce_levels[i] = (labels[i] == 1) ? 0 : -1;
    }
}

// Called by the GPU path after it has computed per-polygon bounce data on the device.
// Moves arrays directly into the globals used by save_results_with_bounces().
// origin_pts is optional (GPU paths pass empty); defaults to all-NaN when empty.
void set_bounce_globals(std::vector<int> bounce_levels,
                        std::vector<std::array<double,3>> incident_dirs,
                        std::vector<std::array<double,3>> origin_pts) {
    g_last_bounce_levels = std::move(bounce_levels);
    g_last_incident_dirs  = std::move(incident_dirs);
    g_last_origin_pts     = std::move(origin_pts);
}

// Local (static) Moller-Trumbore ray-triangle intersection
static bool ray_tri_intersect_refl(
    const double* origin, const double* dir,
    const double* v0, const double* v1, const double* v2,
    double& t)
{
    const double EPSILON = 1e-6;
    double edge1[3], edge2[3], h[3], s[3], q[3];
    double a, f, u, v;

    for (int i = 0; i < 3; ++i) {
        edge1[i] = v1[i] - v0[i];
        edge2[i] = v2[i] - v0[i];
    }

    h[0] = dir[1] * edge2[2] - dir[2] * edge2[1];
    h[1] = dir[2] * edge2[0] - dir[0] * edge2[2];
    h[2] = dir[0] * edge2[1] - dir[1] * edge2[0];

    a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];
    if (std::abs(a) < EPSILON) return false;

    f = 1.0 / a;
    for (int i = 0; i < 3; ++i) s[i] = origin[i] - v0[i];

    u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if (u < 0.0 || u > 1.0) return false;

    q[0] = s[1] * edge1[2] - s[2] * edge1[1];
    q[1] = s[2] * edge1[0] - s[0] * edge1[2];
    q[2] = s[0] * edge1[1] - s[1] * edge1[0];

    v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
    if (v < 0.0 || u + v > 1.0) return false;

    t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);
    return (t > EPSILON);
}


SRPResult calculate_labels_ray_casting_reflections(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    int max_reflections,
    bool verbose,
    const std::string& primary_emitter_name)
{
    if (triangles.empty())
        return {};

    const double default_alpha = 0.5;
    const double default_mu    = 0.5;
    const double phi0  = g_srp_phi0;
    const double MIN_INTENSITY = 1e-10;

    // --- Normalize sun direction ---
    double len = std::sqrt(sun_vector[0] * sun_vector[0] +
                           sun_vector[1] * sun_vector[1] +
                           sun_vector[2] * sun_vector[2]);
    if (len < 1e-12)
        throw std::runtime_error("Sun vector is zero");
    double dir_x = sun_vector[0] / len;
    double dir_y = sun_vector[1] / len;
    double dir_z = sun_vector[2] / len;

    size_t N = triangles.size();
    const double epsilon = 1e-5;   // 1e-5 matches RTX backface_filter threshold  //PHY

    // --- Initialize labels and centers ---
    std::vector<int> labels(N, 1);
    std::vector<std::array<double, 3>> centers(N);
    for (size_t i = 0; i < N; ++i) {
        centers[i] = {
            (triangles[i].v1_x + triangles[i].v2_x + triangles[i].v3_x) / 3.0,
            (triangles[i].v1_y + triangles[i].v2_y + triangles[i].v3_y) / 3.0,
            (triangles[i].v1_z + triangles[i].v2_z + triangles[i].v3_z) / 3.0
        };
        double dot_norm = triangles[i].normal_x * dir_x +
                          triangles[i].normal_y * dir_y +
                          triangles[i].normal_z * dir_z;
        if (dot_norm <= epsilon) {
            labels[i] = 0;
        }
    }

    // ===================== Build BVH (identical to original) =====================
    std::vector<AABB> triangle_bounds(N);
    for (size_t i = 0; i < N; ++i) {
        triangle_bounds[i] = {
            std::min({triangles[i].v1_x, triangles[i].v2_x, triangles[i].v3_x}),
            std::min({triangles[i].v1_y, triangles[i].v2_y, triangles[i].v3_y}),
            std::min({triangles[i].v1_z, triangles[i].v2_z, triangles[i].v3_z}),
            std::max({triangles[i].v1_x, triangles[i].v2_x, triangles[i].v3_x}),
            std::max({triangles[i].v1_y, triangles[i].v2_y, triangles[i].v3_y}),
            std::max({triangles[i].v1_z, triangles[i].v2_z, triangles[i].v3_z})
        };
    }

    std::vector<BVHNode> nodes;
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    auto build_bvh = [&](auto& self, size_t start, size_t end, size_t depth) -> size_t {
        BVHNode node;
        if (end - start == 1) {
            node.is_leaf = true;
            node.triangle_idx = indices[start];
            node.bounds = triangle_bounds[node.triangle_idx];
            nodes.push_back(node);
            return nodes.size() - 1;
        }

        double x_min = std::numeric_limits<double>::infinity(), x_max = -x_min;
        double y_min = x_min, y_max = -x_min;
        double z_min = x_min, z_max = -x_min;
        for (size_t i = start; i < end; ++i) {
            size_t idx = indices[i];
            x_min = std::min(x_min, triangle_bounds[idx].min_x);
            x_max = std::max(x_max, triangle_bounds[idx].max_x);
            y_min = std::min(y_min, triangle_bounds[idx].min_y);
            y_max = std::max(y_max, triangle_bounds[idx].max_y);
            z_min = std::min(z_min, triangle_bounds[idx].min_z);
            z_max = std::max(z_max, triangle_bounds[idx].max_z);
        }
        double x_range = x_max - x_min;
        double y_range = y_max - y_min;
        double z_range = z_max - z_min;

        int axis = 0;
        if (y_range > x_range && y_range > z_range) axis = 1;
        else if (z_range > x_range) axis = 2;

        std::sort(indices.begin() + start, indices.begin() + end, [&](size_t a, size_t b) {
            return centers[a][axis] < centers[b][axis];
        });

        size_t mid = (start + end) / 2;
        node.is_leaf = false;
        node.left  = self(self, start, mid, depth + 1);
        node.right = self(self, mid,   end, depth + 1);

        node.bounds = {
            std::min(nodes[node.left].bounds.min_x, nodes[node.right].bounds.min_x),
            std::min(nodes[node.left].bounds.min_y, nodes[node.right].bounds.min_y),
            std::min(nodes[node.left].bounds.min_z, nodes[node.right].bounds.min_z),
            std::max(nodes[node.left].bounds.max_x, nodes[node.right].bounds.max_x),
            std::max(nodes[node.left].bounds.max_y, nodes[node.right].bounds.max_y),
            std::max(nodes[node.left].bounds.max_z, nodes[node.right].bounds.max_z)
        };
        nodes.push_back(node);
        return nodes.size() - 1;
    };

    size_t root = build_bvh(build_bvh, 0, N, 0);

    // Ray-AABB intersection (identical to original)
    auto ray_aabb_intersection = [](const double origin[3], const double dir[3],
                                    const AABB& box, double& t_min, double& t_max) {
        double t1 = (box.min_x - origin[0]) / (dir[0] + 1e-10);
        double t2 = (box.max_x - origin[0]) / (dir[0] + 1e-10);
        t_min = std::min(t1, t2);
        t_max = std::max(t1, t2);
        t1 = (box.min_y - origin[1]) / (dir[1] + 1e-10);
        t2 = (box.max_y - origin[1]) / (dir[1] + 1e-10);
        t_min = std::max(t_min, std::min(t1, t2));
        t_max = std::min(t_max, std::max(t1, t2));
        t1 = (box.min_z - origin[2]) / (dir[2] + 1e-10);
        t2 = (box.max_z - origin[2]) / (dir[2] + 1e-10);
        t_min = std::max(t_min, std::min(t1, t2));
        t_max = std::min(t_max, std::max(t1, t2));
        return t_max >= t_min && t_max >= -1e-6;
    };

    // ===================== Direct shadow testing (identical to original) =====================
    for (size_t i = 0; i < N; ++i) {
        if (labels[i] == 0) continue;
        // Offset origin along normal — prevent false self-shadow from adjacent panels  //PHY
        double origin[3] = {
            centers[i][0] + 1e-4 * triangles[i].normal_x,
            centers[i][1] + 1e-4 * triangles[i].normal_y,
            centers[i][2] + 1e-4 * triangles[i].normal_z
        };
        double dir[3]    = { dir_x, dir_y, dir_z };
        std::vector<size_t> stack = { root };
        bool shadowed = false;

        while (!stack.empty() && !shadowed) {
            size_t node_idx = stack.back();
            stack.pop_back();
            const BVHNode& node = nodes[node_idx];

            double t_min, t_max;
            if (!ray_aabb_intersection(origin, dir, node.bounds, t_min, t_max)) continue;

            if (node.is_leaf) {
                size_t j = node.triangle_idx;
                if (i == j) continue;

                double diff_x = centers[j][0] - origin[0];
                double diff_y = centers[j][1] - origin[1];
                double diff_z = centers[j][2] - origin[2];
                double proj = diff_x * dir_x + diff_y * dir_y + diff_z * dir_z;
                if (proj <= epsilon) continue;

                double v0[3] = { triangles[j].v1_x, triangles[j].v1_y, triangles[j].v1_z };
                double v1[3] = { triangles[j].v2_x, triangles[j].v2_y, triangles[j].v2_z };
                double v2[3] = { triangles[j].v3_x, triangles[j].v3_y, triangles[j].v3_z };
                double t;
                if (ray_tri_intersect_refl(origin, dir, v0, v1, v2, t)) {
                    labels[i] = 0;
                    shadowed = true;
                }
            }
            else {
                stack.push_back(node.left);
                stack.push_back(node.right);
            }
        }
    }

    // If a primary emitter is specified, restrict direct illumination to that component only.
    // All other polygons can only receive reflected light, not direct sunlight.
    if (!primary_emitter_name.empty()) {
        for (size_t i = 0; i < N; ++i) {
            if (labels[i] == 1 && triangles[i].component_name != primary_emitter_name)
                labels[i] = 0;
        }
    }

    // Compute direct-illumination SRP
    SRPResult result = compute_srp_forces(triangles, labels, dir_x, dir_y, dir_z);

    // Track which bounce level first illuminated each polygon (0=direct, 1=1st bounce, etc., -1=never)
    std::vector<int> bounce_levels(N, -1);
    // Track incident direction for each illuminated polygon (FROM which light arrived).
    // bounce=0: the sun direction - stored so the visualizer can draw direct-sun rays.
    // bounce>0: set below in the bounce loop.
    std::vector<std::array<double,3>> incident_dirs(N, {0.0, 0.0, 0.0});
    // Track exact ray origin (centroid of emitting triangle) for reflected polygons.
    // NaN for bounce=0 (direct sun; no specific origin point - sun is at infinity).
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::array<double,3>> origin_pts(N, {NaN, NaN, NaN});
    for (size_t i = 0; i < N; ++i) {
        if (labels[i] == 1) {
            bounce_levels[i] = 0;  // directly illuminated
            incident_dirs[i] = { dir_x, dir_y, dir_z };  // sun direction = FROM which sunlight comes
        }
    }

    // ===================== Multi-bounce specular reflections =====================
    if (max_reflections <= 0) {
        // Always update globals so visualization reflects the current run, not a stale one
        g_last_bounce_levels = bounce_levels;
        g_last_incident_dirs  = incident_dirs;
        g_last_origin_pts     = origin_pts;
        return result;
    }

    // Offset applied to every reflected ray origin along the triangle's surface normal.
    // Prevents rays from hitting the thin edge-strip triangles of the same physical plate
    // (plate thickness is ~1-2 mm; this offset lifts the ray origin clear of those strips).
    static constexpr double SURFACE_OFFSET = 2e-3;  // 2 mm

    struct ReflectionRay {
        size_t source_idx;   // polygon the ray originates from (self-intersection guard)
        double origin[3];
        double direction[3];
        double intensity;
    };

    // --- Generate first-bounce reflected rays from all directly-illuminated polygons ---
    std::vector<ReflectionRay> active_rays;
    active_rays.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        if (labels[i] == 0) continue;

        double nx = triangles[i].normal_x;
        double ny = triangles[i].normal_y;
        double nz = triangles[i].normal_z;
        // s = sun direction (toward source), cos_theta = n . s
        double cos_theta = nx * dir_x + ny * dir_y + nz * dir_z;

        // Reflected direction: r = 2(n . s) n - s
        double rx = 2.0 * cos_theta * nx - dir_x;
        double ry = 2.0 * cos_theta * ny - dir_y;
        double rz = 2.0 * cos_theta * nz - dir_z;
        double rlen = std::sqrt(rx * rx + ry * ry + rz * rz);
        if (rlen < 1e-12) continue;
        rx /= rlen;  ry /= rlen;  rz /= rlen;

        // Per-triangle optical properties (fall back to defaults for legacy CSV)
        double alpha_i = (triangles[i].reflectance > 0.0 || triangles[i].specularity > 0.0)
            ? triangles[i].reflectance : default_alpha;
        double mu_i    = (triangles[i].reflectance > 0.0 || triangles[i].specularity > 0.0)
            ? triangles[i].specularity : default_mu;
        // Reflected ray TOTAL POWER (W): the polygon intercepts phi0*A_i*cos_theta of solar
        // power, of which alpha*mu is specularly reflected into the single representative ray.
        // (Previously this stored only the flux density phi0*alpha*mu, and the per-hit force
        //  formula compensated by multiplying by A_j*cos_theta_j -- physically wrong unless
        //  A_i*cos_theta_i happens to equal A_j*cos_theta_j. See Li 2018 Eq. (5).)
        double intensity = phi0 * triangles[i].area * cos_theta * alpha_i * mu_i;

        ReflectionRay ray;
        ray.source_idx  = i;
        ray.origin[0]   = centers[i][0] + SURFACE_OFFSET * triangles[i].normal_x;
        ray.origin[1]   = centers[i][1] + SURFACE_OFFSET * triangles[i].normal_y;
        ray.origin[2]   = centers[i][2] + SURFACE_OFFSET * triangles[i].normal_z;
        ray.direction[0] = rx;
        ray.direction[1] = ry;
        ray.direction[2] = rz;
        ray.intensity    = intensity;
        active_rays.push_back(ray);
    }

    if (verbose) std::cout << "  Reflection bounce 0 (direct): " << active_rays.size() << " reflected rays generated\n";

    // --- Bounce loop ---
    for (int bounce = 0; bounce < max_reflections && !active_rays.empty(); ++bounce) {
        std::vector<ReflectionRay> next_rays;

        for (const auto& ray : active_rays) {
            // ---- BVH traversal: find CLOSEST hit ----
            size_t  closest_idx = SIZE_MAX;
            double  closest_t   = std::numeric_limits<double>::infinity();

            std::vector<size_t> stack = { root };
            while (!stack.empty()) {
                size_t node_idx = stack.back();
                stack.pop_back();
                const BVHNode& node = nodes[node_idx];

                double t_min, t_max;
                if (!ray_aabb_intersection(ray.origin, ray.direction, node.bounds, t_min, t_max))
                    continue;
                if (t_min > closest_t) continue;  // prune: no closer hit possible

                if (node.is_leaf) {
                    size_t j = node.triangle_idx;
                    if (j == ray.source_idx) continue;  // self-intersection avoidance

                    double v0[3] = { triangles[j].v1_x, triangles[j].v1_y, triangles[j].v1_z };
                    double v1[3] = { triangles[j].v2_x, triangles[j].v2_y, triangles[j].v2_z };
                    double v2[3] = { triangles[j].v3_x, triangles[j].v3_y, triangles[j].v3_z };
                    double t;
                    if (ray_tri_intersect_refl(ray.origin, ray.direction, v0, v1, v2, t)) {
                        if (t < closest_t) {
                            closest_t   = t;
                            closest_idx = j;
                        }
                    }
                }
                else {
                    stack.push_back(node.left);
                    stack.push_back(node.right);
                }
            }

            if (closest_idx == SIZE_MAX) continue;  // ray escaped the scene

            // Update bounce level, incident direction, and ray origin for first hit only
            if (bounce_levels[closest_idx] == -1) {
                bounce_levels[closest_idx] = bounce + 1;  // first time hit at bounce level (bounce+1)
                // Store incident direction: the direction FROM which the reflected light came
                // (i.e., the reverse of the ray's travel direction, pointing back toward the source)
                incident_dirs[closest_idx] = { -ray.direction[0],
                                               -ray.direction[1],
                                               -ray.direction[2] };
                // Store exact ray origin (centroid of the emitting triangle) for visualization
                origin_pts[closest_idx] = { ray.origin[0], ray.origin[1], ray.origin[2] };
            }

            // Check that the hit polygon's front face is toward the incoming light
            const Triangle& tj = triangles[closest_idx];
            // s_eq = direction from hit polygon toward the light source = -ray.direction
            double s_eq[3] = { -ray.direction[0], -ray.direction[1], -ray.direction[2] };
            double cos_theta_j = tj.normal_x * s_eq[0] +
                                 tj.normal_y * s_eq[1] +
                                 tj.normal_z * s_eq[2];
            if (cos_theta_j <= 0.0) continue;  // back face hit, light blocked

            // ---- Accumulate SRP force from reflected light on hit polygon ----
            // Per-triangle optical properties of the hit triangle
            double alpha_j = (tj.reflectance > 0.0 || tj.specularity > 0.0)
                ? tj.reflectance : default_alpha;
            double mu_j    = (tj.reflectance > 0.0 || tj.specularity > 0.0)
                ? tj.specularity : default_mu;

            // SRP force: ray.intensity = phi0 * A_emitter * cos_emitter * alpha * mu — beam
            // cross-section is conserved through specular bounces. NOT multiplied by A_j * cos_j.
            double base    = ray.intensity;                                                                        //PHY
            double s_coeff = -base * ((1.0 - alpha_j) + alpha_j * (1.0 - mu_j));                                //PHY
            double n_coeff = -base * (2.0 * alpha_j * mu_j * cos_theta_j +                                      //PHY
                                      (2.0 / 3.0) * alpha_j * (1.0 - mu_j));

            double fx = s_coeff * s_eq[0] + n_coeff * tj.normal_x;   //PHY
            double fy = s_coeff * s_eq[1] + n_coeff * tj.normal_y;   //PHY
            double fz = s_coeff * s_eq[2] + n_coeff * tj.normal_z;   //PHY

            result.total_force[0] += fx;   //PHY
            result.total_force[1] += fy;   //PHY
            result.total_force[2] += fz;   //PHY

            double cx = centers[closest_idx][0];
            double cy = centers[closest_idx][1];
            double cz = centers[closest_idx][2];
            result.total_moment[0] += cy * fz - cz * fy;   //PHY
            result.total_moment[1] += cz * fx - cx * fz;   //PHY
            result.total_moment[2] += cx * fy - cy * fx;   //PHY

            // ---- Boundary co-hits: triangles within BOUNDARY_EPS of closest_t ----
            // Both triangles sharing the hit edge are illuminated and emit reflected rays.
            {
                const double BOUNDARY_EPS = 1e-5;
                std::vector<size_t> b_stk = { root };
                while (!b_stk.empty()) {
                    size_t bidx = b_stk.back(); b_stk.pop_back();
                    const BVHNode& bn = nodes[bidx];
                    double tb_min, tb_max;
                    if (!ray_aabb_intersection(ray.origin, ray.direction, bn.bounds, tb_min, tb_max))
                        continue;
                    if (tb_min > closest_t + BOUNDARY_EPS) continue;
                    if (!bn.is_leaf) {
                        b_stk.push_back(bn.left);
                        b_stk.push_back(bn.right);
                        continue;
                    }
                    size_t jco = bn.triangle_idx;
                    if (jco == ray.source_idx || jco == closest_idx) continue;
                    double v0c[3] = {triangles[jco].v1_x,triangles[jco].v1_y,triangles[jco].v1_z};
                    double v1c[3] = {triangles[jco].v2_x,triangles[jco].v2_y,triangles[jco].v2_z};
                    double v2c[3] = {triangles[jco].v3_x,triangles[jco].v3_y,triangles[jco].v3_z};
                    double t_co;
                    if (!ray_tri_intersect_refl(ray.origin, ray.direction, v0c, v1c, v2c, t_co)) continue;
                    if (std::abs(t_co - closest_t) > BOUNDARY_EPS) continue;

                    const Triangle& tjco = triangles[jco];
                    double s_eq_co[3] = {-ray.direction[0], -ray.direction[1], -ray.direction[2]};
                    double cos_co = tjco.normal_x*s_eq_co[0] + tjco.normal_y*s_eq_co[1] + tjco.normal_z*s_eq_co[2];
                    if (cos_co <= 0.0) continue;

                    if (bounce_levels[jco] == -1) {
                        bounce_levels[jco] = bounce + 1;
                        incident_dirs[jco] = {s_eq_co[0], s_eq_co[1], s_eq_co[2]};
                        origin_pts[jco]    = {ray.origin[0], ray.origin[1], ray.origin[2]};
                    }
                    result.labels[jco] = 1;

                    double alpha_co = (tjco.reflectance > 0.0 || tjco.specularity > 0.0)
                        ? tjco.reflectance : default_alpha;
                    double mu_co = (tjco.reflectance > 0.0 || tjco.specularity > 0.0)
                        ? tjco.specularity : default_mu;
                    double base_co = ray.intensity;                                                                          //PHY
                    double sc_co   = -base_co * ((1.0 - alpha_co) + alpha_co*(1.0 - mu_co));                               //PHY
                    double nc_co   = -base_co * (2.0*alpha_co*mu_co*cos_co + (2.0/3.0)*alpha_co*(1.0 - mu_co));            //PHY
                    double fx_co = sc_co*s_eq_co[0] + nc_co*tjco.normal_x;   //PHY
                    double fy_co = sc_co*s_eq_co[1] + nc_co*tjco.normal_y;   //PHY
                    double fz_co = sc_co*s_eq_co[2] + nc_co*tjco.normal_z;   //PHY
                    result.total_force[0] += fx_co;   //PHY
                    result.total_force[1] += fy_co;   //PHY
                    result.total_force[2] += fz_co;   //PHY
                    result.total_moment[0] += centers[jco][1]*fz_co - centers[jco][2]*fy_co;   //PHY
                    result.total_moment[1] += centers[jco][2]*fx_co - centers[jco][0]*fz_co;   //PHY
                    result.total_moment[2] += centers[jco][0]*fy_co - centers[jco][1]*fx_co;   //PHY

                    double new_int_co = ray.intensity * alpha_co * mu_co;
                    if (new_int_co < MIN_INTENSITY) continue;
                    double rnx_co = 2.0*cos_co*tjco.normal_x - s_eq_co[0];
                    double rny_co = 2.0*cos_co*tjco.normal_y - s_eq_co[1];
                    double rnz_co = 2.0*cos_co*tjco.normal_z - s_eq_co[2];
                    double rlen_co = std::sqrt(rnx_co*rnx_co + rny_co*rny_co + rnz_co*rnz_co);
                    if (rlen_co < 1e-12) continue;
                    rnx_co /= rlen_co; rny_co /= rlen_co; rnz_co /= rlen_co;
                    ReflectionRay next_co;
                    next_co.source_idx  = jco;
                    next_co.origin[0]   = centers[jco][0] + SURFACE_OFFSET * tjco.normal_x;
                    next_co.origin[1]   = centers[jco][1] + SURFACE_OFFSET * tjco.normal_y;
                    next_co.origin[2]   = centers[jco][2] + SURFACE_OFFSET * tjco.normal_z;
                    next_co.direction[0] = rnx_co;
                    next_co.direction[1] = rny_co;
                    next_co.direction[2] = rnz_co;
                    next_co.intensity    = new_int_co;
                    next_rays.push_back(next_co);
                }
            }

            // ---- Generate next-bounce reflected ray ----
            double new_intensity = ray.intensity * alpha_j * mu_j;
            if (new_intensity < MIN_INTENSITY) continue;

            double nj_x = tj.normal_x, nj_y = tj.normal_y, nj_z = tj.normal_z;
            // r_next = 2(nj . s_eq) nj - s_eq
            double rnx = 2.0 * cos_theta_j * nj_x - s_eq[0];
            double rny = 2.0 * cos_theta_j * nj_y - s_eq[1];
            double rnz = 2.0 * cos_theta_j * nj_z - s_eq[2];
            double rnlen = std::sqrt(rnx * rnx + rny * rny + rnz * rnz);
            if (rnlen < 1e-12) continue;
            rnx /= rnlen;  rny /= rnlen;  rnz /= rnlen;

            ReflectionRay next;
            next.source_idx  = closest_idx;
            next.origin[0]   = cx + SURFACE_OFFSET * nj_x;
            next.origin[1]   = cy + SURFACE_OFFSET * nj_y;
            next.origin[2]   = cz + SURFACE_OFFSET * nj_z;
            next.direction[0] = rnx;
            next.direction[1] = rny;
            next.direction[2] = rnz;
            next.intensity    = new_intensity;
            next_rays.push_back(next);
        }

        if (verbose) std::cout << "  Reflection bounce " << (bounce + 1) << ": "
                               << next_rays.size() << " rays (from " << active_rays.size() << ")\n";

        active_rays = std::move(next_rays);
    }

    // Store bounce levels, incident directions, and ray origins for visualization
    g_last_bounce_levels = bounce_levels;
    g_last_incident_dirs  = incident_dirs;
    g_last_origin_pts     = origin_pts;

    return result;
}
