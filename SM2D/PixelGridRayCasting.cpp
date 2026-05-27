// ---------------------------------------------------------------------------
// PixelGridRayCasting.cpp
// CPU pixel-grid illumination: a 2-D orthographic grid of parallel rays
// (perpendicular to the sun direction) replaces the single center-ray-per-
// polygon approach.
//
// Algorithm:
//   1. Build an orthonormal frame {û, v̂, ŝ} where ŝ = unit sun direction.
//   2. Project all triangle vertices onto the (û, v̂) plane to find extents.
//   3. For every grid cell centre (u_i, v_j) with spacing grid_step:
//        - cast a ray from (u_i·û + v_j·v̂ + t_far·ŝ) in direction -ŝ
//        - find the closest front-facing triangle with BVH traversal
//        - if found: mark that triangle lit, accumulate SRP contribution
//   4. SRP per hit (projected area dA = step²):
//        s_coeff = -Φ₀·step²·[(1-α)+α(1-μ)]        (along ŝ)
//        n_coeff = -Φ₀·step²·[2αμcosθ+(2/3)α(1-μ)] (along n̂_j)
//      These reproduce the standard polygon-SRP formula integrated over the
//      illuminated projected area; multiple rays per polygon are allowed.
//
// Reference:
//   calculate_labels_ray_casting() - single center-ray per polygon (CPU).
//   Both methods use alpha=0.5, mu=0.5, Phi0=1.
// ---------------------------------------------------------------------------

#include "ShadowAlgorithms.h"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <numeric>
#include <iostream>
#include <functional>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Local BVH types
// ---------------------------------------------------------------------------

struct PGAABB {
    double min_x, min_y, min_z;
    double max_x, max_y, max_z;
};

struct PGBVHNode {
    PGAABB  bounds;
    size_t  triangle_idx;
    size_t  left, right;
    bool    is_leaf;
};

// ---------------------------------------------------------------------------
// Ray-AABB slab test
// ---------------------------------------------------------------------------
static bool pg_ray_aabb(const double*  o,
                        const double*  d,
                        const PGAABB& b,
                        double& tmin_out, double& tmax_out)
{
    double tmin = 0.0, tmax = 1e30;
    for (int ax = 0; ax < 3; ++ax) {
        double inv = 1.0 / d[ax];
        double lo  = ((&b.min_x)[ax] - o[ax]) * inv;
        double hi  = ((&b.max_x)[ax] - o[ax]) * inv;
        if (lo > hi) std::swap(lo, hi);
        tmin = std::max(tmin, lo);
        tmax = std::min(tmax, hi);
        if (tmin > tmax) return false;
    }
    tmin_out = tmin;
    tmax_out = tmax;
    return tmax > 1e-10;
}

// ---------------------------------------------------------------------------
// Möller-Trumbore ray-triangle intersection
// ---------------------------------------------------------------------------
static bool pg_ray_tri(const double*  o,
                       const double*  d,
                       const double* v0, const double* v1, const double* v2,
                       double& t_out)
{
    const double EPS = 1e-12;
    double e1[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
    double e2[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
    double h[3]  = { d[1]*e2[2]-d[2]*e2[1],
                     d[2]*e2[0]-d[0]*e2[2],
                     d[0]*e2[1]-d[1]*e2[0] };
    double a = e1[0]*h[0] + e1[1]*h[1] + e1[2]*h[2];
    if (std::fabs(a) < EPS) return false;
    double f = 1.0 / a;
    double s[3] = { o[0]-v0[0], o[1]-v0[1], o[2]-v0[2] };
    double u = f * (s[0]*h[0] + s[1]*h[1] + s[2]*h[2]);
    if (u < 0.0 || u > 1.0) return false;
    double q[3] = { s[1]*e1[2]-s[2]*e1[1],
                    s[2]*e1[0]-s[0]*e1[2],
                    s[0]*e1[1]-s[1]*e1[0] };
    double v = f * (d[0]*q[0] + d[1]*q[1] + d[2]*q[2]);
    if (v < 0.0 || u + v > 1.0) return false;
    double t = f * (e2[0]*q[0] + e2[1]*q[1] + e2[2]*q[2]);
    if (t < EPS) return false;
    t_out = t;
    return true;
}

// ---------------------------------------------------------------------------
// BVH traversal - returns closest front-facing hit or SIZE_MAX
// ---------------------------------------------------------------------------
static size_t pg_bvh_traverse(const double* o, const double* d,
                               const std::vector<PGBVHNode>& nodes, size_t root,
                               const std::vector<Triangle>& tris,
                               double& out_t,
                               size_t skip_idx = SIZE_MAX)
{
    out_t = 1e30;
    size_t best = SIZE_MAX;

    size_t stack[256];
    int sp = 0;
    stack[sp++] = root;

    while (sp > 0) {
        size_t idx = stack[--sp];
        const PGBVHNode& node = nodes[idx];

        double tmin, tmax;
        if (!pg_ray_aabb(o, d, node.bounds, tmin, tmax)) continue;
        if (tmin > out_t) continue;   // already have closer hit

        if (node.is_leaf) {
            size_t j = node.triangle_idx;
            if (j == skip_idx) continue;  // self-intersection guard
            const Triangle& tri = tris[j];
            double v0[3] = { tri.v1_x, tri.v1_y, tri.v1_z };
            double v1[3] = { tri.v2_x, tri.v2_y, tri.v2_z };
            double v2[3] = { tri.v3_x, tri.v3_y, tri.v3_z };
            double t;
            if (pg_ray_tri(o, d, v0, v1, v2, t) && t < out_t) {
                out_t = t;
                best  = j;
            }
        } else {
            if (sp < 254) {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// BVH boundary traversal - returns indices of all triangles whose intersection
// t value is within BOUNDARY_EPS of t_target.  Used to detect shared-edge hits.
// ---------------------------------------------------------------------------
static void pg_bvh_collect_near_t(const double* o, const double* d,
                                   const std::vector<PGBVHNode>& nodes, size_t root,
                                   const std::vector<Triangle>& tris,
                                   double t_target, double eps,
                                   size_t skip_primary,
                                   std::vector<size_t>& out_hits)
{
    size_t stack[256];
    int sp = 0;
    stack[sp++] = root;
    while (sp > 0) {
        size_t idx = stack[--sp];
        const PGBVHNode& node = nodes[idx];
        double tmin, tmax;
        if (!pg_ray_aabb(o, d, node.bounds, tmin, tmax)) continue;
        if (tmin > t_target + eps) continue;
        if (node.is_leaf) {
            size_t j = node.triangle_idx;
            if (j == skip_primary) continue;
            const Triangle& tri = tris[j];
            double v0[3] = {tri.v1_x, tri.v1_y, tri.v1_z};
            double v1[3] = {tri.v2_x, tri.v2_y, tri.v2_z};
            double v2[3] = {tri.v3_x, tri.v3_y, tri.v3_z};
            double t;
            if (pg_ray_tri(o, d, v0, v1, v2, t) && std::abs(t - t_target) <= eps)
                out_hits.push_back(j);
        } else {
            if (sp < 254) {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BVH builder
// ---------------------------------------------------------------------------
static std::vector<PGBVHNode> pg_build_bvh(const std::vector<Triangle>& tris,
                                            size_t& root_out)
{
    size_t N = tris.size();
    std::vector<PGAABB> bounds(N);
    for (size_t i = 0; i < N; ++i) {
        bounds[i] = {
            std::min({tris[i].v1_x, tris[i].v2_x, tris[i].v3_x}),
            std::min({tris[i].v1_y, tris[i].v2_y, tris[i].v3_y}),
            std::min({tris[i].v1_z, tris[i].v2_z, tris[i].v3_z}),
            std::max({tris[i].v1_x, tris[i].v2_x, tris[i].v3_x}),
            std::max({tris[i].v1_y, tris[i].v2_y, tris[i].v3_y}),
            std::max({tris[i].v1_z, tris[i].v2_z, tris[i].v3_z}),
        };
    }

    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    std::vector<PGBVHNode> nodes;
    nodes.reserve(4 * N);   // a full BVH has 2N-1 nodes; 4N is safe

    std::function<size_t(size_t, size_t)> build = [&](size_t lo, size_t hi) -> size_t {
        PGBVHNode node{};

        if (hi - lo == 1) {
            node.is_leaf      = true;
            node.triangle_idx = idx[lo];
            node.bounds       = bounds[idx[lo]];
            nodes.push_back(node);
            return nodes.size() - 1;
        }

        // Compute bounding box of the range
        PGAABB merged = bounds[idx[lo]];
        for (size_t i = lo + 1; i < hi; ++i) {
            const auto& b = bounds[idx[i]];
            merged.min_x = std::min(merged.min_x, b.min_x);
            merged.min_y = std::min(merged.min_y, b.min_y);
            merged.min_z = std::min(merged.min_z, b.min_z);
            merged.max_x = std::max(merged.max_x, b.max_x);
            merged.max_y = std::max(merged.max_y, b.max_y);
            merged.max_z = std::max(merged.max_z, b.max_z);
        }

        // Split along the longest axis
        double dx = merged.max_x - merged.min_x;
        double dy = merged.max_y - merged.min_y;
        double dz = merged.max_z - merged.min_z;
        int axis = (dy > dx && dy > dz) ? 1 : (dz > dx ? 2 : 0);

        std::sort(idx.begin() + lo, idx.begin() + hi,
            [&](size_t a, size_t b_) {
                double ca = (&bounds[a].min_x)[axis] + (&bounds[a].max_x)[axis];
                double cb = (&bounds[b_].min_x)[axis] + (&bounds[b_].max_x)[axis];
                return ca < cb;
            });

        size_t mid = (lo + hi) / 2;

        // Placeholder for the internal node (index may shift after children are pushed)
        node.is_leaf = false;
        node.bounds  = merged;
        nodes.push_back(node);
        size_t self = nodes.size() - 1;

        // Build children - children's push_back may grow vector, but 'self' remains valid
        nodes[self].left  = build(lo, mid);
        nodes[self].right = build(mid, hi);
        return self;
    };

    root_out = build(0, N);
    return nodes;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
SRPResult calculate_labels_pixel_grid(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections,
    bool verbose,
    const std::string& primary_emitter_name)
{
    if (triangles.empty()) return {};
    if (grid_step <= 0.0)
        throw std::runtime_error("grid_step must be positive");

    const double phi0 = g_srp_phi0;
    const size_t N = triangles.size();

    // Validate optical properties when reflections are requested
    if (max_reflections > 0) {
        bool has_optical_data = false;
        for (const auto& t : triangles)
            if (t.reflectance > 0.0 || t.specularity > 0.0) { has_optical_data = true; break; }
        if (!has_optical_data)
            throw std::runtime_error(
                "Reflections requested but all triangles have zero reflectance and specularity. "
                "Ensure the HDF5 file contains 'reflectance' and 'specularity' datasets.");
    }

    // ------------------------------------------------------------------ //
    // 1. Normalize sun direction                                           //
    // ------------------------------------------------------------------ //
    double slen = std::sqrt(sun_vector[0]*sun_vector[0] +
                            sun_vector[1]*sun_vector[1] +
                            sun_vector[2]*sun_vector[2]);
    if (slen < 1e-12) throw std::runtime_error("Sun vector is zero");
    double sx = sun_vector[0]/slen,
           sy = sun_vector[1]/slen,
           sz = sun_vector[2]/slen;

    // ------------------------------------------------------------------ //
    // 2. Orthonormal basis {û, v̂} perpendicular to ŝ                    //
    // ------------------------------------------------------------------ //
    double hx, hy, hz;
    if (std::fabs(sx) < 0.9) { hx = 1; hy = 0; hz = 0; }
    else                      { hx = 0; hy = 1; hz = 0; }
    // û = normalize(ŝ × h̃)
    double ux = sy*hz - sz*hy,
           uy = sz*hx - sx*hz,
           uz = sx*hy - sy*hx;
    double ulen = std::sqrt(ux*ux + uy*uy + uz*uz);
    ux /= ulen; uy /= ulen; uz /= ulen;
    // v̂ = ŝ × û  (already unit since ŝ ⊥ û)
    double vx = sy*uz - sz*uy,
           vy = sz*ux - sx*uz,
           vz = sx*uy - sy*ux;

    // ------------------------------------------------------------------ //
    // 3. Project all vertices onto (û, v̂) and along ŝ to find extents   //
    // ------------------------------------------------------------------ //
    double u_min =  1e30, u_max = -1e30;
    double v_min =  1e30, v_max = -1e30;
    double s_max = -1e30;  // max projection along ŝ (sun side)

    for (const auto& tri : triangles) {
        for (const auto& [px, py, pz] : std::array<std::array<double,3>,3>{{
                {tri.v1_x, tri.v1_y, tri.v1_z},
                {tri.v2_x, tri.v2_y, tri.v2_z},
                {tri.v3_x, tri.v3_y, tri.v3_z}}}) {
            double uc = px*ux + py*uy + pz*uz;
            double vc = px*vx + py*vy + pz*vz;
            double sc = px*sx + py*sy + pz*sz;
            u_min = std::min(u_min, uc); u_max = std::max(u_max, uc);
            v_min = std::min(v_min, vc); v_max = std::max(v_max, vc);
            s_max = std::max(s_max, sc);
        }
    }
    // Center the grid on the bounding-box midpoint so that for symmetric shapes
    // the ray distribution is exactly symmetric, eliminating spurious moments.
    // (The GPU version uses the same centering; this brings the CPU into agreement.)
    const double u_center = 0.5 * (u_min + u_max);
    const double v_center = 0.5 * (v_min + v_max);
    const double u_half   = 0.5 * (u_max - u_min) + grid_step;  // half-width + one-step margin
    const double v_half   = 0.5 * (v_max - v_min) + grid_step;
    double t_start = s_max + grid_step;  // ray starts on the sun side

    // ------------------------------------------------------------------ //
    // 4. Build BVH                                                         //
    // ------------------------------------------------------------------ //
    size_t bvh_root;
    auto nodes = pg_build_bvh(triangles, bvh_root);

    // ------------------------------------------------------------------ //
    // 5. Grid loop                                                         //
    // ------------------------------------------------------------------ //
    std::vector<int> labels(N, 0);
    std::array<double,3> total_force  = {0.0, 0.0, 0.0};
    std::array<double,3> total_moment = {0.0, 0.0, 0.0};

    // Use long long to avoid int overflow when grid_step is very small.
    long long nu = static_cast<long long>(std::ceil(2.0 * u_half / grid_step));
    long long nv = static_cast<long long>(std::ceil(2.0 * v_half / grid_step));
    // Re-derive grid origin from the center so cells are symmetric about it.
    u_min = u_center - (nu * grid_step) * 0.5;
    v_min = v_center - (nv * grid_step) * 0.5;
    long long total_rays = nu * nv;

    if (verbose)
        std::cout << "  Pixel grid: " << nu << " x " << nv
                  << " = " << total_rays << " rays (step=" << grid_step << ")\n";

    if (total_rays > 2'000'000'000LL && verbose)
        std::cout << "  WARNING: " << total_rays << " rays - this will be very slow. "
                     "Consider a larger grid step.\n";

    // Pre-compute per-cell area constant
    const double step2 = grid_step * grid_step;

    // Rays travel in direction -ŝ (from sun side toward satellite)
    const double ray_d[3] = { -sx, -sy, -sz };

    struct ReflRay {
        size_t source_idx;
        double origin[3];
        double direction[3];
        double intensity;
    };
    std::vector<ReflRay> active_rays;
    if (max_reflections > 0) active_rays.reserve(1024);

    for (int iu = 0; iu < nu; ++iu) {
        double u_c = u_min + (iu + 0.5) * grid_step;
        for (int iv = 0; iv < nv; ++iv) {
            double v_c = v_min + (iv + 0.5) * grid_step;

            // World-space ray origin (on the sun side of the model)
            double ray_o[3] = {
                u_c*ux + v_c*vx + t_start*sx,
                u_c*uy + v_c*vy + t_start*sy,
                u_c*uz + v_c*vz + t_start*sz
            };

            double t_hit;
            size_t hit = pg_bvh_traverse(ray_o, ray_d, nodes, bvh_root, triangles, t_hit);
            if (hit == SIZE_MAX) continue;

            // Front-face check: polygon normal must face the sun
            const Triangle& tri = triangles[hit];
            double cos_theta = tri.normal_x*sx + tri.normal_y*sy + tri.normal_z*sz;
            if (cos_theta <= 0.0) continue;

            // Skip triangles not belonging to the primary emitter when filtering is active
            if (!primary_emitter_name.empty() && tri.component_name != primary_emitter_name)
                continue;

            labels[hit] = 1;

            // Per-triangle optical properties
            double alpha_t = tri.reflectance;
            double mu_t    = tri.specularity;

            // SRP force contribution from this grid cell
            double sc = -phi0 * step2 * ((1.0 - alpha_t) + alpha_t * (1.0 - mu_t));   //PHY
            double n_a_t = phi0 * step2 * 2.0 * alpha_t * mu_t;                         //PHY
            double n_b_t = phi0 * step2 * (2.0/3.0) * alpha_t * (1.0 - mu_t);          //PHY
            double nc = -(n_a_t * cos_theta + n_b_t);                                   //PHY

            double dfx = sc*sx + nc*tri.normal_x;   //PHY
            double dfy = sc*sy + nc*tri.normal_y;   //PHY
            double dfz = sc*sz + nc*tri.normal_z;   //PHY

            // Polygon centroid for moment arm
            //CHANGED
           /* double cx = (tri.centroid_x != 0.0 || tri.centroid_y != 0.0 || tri.centroid_z != 0.0)
                ? tri.centroid_x : (tri.v1_x + tri.v2_x + tri.v3_x) / 3.0;
            double cy = (tri.centroid_x != 0.0 || tri.centroid_y != 0.0 || tri.centroid_z != 0.0)
                ? tri.centroid_y : (tri.v1_y + tri.v2_y + tri.v3_y) / 3.0;
            double cz = (tri.centroid_x != 0.0 || tri.centroid_y != 0.0 || tri.centroid_z != 0.0)
                ? tri.centroid_z : (tri.v1_z + tri.v2_z + tri.v3_z) / 3.0;

            total_force[0]  += dfx;
            total_force[1]  += dfy;
            total_force[2]  += dfz;
            total_moment[0] += cy*dfz - cz*dfy;
            total_moment[1] += cz*dfx - cx*dfz;
            total_moment[2] += cx*dfy - cy*dfx;*/

            total_force[0] += dfx;   //PHY
            total_force[1] += dfy;   //PHY
            total_force[2] += dfz;   //PHY

            double hit_x = ray_o[0] + t_hit * ray_d[0];
            double hit_y = ray_o[1] + t_hit * ray_d[1];
            double hit_z = ray_o[2] + t_hit * ray_d[2];
            total_moment[0] += hit_y * dfz - hit_z * dfy;   //PHY
            total_moment[1] += hit_z * dfx - hit_x * dfz;   //PHY
            total_moment[2] += hit_x * dfy - hit_y * dfx;   //PHY

            // ---- Per-pixel reflected ray for primary hit ----
            if (max_reflections > 0) {
                double alpha_i = tri.reflectance, mu_i = tri.specularity;
                if (alpha_i * mu_i > 1e-14) {
                    double rnx = 2.0*cos_theta*tri.normal_x - sx;
                    double rny = 2.0*cos_theta*tri.normal_y - sy;
                    double rnz = 2.0*cos_theta*tri.normal_z - sz;
                    double rlen = std::sqrt(rnx*rnx + rny*rny + rnz*rnz);
                    if (rlen > 1e-12) {
                        rnx /= rlen; rny /= rlen; rnz /= rlen;
                        ReflRay rr;
                        rr.source_idx   = hit;
                        rr.origin[0]    = hit_x; rr.origin[1] = hit_y; rr.origin[2] = hit_z;
                        rr.direction[0] = rnx;   rr.direction[1] = rny; rr.direction[2] = rnz;
                        rr.intensity    = phi0 * step2 * alpha_i * mu_i;   //PHY
                        active_rays.push_back(rr);
                    }
                }
            }

            // ---- Boundary co-hits: label neighbours for visualization only ----
            // Standard ray tracing: the BVH winner receives the full pixel contribution.
            // Edge-adjacent triangles are marked illuminated for display but receive no
            // force and spawn no reflected ray - exact edge hits are measure-zero.
            {
                const double BOUNDARY_EPS = 1e-5;
                std::vector<size_t> co_hits;
                pg_bvh_collect_near_t(ray_o, ray_d, nodes, bvh_root, triangles,
                                      t_hit, BOUNDARY_EPS, hit, co_hits);
                for (size_t jco : co_hits) {
                    const Triangle& tco = triangles[jco];
                    double cos_co = tco.normal_x*sx + tco.normal_y*sy + tco.normal_z*sz;
                    if (cos_co <= 0.0) continue;
                    if (!primary_emitter_name.empty() && tco.component_name != primary_emitter_name)
                        continue;
                    labels[jco] = 1;
                }
            }
        }
    }

    // ------------------------------------------------------------------ //
    // 6. Bounce tracking init + reflection loop                           //
    // ------------------------------------------------------------------ //
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> bounce_levels(N, -1);
    std::vector<std::array<double,3>> incident_dirs(N, {0.0, 0.0, 0.0});
    std::vector<std::array<double,3>> origin_pts(N, {NaN, NaN, NaN});
    for (size_t i = 0; i < N; ++i) {
        if (labels[i] == 1) {
            bounce_levels[i] = 0;
            incident_dirs[i] = { sx, sy, sz };  // sun direction for direct-illumination visualization
        }
    }

    if (verbose)
        std::cout << "  Pixel-grid reflection: " << active_rays.size() << " rays\n";

    for (int bounce = 0; bounce < max_reflections && !active_rays.empty(); ++bounce) {
        std::vector<ReflRay> next_rays;
        for (const auto& ray : active_rays) {
            double t_hit;
            size_t hit = pg_bvh_traverse(ray.origin, ray.direction,
                                         nodes, bvh_root, triangles, t_hit,
                                         ray.source_idx);
            if (hit == SIZE_MAX) continue;

            const Triangle& tk = triangles[hit];
            double s_eq[3] = { -ray.direction[0], -ray.direction[1], -ray.direction[2] };
            double cos_theta_k = tk.normal_x*s_eq[0] + tk.normal_y*s_eq[1] + tk.normal_z*s_eq[2];
            if (cos_theta_k <= 0.0) continue;

            if (bounce_levels[hit] == -1) {
                bounce_levels[hit] = bounce + 1;
                incident_dirs[hit] = { s_eq[0], s_eq[1], s_eq[2] };
                origin_pts[hit]    = { ray.origin[0], ray.origin[1], ray.origin[2] };
            }

            double alpha_k = tk.reflectance;
            double mu_k    = tk.specularity;

            double hp_x = ray.origin[0] + t_hit * ray.direction[0];
            double hp_y = ray.origin[1] + t_hit * ray.direction[1];
            double hp_z = ray.origin[2] + t_hit * ray.direction[2];

            double s_coeff = -ray.intensity * ((1.0 - alpha_k) + alpha_k * (1.0 - mu_k));                        //PHY
            double n_coeff = -ray.intensity * (2.0*alpha_k*mu_k*cos_theta_k + (2.0/3.0)*alpha_k*(1.0 - mu_k));   //PHY

            double dfx = s_coeff*s_eq[0] + n_coeff*tk.normal_x;   //PHY
            double dfy = s_coeff*s_eq[1] + n_coeff*tk.normal_y;   //PHY
            double dfz = s_coeff*s_eq[2] + n_coeff*tk.normal_z;   //PHY

            total_force[0] += dfx; total_force[1] += dfy; total_force[2] += dfz;   //PHY
            total_moment[0] += hp_y*dfz - hp_z*dfy;   //PHY
            total_moment[1] += hp_z*dfx - hp_x*dfz;   //PHY
            total_moment[2] += hp_x*dfy - hp_y*dfx;   //PHY

            double new_intensity = ray.intensity * alpha_k * mu_k;
            if (new_intensity < 1e-10) continue;

            double nkx = tk.normal_x, nky = tk.normal_y, nkz = tk.normal_z;
            double rnx = 2.0*cos_theta_k*nkx - s_eq[0];
            double rny = 2.0*cos_theta_k*nky - s_eq[1];
            double rnz = 2.0*cos_theta_k*nkz - s_eq[2];
            double rnlen = std::sqrt(rnx*rnx + rny*rny + rnz*rnz);
            if (rnlen < 1e-12) continue;
            rnx /= rnlen; rny /= rnlen; rnz /= rnlen;

            ReflRay next;
            next.source_idx   = hit;
            next.origin[0]    = hp_x; next.origin[1] = hp_y; next.origin[2] = hp_z;
            next.direction[0] = rnx;  next.direction[1] = rny; next.direction[2] = rnz;
            next.intensity    = new_intensity;
            next_rays.push_back(next);
        }
        if (verbose)
            std::cout << "  Pixel-grid bounce " << (bounce+1) << ": " << next_rays.size() << " rays\n";
        active_rays = std::move(next_rays);
    }

    set_bounce_globals(std::move(bounce_levels), std::move(incident_dirs), std::move(origin_pts));

    SRPResult result;
    result.labels       = std::move(labels);
    result.total_force  = total_force;
    result.total_moment = total_moment;
    return result;
}
