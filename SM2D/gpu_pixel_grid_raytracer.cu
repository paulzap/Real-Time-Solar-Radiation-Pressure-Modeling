// ---------------------------------------------------------------------------
// gpu_pixel_grid_raytracer.cu
// GPU pixel-grid illumination: each CUDA thread handles one (u_i, v_j) grid
// cell, tracing a parallel ray from the sun side in direction -ŝ.
//
// Same physics as PixelGridRayCasting.cpp (CPU counterpart):
//   dA_proj = grid_step²  per ray cell
//   s_coeff = -Φ₀·step²·[(1-α)+α(1-μ)]
//   n_coeff = -Φ₀·step²·[2αμcosθ+(2/3)α(1-μ)]
//
// BVH is built on the host and transferred to the device; re-used across calls
// as long as the triangle set does not change (GPU cache).
// ---------------------------------------------------------------------------

#include "ShadowAlgorithms.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>          // for debug file output
#include <limits>
#include <stdexcept>
#include <functional>

// ---------------------------------------------------------------------------
// Debug switch: 0 = off, 1 = basic prints, 2 = per‑triangle contributions
// ---------------------------------------------------------------------------
#define DEBUG_PRINT 1

#if DEBUG_PRINT
#include <iomanip>
#endif

// ---------------------------------------------------------------------------
// Custom atomicAdd for double for older architectures (< sm_60)
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// ---------------------------------------------------------------------------
// CUDA error-check macro
// ---------------------------------------------------------------------------
#define PG_CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            char buf[512]; \
            snprintf(buf, sizeof(buf), "CUDA error %s:%d - %s", \
                     __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(buf); \
        } \
    } while(0)

// ---------------------------------------------------------------------------
// Device-side structs – now using double where possible
// ---------------------------------------------------------------------------
struct PGTriDev {
    double v[3][3];         // vertices [0..2][xyz]
    double normal[3];
    double center[3];
    float reflectance;      // rho - fraction of light reflected (still float)
    float specularity;      // s   - specular fraction of reflected (still float)
};

struct PGBVHNodeDev {
    double  bounds[6];      // [min_x, min_y, min_z, max_x, max_y, max_z]
    unsigned int left;
    unsigned int right;
    unsigned int triangle_idx;
    int    is_leaf;         // 1=leaf, 0=internal
};

// ---------------------------------------------------------------------------
// Host-side BVH helpers - used only during GPU BVH construction, not for traversal.
// ---------------------------------------------------------------------------
struct PGHostAABB { double mn[3], mx[3]; };
struct PGHostNode {
    PGHostAABB b;
    size_t ti, l, r;
    bool   leaf;
};

// ---------------------------------------------------------------------------
// Device functions - BVH traversal helpers (now with double)
// ---------------------------------------------------------------------------
__device__ static inline bool pg_ray_aabb_dev(
    const double* __restrict__ o, const double* __restrict__ d,
    const double* __restrict__ b, double& tmin_out, double& tmax_out)
{
    double tmin = 0.0, tmax = 1e30;
    for (int ax = 0; ax < 3; ++ax) {
        double inv = 1.0 / d[ax];
        double lo = (b[ax] - o[ax]) * inv;      // b[0..2] = min
        double hi = (b[ax + 3] - o[ax]) * inv;  // b[3..5] = max
        if (lo > hi) { double tmp = lo; lo = hi; hi = tmp; }
        tmin = fmax(tmin, lo);
        tmax = fmin(tmax, hi);
        if (tmin > tmax) return false;
    }
    tmin_out = tmin;
    tmax_out = tmax;
    return tmax > 1e-8;
}

__device__ static bool pg_ray_tri_dev(
    const double* __restrict__ o, const double* __restrict__ d,
    const double* v0, const double* v1, const double* v2,
    double& t_out)
{
    const double EPS = 1e-8;
    double e1[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
    double e2[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };
    double h[3] = { d[1] * e2[2] - d[2] * e2[1],
                    d[2] * e2[0] - d[0] * e2[2],
                    d[0] * e2[1] - d[1] * e2[0] };
    double a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
    if (fabs(a) < EPS) return false;
    double f = 1.0 / a;
    double s[3] = { o[0] - v0[0], o[1] - v0[1], o[2] - v0[2] };
    double u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if (u < 0.0 || u > 1.0) return false;
    double q[3] = { s[1] * e1[2] - s[2] * e1[1],
                    s[2] * e1[0] - s[0] * e1[2],
                    s[0] * e1[1] - s[1] * e1[0] };
    double v = f * (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]);
    if (v < 0.0 || u + v > 1.0) return false;
    double t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
    if (t < EPS) return false;
    t_out = t;
    return true;
}

// ---------------------------------------------------------------------------
// Kernel: one thread per (iu, iv) grid cell – now uses double
// ---------------------------------------------------------------------------
__global__ void pixelGridKernel(
    const PGTriDev* __restrict__ triangles,
    const PGBVHNodeDev* __restrict__ nodes,
    unsigned int root_idx,
    // grid geometry
    double ux, double uy, double uz,   // û basis
    double vx, double vy, double vz,   // v̂ basis
    double sx, double sy, double sz,   // ŝ (toward sun)
    double u_min, double v_min,
    double grid_step,
    int   nu, int nv,
    double t_start,                  // ray-origin offset along ŝ
    double step2,                    // = grid_step²
    // outputs
    int* __restrict__ d_labels,         // [N]
    double* __restrict__ d_force,       // [3] global accumulators
    double* __restrict__ d_moment,      // [3]
    // per-pixel hit data for bounce pass
    int*    d_hit_tri_out,              // [nu*nv]  triangle index (-1 = miss)
    double* d_hit_pt_out,               // [nu*nv*3] world-space hit point
    double* d_hit_intensity_out,        // [nu*nv]  step2 * alpha * mu
    double* d_hit_dir_out,              // [nu*nv*3] s_eq toward sun
    // bounce tracking (interactive only; nullptr for bench)
    int*    d_bounce_levels,            // [N]  atomicMin to 0 on direct hit
    float*  d_incident_dirs)            // [N*3] CAS sentinel=-1 int
{
    // 2-D launch: blockIdx.x / threadIdx.x → iu, blockIdx.y / threadIdx.y → iv
    int iu = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int iv = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (iu >= nu || iv >= nv) return;

    double u_c = u_min + (iu + 0.5) * grid_step;
    double v_c = v_min + (iv + 0.5) * grid_step;

    // World-space ray origin (on sun side)
    double ray_o[3] = {
        u_c * ux + v_c * vx + t_start * sx,
        u_c * uy + v_c * vy + t_start * sy,
        u_c * uz + v_c * vz + t_start * sz
    };
    // Ray direction = -ŝ
    double ray_d[3] = { -sx, -sy, -sz };

    // BVH traversal - iterative DFS
    unsigned int stack[128];
    int sp = 0;
    stack[sp++] = root_idx;

    int   closest = -1;
    double closest_t = 1e30;

    while (sp > 0) {
        unsigned int node_idx = stack[--sp];
        const PGBVHNodeDev& node = nodes[node_idx];

        double tmin, tmax;
        if (!pg_ray_aabb_dev(ray_o, ray_d, node.bounds, tmin, tmax)) continue;
        if (tmin > closest_t) continue;

        if (node.is_leaf) {
            unsigned int j = node.triangle_idx;
            double t;
            if (pg_ray_tri_dev(ray_o, ray_d,
                triangles[j].v[0],
                triangles[j].v[1],
                triangles[j].v[2], t)) {
                if (t < closest_t) {
                    closest_t = t;
                    closest = (int)j;
                }
            }
        }
        else {
            if (sp < 126) {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }
    }

    if (closest < 0) return;

    // Front-face check
    double cos_theta = triangles[closest].normal[0] * sx
        + triangles[closest].normal[1] * sy
        + triangles[closest].normal[2] * sz;
    if (cos_theta <= 1e-6) return;

    // Mark polygon lit
    atomicOr(&d_labels[closest], 1);

    // Per-triangle optical properties
    double alpha_t = (double)triangles[closest].reflectance;
    double mu_t = (double)triangles[closest].specularity;

    // SRP contribution from this grid cell
    double sc = -step2 * ((1.0 - alpha_t) + alpha_t * (1.0 - mu_t));
    double n_a_t = step2 * 2.0 * alpha_t * mu_t;
    double n_b_t = step2 * (2.0 / 3.0) * alpha_t * (1.0 - mu_t);
    double nc = -(n_a_t * cos_theta + n_b_t);   // force coeff along n̂_j
    double dfx = sc * sx + nc * triangles[closest].normal[0];
    double dfy = sc * sy + nc * triangles[closest].normal[1];
    double dfz = sc * sz + nc * triangles[closest].normal[2];

    // Accumulate force (double atomic add)
    atomicAdd(&d_force[0], dfx);
    atomicAdd(&d_force[1], dfy);
    atomicAdd(&d_force[2], dfz);

    // Moment about origin using hit point
    double hit_x = ray_o[0] + closest_t * ray_d[0];
    double hit_y = ray_o[1] + closest_t * ray_d[1];
    double hit_z = ray_o[2] + closest_t * ray_d[2];
    atomicAdd(&d_moment[0], hit_y * dfz - hit_z * dfy);
    atomicAdd(&d_moment[1], hit_z * dfx - hit_x * dfz);
    atomicAdd(&d_moment[2], hit_x * dfy - hit_y * dfx);

    // Per-pixel hit data for bounce pass
    if (d_hit_tri_out) {
        int idx = iv * nu + iu;
        d_hit_tri_out[idx]        = closest;
        d_hit_pt_out[idx*3+0]     = hit_x;
        d_hit_pt_out[idx*3+1]     = hit_y;
        d_hit_pt_out[idx*3+2]     = hit_z;
        d_hit_intensity_out[idx]  = step2 * alpha_t * mu_t;
        d_hit_dir_out[idx*3+0]    = sx;
        d_hit_dir_out[idx*3+1]    = sy;
        d_hit_dir_out[idx*3+2]    = sz;
    }

    // Bounce tracking (interactive only)
    if (d_bounce_levels)
        atomicMin(&d_bounce_levels[closest], 0);
    if (d_incident_dirs) {
        int prev = atomicCAS(reinterpret_cast<int*>(&d_incident_dirs[closest*3]),
                             -1, __float_as_int((float)sx));
        if (prev == -1) {
            d_incident_dirs[closest*3+1] = (float)sy;
            d_incident_dirs[closest*3+2] = (float)sz;
        }
    }
}

// ---------------------------------------------------------------------------
// Bounce kernel: one thread per (iu, iv), reads previous-bounce hit state,
// fires specular reflection ray, accumulates force/moment on receiver.
// ---------------------------------------------------------------------------
__global__ void pixelGridBounceKernel(
    const PGTriDev* __restrict__ triangles,
    const PGBVHNodeDev* __restrict__ nodes,
    unsigned int root_idx,
    int nu, int nv,
    // input per-pixel state (from previous pass)
    const int*    __restrict__ d_in_tri,        // [nu*nv]
    const double* __restrict__ d_in_pt,         // [nu*nv*3]
    const double* __restrict__ d_in_intensity,  // [nu*nv]
    const double* __restrict__ d_in_dir,        // [nu*nv*3] s_eq toward prev source
    // output per-pixel state (for next pass)
    int*    d_out_tri,
    double* d_out_pt,
    double* d_out_intensity,
    double* d_out_dir,
    // force/moment accumulators
    double* d_force,
    double* d_moment,
    // bounce tracking (nullptr for bench)
    int*   d_bounce_levels,
    float* d_incident_dirs,
    float* d_origin_pts,
    int    bounce_num)
{
    int iu = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int iv = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (iu >= nu || iv >= nv) return;

    int idx = iv * nu + iu;

    // Default: no hit forwarded
    if (d_out_tri) d_out_tri[idx] = -1;

    int src_tri = d_in_tri[idx];
    if (src_tri < 0) return;

    double intensity = d_in_intensity[idx];
    if (intensity < 1e-30) return;

    // s_eq: direction from src surface toward its light source
    double seq_x = d_in_dir[idx*3+0];
    double seq_y = d_in_dir[idx*3+1];
    double seq_z = d_in_dir[idx*3+2];

    double nx = triangles[src_tri].normal[0];
    double ny = triangles[src_tri].normal[1];
    double nz = triangles[src_tri].normal[2];

    double cos_src = nx*seq_x + ny*seq_y + nz*seq_z;
    if (cos_src <= 1e-6) return;

    // Specular reflection direction from src triangle
    double rx = 2.0*cos_src*nx - seq_x;
    double ry = 2.0*cos_src*ny - seq_y;
    double rz = 2.0*cos_src*nz - seq_z;
    double rlen = sqrt(rx*rx + ry*ry + rz*rz);
    if (rlen < 1e-12) return;
    rx /= rlen; ry /= rlen; rz /= rlen;

    // Ray origin: hit point of previous pass + offset along src normal
    double ox = d_in_pt[idx*3+0] + 1e-4*nx;
    double oy = d_in_pt[idx*3+1] + 1e-4*ny;
    double oz = d_in_pt[idx*3+2] + 1e-4*nz;

    double ray_o[3] = { ox, oy, oz };
    double ray_d[3] = { rx, ry, rz };

    // BVH traversal
    unsigned int stack[128];
    int sp = 0;
    stack[sp++] = root_idx;

    int    hit_tri = -1;
    double hit_t   = 1e30;

    while (sp > 0) {
        unsigned int node_idx = stack[--sp];
        const PGBVHNodeDev& node = nodes[node_idx];

        double tmin, tmax;
        if (!pg_ray_aabb_dev(ray_o, ray_d, node.bounds, tmin, tmax)) continue;
        if (tmin > hit_t) continue;

        if (node.is_leaf) {
            unsigned int j = node.triangle_idx;
            if ((int)j == src_tri) continue;  // skip source triangle
            double t;
            if (pg_ray_tri_dev(ray_o, ray_d,
                    triangles[j].v[0], triangles[j].v[1], triangles[j].v[2], t)) {
                if (t < hit_t) { hit_t = t; hit_tri = (int)j; }
            }
        } else {
            if (sp < 126) { stack[sp++] = node.left; stack[sp++] = node.right; }
        }
    }

    if (hit_tri < 0) return;

    double n2x = triangles[hit_tri].normal[0];
    double n2y = triangles[hit_tri].normal[1];
    double n2z = triangles[hit_tri].normal[2];
    // s_eq for receiver: light arrives from -refl direction
    double s2x = -rx, s2y = -ry, s2z = -rz;
    double cos2 = n2x*s2x + n2y*s2y + n2z*s2z;
    if (cos2 <= 1e-6) return;

    double hit_x = ox + hit_t * rx;
    double hit_y = oy + hit_t * ry;
    double hit_z = oz + hit_t * rz;

    double alpha2 = (double)triangles[hit_tri].reflectance;
    double mu2    = (double)triangles[hit_tri].specularity;

    double sc2 = -intensity * ((1.0 - alpha2) + alpha2*(1.0 - mu2));
    double nc2 = -intensity * (2.0*alpha2*mu2*cos2 + (2.0/3.0)*alpha2*(1.0 - mu2));
    double dfx = sc2*s2x + nc2*n2x;
    double dfy = sc2*s2y + nc2*n2y;
    double dfz = sc2*s2z + nc2*n2z;

    atomicAdd(&d_force[0], dfx);
    atomicAdd(&d_force[1], dfy);
    atomicAdd(&d_force[2], dfz);
    atomicAdd(&d_moment[0], hit_y*dfz - hit_z*dfy);
    atomicAdd(&d_moment[1], hit_z*dfx - hit_x*dfz);
    atomicAdd(&d_moment[2], hit_x*dfy - hit_y*dfx);

    // Bounce tracking (interactive only)
    if (d_bounce_levels)
        atomicMin(&d_bounce_levels[hit_tri], bounce_num + 1);
    if (d_incident_dirs) {
        int prev = atomicCAS(reinterpret_cast<int*>(&d_incident_dirs[hit_tri*3]),
                             -1, __float_as_int((float)s2x));
        if (prev == -1) {
            d_incident_dirs[hit_tri*3+1] = (float)s2y;
            d_incident_dirs[hit_tri*3+2] = (float)s2z;
        }
    }
    if (d_origin_pts) {
        int prev = atomicCAS(reinterpret_cast<int*>(&d_origin_pts[hit_tri*3]),
                             -1, __float_as_int((float)d_in_pt[idx*3+0]));
        if (prev == -1) {
            d_origin_pts[hit_tri*3+1] = (float)d_in_pt[idx*3+1];
            d_origin_pts[hit_tri*3+2] = (float)d_in_pt[idx*3+2];
        }
    }

    // Forward state for next bounce
    if (d_out_tri) {
        d_out_tri[idx]         = hit_tri;
        d_out_pt[idx*3+0]      = hit_x;
        d_out_pt[idx*3+1]      = hit_y;
        d_out_pt[idx*3+2]      = hit_z;
        d_out_intensity[idx]   = intensity * alpha2 * mu2;
        d_out_dir[idx*3+0]     = s2x;
        d_out_dir[idx*3+1]     = s2y;
        d_out_dir[idx*3+2]     = s2z;
    }
}

// ---------------------------------------------------------------------------
// GPU cache - rebuilt only when triangle set changes
// ---------------------------------------------------------------------------
struct PGGPUCache {
    PGTriDev*     d_tri           = nullptr;
    PGBVHNodeDev* d_nodes         = nullptr;
    int*          d_labels        = nullptr;
    int*          h_pinned_labels = nullptr;
    double*       d_force         = nullptr;
    double*       d_moment        = nullptr;
    unsigned int  root_idx        = 0;
    size_t        N               = 0;
    size_t        numNodes        = 0;
    bool          valid           = false;
    cudaStream_t  stream          = nullptr;
    // Per-pixel bounce ping-pong buffers (size = nu*nv each)
    int*    d_bounce_tri_A   = nullptr;
    int*    d_bounce_tri_B   = nullptr;
    double* d_bounce_pt_A    = nullptr;
    double* d_bounce_pt_B    = nullptr;
    double* d_bounce_int_A   = nullptr;
    double* d_bounce_int_B   = nullptr;
    double* d_bounce_dir_A   = nullptr;
    double* d_bounce_dir_B   = nullptr;
    size_t  bounce_px_alloc  = 0;
    // Per-triangle bounce tracking for visualization (size = N)
    int*    d_bounce_levels  = nullptr;  // INT_MAX = never hit
    float*  d_incident_dirs  = nullptr;  // sentinel -1 as int (0xFFFFFFFF)
    float*  d_origin_pts     = nullptr;  // sentinel -1 as int (0xFFFFFFFF)
};

static PGGPUCache g_pg_cache;

// ---------------------------------------------------------------------------
// Host function
// ---------------------------------------------------------------------------
SRPResult calculate_labels_pixel_grid_gpu(
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
    // Normalize sun direction (now double)
    // ------------------------------------------------------------------ //
    double slen = std::sqrt(sun_vector[0] * sun_vector[0] +
        sun_vector[1] * sun_vector[1] +
        sun_vector[2] * sun_vector[2]);
    if (slen < 1e-12) throw std::runtime_error("Sun vector is zero");
    double sx = sun_vector[0] / slen;
    double sy = sun_vector[1] / slen;
    double sz = sun_vector[2] / slen;

    // ------------------------------------------------------------------ //
    // Orthonormal basis (double)
    // ------------------------------------------------------------------ //
    double hx, hy, hz;
    if (fabs(sx) < 0.9) { hx = 1; hy = 0; hz = 0; }
    else { hx = 0; hy = 1; hz = 0; }
    double ux = sy * hz - sz * hy, uy = sz * hx - sx * hz, uz = sx * hy - sy * hx;
    double ulen = sqrt(ux * ux + uy * uy + uz * uz);
    ux /= ulen; uy /= ulen; uz /= ulen;
    double vx = sy * uz - sz * uy, vy = sz * ux - sx * uz, vz = sx * uy - sy * ux;

    // ------------------------------------------------------------------ //
    // Project vertices to find grid extents (double)
    // ------------------------------------------------------------------ //
    double u_min = 1e30, u_max = -1e30;
    double v_min = 1e30, v_max = -1e30;
    double s_max = -1e30;

    for (const auto& tri : triangles) {
        for (const auto& pt : std::array<std::array<double, 3>, 3>{{
            {tri.v1_x, tri.v1_y, tri.v1_z},
            { tri.v2_x, tri.v2_y, tri.v2_z },
            { tri.v3_x, tri.v3_y, tri.v3_z }}}) {
            double uc = pt[0] * ux + pt[1] * uy + pt[2] * uz;
            double vc = pt[0] * vx + pt[1] * vy + pt[2] * vz;
            double sc = pt[0] * sx + pt[1] * sy + pt[2] * sz;
            u_min = std::min(u_min, uc); u_max = std::max(u_max, uc);
            v_min = std::min(v_min, vc); v_max = std::max(v_max, vc);
            s_max = std::max(s_max, sc);
        }
    }

    double gs = grid_step;

    // Center the grid on the shape's bounding-box midpoint so that for
    // symmetric shapes the ray distribution is exactly symmetric about the
    // origin.  The old approach (u_min -= step) placed the grid origin at
    // an arbitrary edge, creating a systematic ~half-cell offset in the
    // average hit position, which produced a spurious moment proportional
    // to  F_total * (offset distance)  - shown as M_err ≈ 100 % for shapes
    // whose true moment is zero.
    const double u_center = 0.5 * (u_min + u_max);
    const double v_center = 0.5 * (v_min + v_max);
    const double u_half   = 0.5 * (u_max - u_min) + gs;  // half-width + one-step margin
    const double v_half   = 0.5 * (v_max - v_min) + gs;

    double t_start = s_max + gs;

    long long nu_ll = (long long)std::ceil(2.0 * u_half / gs);
    long long nv_ll = (long long)std::ceil(2.0 * v_half / gs);
    // Re-derive the grid origin from the center so the cells are symmetric.
    // Ray centre for cell (iu, iv) = (u_min + (iu+0.5)*gs, v_min + (iv+0.5)*gs).
    u_min = u_center - (nu_ll * gs) * 0.5;
    v_min = v_center - (nv_ll * gs) * 0.5;
    long long total_rays = nu_ll * nv_ll;

    if (nu_ll > 2'000'000'000LL || nv_ll > 2'000'000'000LL) {
        throw std::runtime_error(
            "Pixel grid step is too small: grid dimensions (" +
            std::to_string(nu_ll) + " x " + std::to_string(nv_ll) +
            ") exceed CUDA limits. Use a larger step.");
    }
    int nu = (int)nu_ll;
    int nv = (int)nv_ll;


    // ------------------------------------------------------------------ //
    // Rebuild GPU cache if triangles changed
    // ------------------------------------------------------------------ //
    if (!g_pg_cache.valid || g_pg_cache.N != N) {
        // Free old geometry/label resources
        if (g_pg_cache.d_tri)           PG_CUDA_CHECK(cudaFree(g_pg_cache.d_tri));
        if (g_pg_cache.d_nodes)         PG_CUDA_CHECK(cudaFree(g_pg_cache.d_nodes));
        if (g_pg_cache.d_labels)        PG_CUDA_CHECK(cudaFree(g_pg_cache.d_labels));
        if (g_pg_cache.h_pinned_labels) PG_CUDA_CHECK(cudaFreeHost(g_pg_cache.h_pinned_labels));
        if (g_pg_cache.d_force)         PG_CUDA_CHECK(cudaFree(g_pg_cache.d_force));
        if (g_pg_cache.d_moment)        PG_CUDA_CHECK(cudaFree(g_pg_cache.d_moment));
        // Free bounce tracking (N-sized)
        if (g_pg_cache.d_bounce_levels) PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_levels));
        if (g_pg_cache.d_incident_dirs) PG_CUDA_CHECK(cudaFree(g_pg_cache.d_incident_dirs));
        if (g_pg_cache.d_origin_pts)    PG_CUDA_CHECK(cudaFree(g_pg_cache.d_origin_pts));
        // Free ping-pong bounce buffers (nu*nv-sized, freed separately)
        if (g_pg_cache.d_bounce_tri_A)  PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_tri_A));
        if (g_pg_cache.d_bounce_tri_B)  PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_tri_B));
        if (g_pg_cache.d_bounce_pt_A)   PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_pt_A));
        if (g_pg_cache.d_bounce_pt_B)   PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_pt_B));
        if (g_pg_cache.d_bounce_int_A)  PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_int_A));
        if (g_pg_cache.d_bounce_int_B)  PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_int_B));
        if (g_pg_cache.d_bounce_dir_A)  PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_dir_A));
        if (g_pg_cache.d_bounce_dir_B)  PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_dir_B));
        if (g_pg_cache.stream)          PG_CUDA_CHECK(cudaStreamDestroy(g_pg_cache.stream));
        g_pg_cache = PGGPUCache{};

        PG_CUDA_CHECK(cudaStreamCreate(&g_pg_cache.stream));
        PG_CUDA_CHECK(cudaHostAlloc(&g_pg_cache.h_pinned_labels,
            N * sizeof(int), cudaHostAllocDefault));

        // ---- Build BVH on host (double) ----
        std::vector<PGHostAABB> tb(N);
        for (size_t i = 0; i < N; ++i) {
            tb[i] = { {std::min({triangles[i].v1_x,triangles[i].v2_x,triangles[i].v3_x}),
                       std::min({triangles[i].v1_y,triangles[i].v2_y,triangles[i].v3_y}),
                       std::min({triangles[i].v1_z,triangles[i].v2_z,triangles[i].v3_z})},
                      {std::max({triangles[i].v1_x,triangles[i].v2_x,triangles[i].v3_x}),
                       std::max({triangles[i].v1_y,triangles[i].v2_y,triangles[i].v3_y}),
                       std::max({triangles[i].v1_z,triangles[i].v2_z,triangles[i].v3_z})} };
        }
        std::vector<PGHostNode> hn;
        hn.reserve(4 * N);
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);

        std::function<size_t(size_t, size_t)> bld = [&](size_t lo, size_t hi) -> size_t {
            PGHostNode nd{};
            if (hi - lo == 1) {
                nd.leaf = true; nd.ti = idx[lo]; nd.b = tb[idx[lo]];
                hn.push_back(nd); return hn.size() - 1;
            }
            PGHostAABB merged = tb[idx[lo]];
            for (size_t k = lo + 1; k < hi; ++k)
                for (int ax = 0; ax < 3; ++ax) {
                    merged.mn[ax] = std::min(merged.mn[ax], tb[idx[k]].mn[ax]);
                    merged.mx[ax] = std::max(merged.mx[ax], tb[idx[k]].mx[ax]);
                }
            int axis = 0;
            double dx = merged.mx[0] - merged.mn[0];
            double dy = merged.mx[1] - merged.mn[1];
            double dz = merged.mx[2] - merged.mn[2];
            if (dy > dx && dy > dz) axis = 1; else if (dz > dx) axis = 2;
            std::sort(idx.begin() + lo, idx.begin() + hi, [&](size_t a, size_t b_) {
                return tb[a].mn[axis] + tb[a].mx[axis] < tb[b_].mn[axis] + tb[b_].mx[axis];
                });
            size_t mid = (lo + hi) / 2;
            nd.leaf = false; nd.b = merged;
            hn.push_back(nd);
            size_t self = hn.size() - 1;
            hn[self].l = bld(lo, mid);
            hn[self].r = bld(mid, hi);
            return self;
            };
        size_t root = bld(0, N);
        g_pg_cache.root_idx = (unsigned int)root;
        g_pg_cache.numNodes = hn.size();

        // ---- Pack GPU structs (double) ----
        std::vector<PGTriDev> triDev(N);
        for (size_t i = 0; i < N; ++i) {
            triDev[i].v[0][0] = triangles[i].v1_x; triDev[i].v[0][1] = triangles[i].v1_y; triDev[i].v[0][2] = triangles[i].v1_z;
            triDev[i].v[1][0] = triangles[i].v2_x; triDev[i].v[1][1] = triangles[i].v2_y; triDev[i].v[1][2] = triangles[i].v2_z;
            triDev[i].v[2][0] = triangles[i].v3_x; triDev[i].v[2][1] = triangles[i].v3_y; triDev[i].v[2][2] = triangles[i].v3_z;
            triDev[i].normal[0] = triangles[i].normal_x;
            triDev[i].normal[1] = triangles[i].normal_y;
            triDev[i].normal[2] = triangles[i].normal_z;
            triDev[i].center[0] = (triangles[i].v1_x + triangles[i].v2_x + triangles[i].v3_x) / 3.0;
            triDev[i].center[1] = (triangles[i].v1_y + triangles[i].v2_y + triangles[i].v3_y) / 3.0;
            triDev[i].center[2] = (triangles[i].v1_z + triangles[i].v2_z + triangles[i].v3_z) / 3.0;
            triDev[i].reflectance = (float)triangles[i].reflectance;
            triDev[i].specularity = (float)triangles[i].specularity;
        }

        std::vector<PGBVHNodeDev> nodeDev(hn.size());
        for (size_t i = 0; i < hn.size(); ++i) {
            const auto& h = hn[i];
            nodeDev[i].bounds[0] = h.b.mn[0]; nodeDev[i].bounds[1] = h.b.mn[1]; nodeDev[i].bounds[2] = h.b.mn[2];
            nodeDev[i].bounds[3] = h.b.mx[0]; nodeDev[i].bounds[4] = h.b.mx[1]; nodeDev[i].bounds[5] = h.b.mx[2];
            nodeDev[i].is_leaf = h.leaf ? 1 : 0;
            nodeDev[i].triangle_idx = h.leaf ? (unsigned int)h.ti : 0u;
            nodeDev[i].left = h.leaf ? 0u : (unsigned int)h.l;
            nodeDev[i].right = h.leaf ? 0u : (unsigned int)h.r;
        }

        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_tri,    N * sizeof(PGTriDev)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_nodes,  hn.size() * sizeof(PGBVHNodeDev)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_labels, N * sizeof(int)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_force,  3 * sizeof(double)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_moment, 3 * sizeof(double)));
        // Bounce tracking arrays (per-triangle, N elements)
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_bounce_levels, N * sizeof(int)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_incident_dirs, N * 3 * sizeof(float)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_origin_pts,    N * 3 * sizeof(float)));

        PG_CUDA_CHECK(cudaMemcpy(g_pg_cache.d_tri, triDev.data(), N * sizeof(PGTriDev), cudaMemcpyHostToDevice));
        PG_CUDA_CHECK(cudaMemcpy(g_pg_cache.d_nodes, nodeDev.data(), hn.size() * sizeof(PGBVHNodeDev), cudaMemcpyHostToDevice));

        g_pg_cache.N = N;
        g_pg_cache.valid = true;
    }

    // ------------------------------------------------------------------ //
    // Reset per-call arrays
    // ------------------------------------------------------------------ //
    PG_CUDA_CHECK(cudaMemsetAsync(g_pg_cache.d_labels, 0, N * sizeof(int), g_pg_cache.stream));
    PG_CUDA_CHECK(cudaMemsetAsync(g_pg_cache.d_force,  0, 3 * sizeof(double), g_pg_cache.stream));
    PG_CUDA_CHECK(cudaMemsetAsync(g_pg_cache.d_moment, 0, 3 * sizeof(double), g_pg_cache.stream));
    // Bounce tracking: INT_MAX for levels (0x7FFFFFFF bytes → 0x7F7F7F7F ≈ INT_MAX/2, fine for sentinel)
    PG_CUDA_CHECK(cudaMemsetAsync(g_pg_cache.d_bounce_levels, 0x7F,
        N * sizeof(int), g_pg_cache.stream));
    // Sentinel -1 as int for incident dirs and origin pts (0xFF bytes → 0xFFFFFFFF = -1 as int)
    PG_CUDA_CHECK(cudaMemsetAsync(g_pg_cache.d_incident_dirs, 0xFF,
        N * 3 * sizeof(float), g_pg_cache.stream));
    PG_CUDA_CHECK(cudaMemsetAsync(g_pg_cache.d_origin_pts, 0xFF,
        N * 3 * sizeof(float), g_pg_cache.stream));

    double step2 = gs * gs;
    size_t px_count = (size_t)nu * (size_t)nv;

    // ------------------------------------------------------------------ //
    // Reallocate ping-pong bounce buffers if grid size grew
    // ------------------------------------------------------------------ //
    if (max_reflections > 0 && px_count > g_pg_cache.bounce_px_alloc) {
        if (g_pg_cache.d_bounce_tri_A) PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_tri_A));
        if (g_pg_cache.d_bounce_tri_B) PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_tri_B));
        if (g_pg_cache.d_bounce_pt_A)  PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_pt_A));
        if (g_pg_cache.d_bounce_pt_B)  PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_pt_B));
        if (g_pg_cache.d_bounce_int_A) PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_int_A));
        if (g_pg_cache.d_bounce_int_B) PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_int_B));
        if (g_pg_cache.d_bounce_dir_A) PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_dir_A));
        if (g_pg_cache.d_bounce_dir_B) PG_CUDA_CHECK(cudaFree(g_pg_cache.d_bounce_dir_B));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_bounce_tri_A, px_count * sizeof(int)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_bounce_tri_B, px_count * sizeof(int)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_bounce_pt_A,  px_count * 3 * sizeof(double)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_bounce_pt_B,  px_count * 3 * sizeof(double)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_bounce_int_A, px_count * sizeof(double)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_bounce_int_B, px_count * sizeof(double)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_bounce_dir_A, px_count * 3 * sizeof(double)));
        PG_CUDA_CHECK(cudaMalloc(&g_pg_cache.d_bounce_dir_B, px_count * 3 * sizeof(double)));
        g_pg_cache.bounce_px_alloc = px_count;
    }

    // ------------------------------------------------------------------ //
    // Launch direct kernel - 2D grid so nu*nv never overflows a single int
    // ------------------------------------------------------------------ //
    const int TW = 16, TH = 16;
    dim3 blockDim2d(TW, TH);
    dim3 gridDim2d((nu + TW - 1) / TW, (nv + TH - 1) / TH);

    // Init d_bounce_tri_A to -1 (0xFF bytes → 0xFFFFFFFF = -1 as int)
    if (max_reflections > 0)
        PG_CUDA_CHECK(cudaMemsetAsync(g_pg_cache.d_bounce_tri_A, 0xFF,
            px_count * sizeof(int), g_pg_cache.stream));

    pixelGridKernel<<<gridDim2d, blockDim2d, 0, g_pg_cache.stream>>>(
        g_pg_cache.d_tri,
        g_pg_cache.d_nodes,
        g_pg_cache.root_idx,
        ux, uy, uz,
        vx, vy, vz,
        sx, sy, sz,
        u_min, v_min,
        gs,
        nu, nv,
        t_start,
        step2,
        g_pg_cache.d_labels,
        g_pg_cache.d_force,
        g_pg_cache.d_moment,
        // per-pixel bounce buffers (nullptr when no reflections requested)
        max_reflections > 0 ? g_pg_cache.d_bounce_tri_A : nullptr,
        max_reflections > 0 ? g_pg_cache.d_bounce_pt_A  : nullptr,
        max_reflections > 0 ? g_pg_cache.d_bounce_int_A : nullptr,
        max_reflections > 0 ? g_pg_cache.d_bounce_dir_A : nullptr,
        // bounce tracking
        g_pg_cache.d_bounce_levels,
        g_pg_cache.d_incident_dirs);
    PG_CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------ //
    // GPU bounce loop
    // ------------------------------------------------------------------ //
    int*    cur_tri = g_pg_cache.d_bounce_tri_A;
    int*    nxt_tri = g_pg_cache.d_bounce_tri_B;
    double* cur_pt  = g_pg_cache.d_bounce_pt_A;
    double* nxt_pt  = g_pg_cache.d_bounce_pt_B;
    double* cur_int = g_pg_cache.d_bounce_int_A;
    double* nxt_int = g_pg_cache.d_bounce_int_B;
    double* cur_dir = g_pg_cache.d_bounce_dir_A;
    double* nxt_dir = g_pg_cache.d_bounce_dir_B;

    for (int b = 0; b < max_reflections; ++b) {
        PG_CUDA_CHECK(cudaMemsetAsync(nxt_tri, 0xFF, px_count * sizeof(int), g_pg_cache.stream));

        pixelGridBounceKernel<<<gridDim2d, blockDim2d, 0, g_pg_cache.stream>>>(
            g_pg_cache.d_tri,
            g_pg_cache.d_nodes,
            g_pg_cache.root_idx,
            nu, nv,
            cur_tri, cur_pt, cur_int, cur_dir,
            nxt_tri, nxt_pt, nxt_int, nxt_dir,
            g_pg_cache.d_force,
            g_pg_cache.d_moment,
            g_pg_cache.d_bounce_levels,
            g_pg_cache.d_incident_dirs,
            g_pg_cache.d_origin_pts,
            b);
        PG_CUDA_CHECK(cudaGetLastError());

        std::swap(cur_tri, nxt_tri);
        std::swap(cur_pt,  nxt_pt);
        std::swap(cur_int, nxt_int);
        std::swap(cur_dir, nxt_dir);
    }

    // ------------------------------------------------------------------ //
    // Copy results back
    // ------------------------------------------------------------------ //
    PG_CUDA_CHECK(cudaMemcpyAsync(g_pg_cache.h_pinned_labels,
        g_pg_cache.d_labels, N * sizeof(int),
        cudaMemcpyDeviceToHost, g_pg_cache.stream));
    double h_force[3]  = {};
    double h_moment[3] = {};
    PG_CUDA_CHECK(cudaMemcpyAsync(h_force,  g_pg_cache.d_force,  3 * sizeof(double),
        cudaMemcpyDeviceToHost, g_pg_cache.stream));
    PG_CUDA_CHECK(cudaMemcpyAsync(h_moment, g_pg_cache.d_moment, 3 * sizeof(double),
        cudaMemcpyDeviceToHost, g_pg_cache.stream));

    // Copy bounce tracking
    std::vector<int>   h_bounce_levels(N);
    std::vector<float> h_incident_dirs(N * 3);
    std::vector<float> h_origin_pts(N * 3);
    PG_CUDA_CHECK(cudaMemcpyAsync(h_bounce_levels.data(), g_pg_cache.d_bounce_levels,
        N * sizeof(int), cudaMemcpyDeviceToHost, g_pg_cache.stream));
    PG_CUDA_CHECK(cudaMemcpyAsync(h_incident_dirs.data(), g_pg_cache.d_incident_dirs,
        N * 3 * sizeof(float), cudaMemcpyDeviceToHost, g_pg_cache.stream));
    PG_CUDA_CHECK(cudaMemcpyAsync(h_origin_pts.data(), g_pg_cache.d_origin_pts,
        N * 3 * sizeof(float), cudaMemcpyDeviceToHost, g_pg_cache.stream));
    PG_CUDA_CHECK(cudaStreamSynchronize(g_pg_cache.stream));

    std::vector<int> labels(g_pg_cache.h_pinned_labels,
        g_pg_cache.h_pinned_labels + N);

    // Apply phi0 scale factor consistently for both direct and bounce forces
    const double phi0 = g_srp_phi0;
    std::array<double, 3> total_force  = { h_force[0]  * phi0, h_force[1]  * phi0, h_force[2]  * phi0 };
    std::array<double, 3> total_moment = { h_moment[0] * phi0, h_moment[1] * phi0, h_moment[2] * phi0 };

    // ------------------------------------------------------------------ //
    // Primary emitter filter (re-run force from labels if needed)
    // ------------------------------------------------------------------ //
    if (!primary_emitter_name.empty()) {
        for (size_t i = 0; i < N; ++i) {
            if (labels[i] == 1 && triangles[i].component_name != primary_emitter_name)
                labels[i] = 0;
        }
        double ds = std::sqrt(sun_vector[0] * sun_vector[0] +
            sun_vector[1] * sun_vector[1] +
            sun_vector[2] * sun_vector[2]);
        if (ds > 1e-12) {
            SRPResult filtered = compute_srp_forces(
                triangles, labels,
                sun_vector[0] / ds, sun_vector[1] / ds, sun_vector[2] / ds);
            total_force  = filtered.total_force;
            total_moment = filtered.total_moment;
        }
    }

    // ------------------------------------------------------------------ //
    // Convert bounce tracking float→double and populate globals
    // ------------------------------------------------------------------ //
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    // Sentinel: 0x7F7F7F7F ≈ 2.14e9, treat anything > 1M as "never hit" → -1
    std::vector<int> bl(N);
    for (size_t i = 0; i < N; ++i)
        bl[i] = (h_bounce_levels[i] > 1000000) ? -1 : h_bounce_levels[i];

    std::vector<std::array<double, 3>> incident_dirs(N, { 0.0, 0.0, 0.0 });
    std::vector<std::array<double, 3>> origin_pts(N, { NaN, NaN, NaN });
    for (size_t i = 0; i < N; ++i) {
        // Sentinel -1 as int = 0xFFFFFFFF; as float that's -nan
        auto is_sentinel = [](float f) {
            return *reinterpret_cast<const unsigned int*>(&f) == 0xFFFFFFFFu;
        };
        if (!is_sentinel(h_incident_dirs[i*3]))
            incident_dirs[i] = { (double)h_incident_dirs[i*3],
                                  (double)h_incident_dirs[i*3+1],
                                  (double)h_incident_dirs[i*3+2] };
        if (!is_sentinel(h_origin_pts[i*3]))
            origin_pts[i] = { (double)h_origin_pts[i*3],
                               (double)h_origin_pts[i*3+1],
                               (double)h_origin_pts[i*3+2] };
    }
    set_bounce_globals(std::move(bl), std::move(incident_dirs), std::move(origin_pts));

#if DEBUG_PRINT >= 1
    {
        int lit_count = 0;
        for (int v : labels) if (v) ++lit_count;
        std::cout << "  GPU pixel-grid: " << nu << "x" << nv << " grid, "
                  << lit_count << "/" << N << " triangles lit\n";
    }
#endif

    SRPResult result;
    result.labels      = std::move(labels);
    result.total_force  = total_force;
    result.total_moment = total_moment;
    return result;
}