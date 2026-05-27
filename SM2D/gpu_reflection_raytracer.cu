// ---------------------------------------------------------------------------
// gpu_reflection_raytracer.cu
// Clone of gpu_shadow_raytracer.cu with multi-bounce specular reflections.
//
// Phase 1 (direct illumination) is identical to the original GPU ray caster.
// Phase 2 adds iterative reflection bounces on the GPU:
//   - First-bounce rays are generated from all directly-illuminated polygons.
//   - Each bounce: trace rays via BVH (closest hit), accumulate SRP forces
//     with atomicAdd, spawn next-bounce rays.
//   - Continues up to max_reflections bounces or until no active rays remain.
//
// Reflection direction:  r = 2(n . s)n - s
// Reflected intensity:   I_refl = I_incident * alpha * mu
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "SatelliteDataset.h"
#include "ShadowAlgorithms.h"

#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <functional>
#include <iostream>

#define REFL_CUDA_CHECK(ans) { reflGpuAssert((ans), __FILE__, __LINE__); }
inline void reflGpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ---------------------- GPU structures (float) ----------------------

struct TriangleDevRefl {
    float v[3][3];       // vertices
    float normal[3];     // unit normal
    float reflectance;   // rho - fraction of light reflected
    float specularity;   // s   - specular fraction of reflected
    float emissivity;    // eps - thermal emission efficiency
    float area;          // triangle area (m^2)
};

struct AABBDevRefl {
    float min[3];
    float max[3];
};

struct BVHNodeDevRefl {
    AABBDevRefl bounds;
    unsigned int triangle_idx;
    unsigned int left, right;
    bool is_leaf;
};

// Constant memory for BVH nodes (separate symbol from original .cu)
#define MAX_REFL_CONSTANT_NODES 1024
__constant__ BVHNodeDevRefl d_refl_constant_nodes[MAX_REFL_CONSTANT_NODES];

// ---------------------- Device helper functions ----------------------

__device__ bool reflRayAABBIntersect(const float origin[3], const float dir[3],
    const AABBDevRefl& box, float& tmin, float& tmax)
{
    tmin = -1e30f; tmax = 1e30f;
    for (int i = 0; i < 3; ++i) {
        if (fabsf(dir[i]) < 1e-10f) {
            if (origin[i] < box.min[i] || origin[i] > box.max[i]) return false;
        }
        else {
            float invD = 1.0f / dir[i];
            float t0 = (box.min[i] - origin[i]) * invD;
            float t1 = (box.max[i] - origin[i]) * invD;
            if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
            if (tmin > tmax + 1e-5f) return false;
        }
    }
    return (tmax >= tmin && tmax > 1e-6f);
}

__device__ bool reflRayTriangleIntersect(const float orig[3], const float dir[3],
    const float v0[3], const float v1[3], const float v2[3],
    float& t)
{
    float edge1[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
    float edge2[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };
    float pvec[3] = {
        dir[1] * edge2[2] - dir[2] * edge2[1],
        dir[2] * edge2[0] - dir[0] * edge2[2],
        dir[0] * edge2[1] - dir[1] * edge2[0]
    };
    float det = edge1[0] * pvec[0] + edge1[1] * pvec[1] + edge1[2] * pvec[2];
    if (fabsf(det) < 1e-7f) return false;

    float invDet = 1.0f / det;
    float tvec[3] = { orig[0] - v0[0], orig[1] - v0[1], orig[2] - v0[2] };
    float u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    float qvec[3] = {
        tvec[1] * edge1[2] - tvec[2] * edge1[1],
        tvec[2] * edge1[0] - tvec[0] * edge1[2],
        tvec[0] * edge1[1] - tvec[1] * edge1[0]
    };
    float v = (dir[0] * qvec[0] + dir[1] * qvec[1] + dir[2] * qvec[2]) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    t = (edge2[0] * qvec[0] + edge2[1] * qvec[1] + edge2[2] * qvec[2]) * invDet;
    return (t > 1e-5f);
}

// ======================== Phase 1 kernels (direct illumination) ========================

__global__ void reflInitLabelsKernel(const TriangleDevRefl* __restrict__ triangles,
    int* __restrict__ labels, size_t N,
    float sunX, float sunY, float sunZ)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float dot_norm = triangles[i].normal[0] * sunX +
                     triangles[i].normal[1] * sunY +
                     triangles[i].normal[2] * sunZ;
    labels[i] = (dot_norm > 1e-6f) ? 1 : 0;
}

// Shadow kernel using global memory for BVH nodes
__global__ void reflShadowKernel_global(const TriangleDevRefl* __restrict__ triangles,
    const BVHNodeDevRefl* __restrict__ nodes,
    const float* __restrict__ centers,
    int* __restrict__ labels, size_t N,
    float sunX, float sunY, float sunZ,
    unsigned int root_idx)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (labels[i] == 0) return;

    float origin[3] = { centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2] };
    float dir[3] = { sunX, sunY, sunZ };

    unsigned int stack[128];
    int sp = 0;
    stack[sp++] = root_idx;

    bool shadowed = false;
    while (sp > 0 && !shadowed) {
        unsigned int nodeIdx = stack[--sp];
        const BVHNodeDevRefl& node = nodes[nodeIdx];

        float tmin, tmax;
        if (!reflRayAABBIntersect(origin, dir, node.bounds, tmin, tmax))
            continue;

        if (node.is_leaf) {
            unsigned int j = node.triangle_idx;
            if (i == j) continue;

            float diff_x = centers[j * 3]     - origin[0];
            float diff_y = centers[j * 3 + 1] - origin[1];
            float diff_z = centers[j * 3 + 2] - origin[2];
            float proj = diff_x * dir[0] + diff_y * dir[1] + diff_z * dir[2];
            if (proj <= 1e-5f) continue;

            float v0[3] = { triangles[j].v[0][0], triangles[j].v[0][1], triangles[j].v[0][2] };
            float v1[3] = { triangles[j].v[1][0], triangles[j].v[1][1], triangles[j].v[1][2] };
            float v2[3] = { triangles[j].v[2][0], triangles[j].v[2][1], triangles[j].v[2][2] };

            float t;
            if (reflRayTriangleIntersect(origin, dir, v0, v1, v2, t)) {
                labels[i] = 0;
                shadowed = true;
            }
        }
        else {
            if (sp < 126) {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }
    }
}

// Shadow kernel using constant memory for BVH nodes
__global__ void reflShadowKernel_constant(const TriangleDevRefl* __restrict__ triangles,
    const float* __restrict__ centers,
    int* __restrict__ labels, size_t N,
    float sunX, float sunY, float sunZ,
    unsigned int root_idx)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (labels[i] == 0) return;

    float origin[3] = { centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2] };
    float dir[3] = { sunX, sunY, sunZ };

    unsigned int stack[128];
    int sp = 0;
    stack[sp++] = root_idx;

    bool shadowed = false;
    while (sp > 0 && !shadowed) {
        unsigned int nodeIdx = stack[--sp];
        const BVHNodeDevRefl& node = d_refl_constant_nodes[nodeIdx];

        float tmin, tmax;
        if (!reflRayAABBIntersect(origin, dir, node.bounds, tmin, tmax))
            continue;

        if (node.is_leaf) {
            unsigned int j = node.triangle_idx;
            if (i == j) continue;

            float diff_x = centers[j * 3]     - origin[0];
            float diff_y = centers[j * 3 + 1] - origin[1];
            float diff_z = centers[j * 3 + 2] - origin[2];
            float proj = diff_x * dir[0] + diff_y * dir[1] + diff_z * dir[2];
            if (proj <= 1e-5f) continue;

            float v0[3] = { triangles[j].v[0][0], triangles[j].v[0][1], triangles[j].v[0][2] };
            float v1[3] = { triangles[j].v[1][0], triangles[j].v[1][1], triangles[j].v[1][2] };
            float v2[3] = { triangles[j].v[2][0], triangles[j].v[2][1], triangles[j].v[2][2] };

            float t;
            if (reflRayTriangleIntersect(origin, dir, v0, v1, v2, t)) {
                labels[i] = 0;
                shadowed = true;
            }
        }
        else {
            if (sp < 126) {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }
    }
}

// ======================== Phase 2 kernels (reflection bounces) ========================

// Generate first-bounce reflected rays from directly-illuminated polygons.
// One thread per polygon; only illuminated ones write a ray.
__global__ void reflGenerateFirstBounceKernel(
    const TriangleDevRefl* __restrict__ triangles,
    const int* __restrict__ labels,
    const float* __restrict__ centers,
    size_t N,
    float sunX, float sunY, float sunZ,
    float default_alpha, float default_mu,
    // output ray buffers
    float* __restrict__ ray_origins,       // [capacity * 3]
    float* __restrict__ ray_dirs,          // [capacity * 3]
    float* __restrict__ ray_intensities,   // [capacity]
    unsigned int* __restrict__ ray_sources,// [capacity]
    unsigned int* __restrict__ ray_count,  // atomic counter
    // bounce tracking
    int* __restrict__ d_bounce_levels,    // [N] set to 0 for directly lit
    float* __restrict__ d_incident_dirs)  // [N*3] sun direction written for directly lit
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (labels[i] == 0) return;

    // Per-triangle optical properties (fall back to defaults for legacy CSV)
    float alpha_i = triangles[i].reflectance;
    float mu_i    = triangles[i].specularity;
    if (alpha_i <= 0.0f && mu_i <= 0.0f) { alpha_i = default_alpha; mu_i = default_mu; }

    float nx = triangles[i].normal[0];
    float ny = triangles[i].normal[1];
    float nz = triangles[i].normal[2];
    float cos_theta = nx * sunX + ny * sunY + nz * sunZ;
    if (cos_theta <= 1e-6f) return;

    // r = 2(n . s)n - s
    float rx = 2.0f * cos_theta * nx - sunX;
    float ry = 2.0f * cos_theta * ny - sunY;
    float rz = 2.0f * cos_theta * nz - sunZ;
    float rlen = sqrtf(rx * rx + ry * ry + rz * rz);
    if (rlen < 1e-12f) return;
    rx /= rlen;  ry /= rlen;  rz /= rlen;

    // Mark this polygon as directly illuminated (bounce 0)
    d_bounce_levels[i] = 0;
    // Record sun direction as the incident direction for bounce=0 visualization
    d_incident_dirs[i * 3 + 0] = sunX;
    d_incident_dirs[i * 3 + 1] = sunY;
    d_incident_dirs[i * 3 + 2] = sunZ;

    unsigned int write_idx = atomicAdd(ray_count, 1u);

    ray_origins[write_idx * 3]     = centers[i * 3];
    ray_origins[write_idx * 3 + 1] = centers[i * 3 + 1];
    ray_origins[write_idx * 3 + 2] = centers[i * 3 + 2];

    ray_dirs[write_idx * 3]     = rx;
    ray_dirs[write_idx * 3 + 1] = ry;
    ray_dirs[write_idx * 3 + 2] = rz;

    // intensity = A_i * cos_theta_i * alpha_i * mu_i
    // Beam cross-section (A_i * cos_theta_i) is conserved through specular bounces,
    // so the host-scalar Phi0 is the only factor added outside the shader.
    float A_i = triangles[i].area;
    if (A_i <= 0.0f) {
        float e1i[3] = { triangles[i].v[1][0] - triangles[i].v[0][0],
                          triangles[i].v[1][1] - triangles[i].v[0][1],
                          triangles[i].v[1][2] - triangles[i].v[0][2] };
        float e2i[3] = { triangles[i].v[2][0] - triangles[i].v[0][0],
                          triangles[i].v[2][1] - triangles[i].v[0][1],
                          triangles[i].v[2][2] - triangles[i].v[0][2] };
        float cri[3] = { e1i[1]*e2i[2] - e1i[2]*e2i[1],
                          e1i[2]*e2i[0] - e1i[0]*e2i[2],
                          e1i[0]*e2i[1] - e1i[1]*e2i[0] };
        A_i = 0.5f * sqrtf(cri[0]*cri[0] + cri[1]*cri[1] + cri[2]*cri[2]);
    }
    ray_intensities[write_idx] = A_i * cos_theta * alpha_i * mu_i;
    ray_sources[write_idx] = (unsigned int)i;
}

// Trace reflection rays, accumulate SRP forces, spawn next-bounce rays.
// One thread per active ray. Uses global memory for BVH nodes.
__global__ void reflTraceAndBounceKernel(
    // current-bounce ray data
    const float* __restrict__ ray_origins,
    const float* __restrict__ ray_dirs,
    const float* __restrict__ ray_intensities,
    const unsigned int* __restrict__ ray_sources,
    unsigned int num_rays,
    // scene data
    const TriangleDevRefl* __restrict__ triangles,
    const BVHNodeDevRefl* __restrict__ nodes,
    const float* __restrict__ centers,
    size_t N,
    unsigned int root_idx,
    // SRP default coefficients (fallback for legacy CSV)
    float default_alpha, float default_mu,
    // accumulated force/moment (atomicAdd)
    float* __restrict__ force_accum,   // [3]
    float* __restrict__ moment_accum,  // [3]
    // next-bounce ray output
    float* __restrict__ next_ray_origins,
    float* __restrict__ next_ray_dirs,
    float* __restrict__ next_ray_intensities,
    unsigned int* __restrict__ next_ray_sources,
    unsigned int* __restrict__ next_ray_count,
    // bounce tracking for visualization
    int bounce_level,                          // current bounce level (1 for first reflected hit, etc.)
    int* __restrict__ d_bounce_levels,         // [N]
    float* __restrict__ d_incident_dirs,       // [N*3]
    float* __restrict__ d_origin_pts)          // [N*3] ray emitter origin for visualization
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) return;

    float origin[3] = { ray_origins[idx * 3], ray_origins[idx * 3 + 1], ray_origins[idx * 3 + 2] };
    float dir[3]    = { ray_dirs[idx * 3],    ray_dirs[idx * 3 + 1],    ray_dirs[idx * 3 + 2] };
    float intensity  = ray_intensities[idx];
    unsigned int source = ray_sources[idx];

    // ---- BVH traversal: find CLOSEST hit ----
    unsigned int stack[128];
    int sp = 0;
    stack[sp++] = root_idx;

    int   closest_idx = -1;
    float closest_t   = 1e30f;

    while (sp > 0) {
        unsigned int nodeIdx = stack[--sp];
        const BVHNodeDevRefl& node = nodes[nodeIdx];

        float tmin, tmax;
        if (!reflRayAABBIntersect(origin, dir, node.bounds, tmin, tmax))
            continue;
        if (tmin > closest_t) continue;  // prune

        if (node.is_leaf) {
            unsigned int j = node.triangle_idx;
            if (j == source) continue;  // self-intersection avoidance

            float t;
            if (reflRayTriangleIntersect(origin, dir,
                    triangles[j].v[0], triangles[j].v[1], triangles[j].v[2], t)) {
                if (t < closest_t) {
                    closest_t   = t;
                    closest_idx = (int)j;
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

    if (closest_idx < 0) return;  // no hit

    unsigned int j = (unsigned int)closest_idx;

    // Check front face: s_eq = -dir (toward source), cos_theta_j = n_j . s_eq
    float s_eq[3] = { -dir[0], -dir[1], -dir[2] };
    float nj[3] = { triangles[j].normal[0], triangles[j].normal[1], triangles[j].normal[2] };
    float cos_theta_j = nj[0] * s_eq[0] + nj[1] * s_eq[1] + nj[2] * s_eq[2];

    if (cos_theta_j <= 1e-6f) return;  // back face

    // ---- Record first-hit bounce level for visualization ----
    // atomicCAS: only writes if current value is still -1 (first hit wins)
    int prev = atomicCAS(&d_bounce_levels[j], -1, bounce_level);
    if (prev == -1) {
        // This thread won - store incident direction (anti-parallel to ray = "from where it came")
        d_incident_dirs[j * 3 + 0] = -dir[0];
        d_incident_dirs[j * 3 + 1] = -dir[1];
        d_incident_dirs[j * 3 + 2] = -dir[2];
        // Store the ray emitter origin so visualizer can draw the full line from emitter to this poly
        d_origin_pts[j * 3 + 0] = origin[0];
        d_origin_pts[j * 3 + 1] = origin[1];
        d_origin_pts[j * 3 + 2] = origin[2];
    }

    // ---- Compute SRP force from reflected light ----
    // Per-triangle optical properties of the HIT triangle
    float alpha_j = triangles[j].reflectance;
    float mu_j    = triangles[j].specularity;
    if (alpha_j <= 0.0f && mu_j <= 0.0f) { alpha_j = default_alpha; mu_j = default_mu; }

    // Use precomputed area if available, otherwise compute from vertices
    float A_j = triangles[j].area;
    if (A_j <= 0.0f) {
        float e1[3] = { triangles[j].v[1][0] - triangles[j].v[0][0],
                         triangles[j].v[1][1] - triangles[j].v[0][1],
                         triangles[j].v[1][2] - triangles[j].v[0][2] };
        float e2[3] = { triangles[j].v[2][0] - triangles[j].v[0][0],
                         triangles[j].v[2][1] - triangles[j].v[0][1],
                         triangles[j].v[2][2] - triangles[j].v[0][2] };
        float cr[3] = { e1[1] * e2[2] - e1[2] * e2[1],
                         e1[2] * e2[0] - e1[0] * e2[2],
                         e1[0] * e2[1] - e1[1] * e2[0] };
        A_j = 0.5f * sqrtf(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
    }

    float base    = intensity;  // intensity already carries A_i*cos_theta_i*chain; conserved through specular bounces //PHY
    float s_coeff = -base * ((1.0f - alpha_j) + alpha_j * (1.0f - mu_j));                           //PHY
    float n_coeff = -base * (2.0f * alpha_j * mu_j * cos_theta_j +                                  //PHY
                             (2.0f / 3.0f) * alpha_j * (1.0f - mu_j));

    float fx = s_coeff * s_eq[0] + n_coeff * nj[0];   //PHY
    float fy = s_coeff * s_eq[1] + n_coeff * nj[1];   //PHY
    float fz = s_coeff * s_eq[2] + n_coeff * nj[2];   //PHY

    atomicAdd(&force_accum[0], fx);   //PHY
    atomicAdd(&force_accum[1], fy);   //PHY
    atomicAdd(&force_accum[2], fz);   //PHY

    // Actual hit point on j - more accurate moment arm than centroid,
    // and correct origin for next-bounce ray.
    float hit_j[3] = { origin[0] + closest_t * dir[0],
                        origin[1] + closest_t * dir[1],
                        origin[2] + closest_t * dir[2] };
    atomicAdd(&moment_accum[0], hit_j[1] * fz - hit_j[2] * fy);   //PHY
    atomicAdd(&moment_accum[1], hit_j[2] * fx - hit_j[0] * fz);   //PHY
    atomicAdd(&moment_accum[2], hit_j[0] * fy - hit_j[1] * fx);   //PHY

    // ---- Spawn next-bounce ray ----
    float new_intensity = intensity * alpha_j * mu_j;
    if (new_intensity < 1e-10f) return;

    // r_next = 2(nj . s_eq) nj - s_eq
    float rnx = 2.0f * cos_theta_j * nj[0] - s_eq[0];
    float rny = 2.0f * cos_theta_j * nj[1] - s_eq[1];
    float rnz = 2.0f * cos_theta_j * nj[2] - s_eq[2];
    float rnlen = sqrtf(rnx * rnx + rny * rny + rnz * rnz);
    if (rnlen < 1e-12f) return;
    rnx /= rnlen;  rny /= rnlen;  rnz /= rnlen;

    unsigned int write_idx = atomicAdd(next_ray_count, 1u);
    next_ray_origins[write_idx * 3]     = hit_j[0];
    next_ray_origins[write_idx * 3 + 1] = hit_j[1];
    next_ray_origins[write_idx * 3 + 2] = hit_j[2];
    next_ray_dirs[write_idx * 3]     = rnx;
    next_ray_dirs[write_idx * 3 + 1] = rny;
    next_ray_dirs[write_idx * 3 + 2] = rnz;
    next_ray_intensities[write_idx] = new_intensity;
    next_ray_sources[write_idx] = j;
}


// ====================== HOST function with GPU cache ======================

struct ReflGPUCache {
    TriangleDevRefl* d_tri = nullptr;
    float* d_centers = nullptr;
    BVHNodeDevRefl* d_nodes = nullptr;
    int* d_labels = nullptr;
    unsigned int root_idx = 0;
    size_t N = 0;
    size_t numNodes = 0;
    bool useConstant = false;
    bool valid = false;
    cudaStream_t stream = nullptr;
    int* h_pinned_labels = nullptr;

    // Reflection ray double buffers (each of size N)
    float* d_ray_origins_A = nullptr;
    float* d_ray_dirs_A    = nullptr;
    float* d_ray_intensities_A = nullptr;
    unsigned int* d_ray_sources_A = nullptr;

    float* d_ray_origins_B = nullptr;
    float* d_ray_dirs_B    = nullptr;
    float* d_ray_intensities_B = nullptr;
    unsigned int* d_ray_sources_B = nullptr;

    unsigned int* d_ray_count_A = nullptr;
    unsigned int* d_ray_count_B = nullptr;

    float* d_force_accum  = nullptr;  // [3]
    float* d_moment_accum = nullptr;  // [3]

    // Per-polygon bounce tracking for visualization
    int*   d_bounce_levels = nullptr;  // [N]  -1=never, 0=direct, 1+=reflection
    float* d_incident_dirs = nullptr;  // [N*3] incident direction for bounce=0+ polys
    float* d_origin_pts    = nullptr;  // [N*3] ray emitter origin (NaN = unknown, i.e. direct sun)
};

static ReflGPUCache g_refl_cache;


SRPResult calculate_labels_ray_casting_reflections_gpu(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    int max_reflections,
    bool verbose,
    const std::string& primary_emitter_name)
{
    if (triangles.empty())
        return {};

    const float default_alpha = 0.5f;
    const float default_mu    = 0.5f;

    double len = sqrt(sun_vector[0] * sun_vector[0] +
                      sun_vector[1] * sun_vector[1] +
                      sun_vector[2] * sun_vector[2]);
    if (len < 1e-12) throw std::runtime_error("Sun vector is zero");
    double d_sunX = sun_vector[0] / len;
    double d_sunY = sun_vector[1] / len;
    double d_sunZ = sun_vector[2] / len;
    float sunX = static_cast<float>(d_sunX);
    float sunY = static_cast<float>(d_sunY);
    float sunZ = static_cast<float>(d_sunZ);

    size_t N = triangles.size();

    // =============== Rebuild cache if needed ===============
    if (!g_refl_cache.valid || g_refl_cache.N != N) {
        // Free old allocations
        if (g_refl_cache.d_tri)     REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_tri));
        if (g_refl_cache.d_centers) REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_centers));
        if (g_refl_cache.d_nodes)   REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_nodes));
        if (g_refl_cache.d_labels)  REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_labels));
        if (g_refl_cache.h_pinned_labels) REFL_CUDA_CHECK(cudaFreeHost(g_refl_cache.h_pinned_labels));
        if (g_refl_cache.stream)    REFL_CUDA_CHECK(cudaStreamDestroy(g_refl_cache.stream));

        // Free reflection buffers
        if (g_refl_cache.d_ray_origins_A)     REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_ray_origins_A));
        if (g_refl_cache.d_ray_dirs_A)        REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_ray_dirs_A));
        if (g_refl_cache.d_ray_intensities_A) REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_ray_intensities_A));
        if (g_refl_cache.d_ray_sources_A)     REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_ray_sources_A));
        if (g_refl_cache.d_ray_origins_B)     REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_ray_origins_B));
        if (g_refl_cache.d_ray_dirs_B)        REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_ray_dirs_B));
        if (g_refl_cache.d_ray_intensities_B) REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_ray_intensities_B));
        if (g_refl_cache.d_ray_sources_B)     REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_ray_sources_B));
        if (g_refl_cache.d_ray_count_A)       REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_ray_count_A));
        if (g_refl_cache.d_ray_count_B)       REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_ray_count_B));
        if (g_refl_cache.d_force_accum)       REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_force_accum));
        if (g_refl_cache.d_moment_accum)      REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_moment_accum));
        if (g_refl_cache.d_bounce_levels)     REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_bounce_levels));
        if (g_refl_cache.d_incident_dirs)     REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_incident_dirs));
        if (g_refl_cache.d_origin_pts)        REFL_CUDA_CHECK(cudaFree(g_refl_cache.d_origin_pts));

        REFL_CUDA_CHECK(cudaStreamCreate(&g_refl_cache.stream));
        REFL_CUDA_CHECK(cudaHostAlloc(&g_refl_cache.h_pinned_labels, N * sizeof(int), cudaHostAllocDefault));

        // ----- Build BVH on host (identical to original) -----
        std::vector<std::array<double, 3>> centers_cpu(N);
        for (size_t i = 0; i < N; ++i) {
            double cx = (triangles[i].v1_x + triangles[i].v2_x + triangles[i].v3_x) / 3.0;
            double cy = (triangles[i].v1_y + triangles[i].v2_y + triangles[i].v3_y) / 3.0;
            double cz = (triangles[i].v1_z + triangles[i].v2_z + triangles[i].v3_z) / 3.0;
            centers_cpu[i] = { cx, cy, cz };
        }

        struct AABB { double min_x, min_y, min_z, max_x, max_y, max_z; };
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

        struct BVHNode {
            AABB bounds;
            size_t triangle_idx;
            size_t left, right;
            bool is_leaf;
        };
        std::vector<BVHNode> nodes;
        std::vector<size_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);

        std::function<size_t(size_t, size_t, size_t)> build_bvh;
        build_bvh = [&](size_t start, size_t end, size_t depth) -> size_t {
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

            std::sort(indices.begin() + start, indices.begin() + end,
                [&](size_t a, size_t b) { return centers_cpu[a][axis] < centers_cpu[b][axis]; });

            size_t mid = (start + end) / 2;

            node.is_leaf = false;
            node.left  = build_bvh(start, mid, depth + 1);
            node.right = build_bvh(mid,   end, depth + 1);

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

        size_t root = build_bvh(0, N, 0);
        g_refl_cache.root_idx = static_cast<unsigned int>(root);
        g_refl_cache.numNodes = nodes.size();

        // Compute and warn about BVH depth
        std::function<size_t(size_t)> computeDepth;
        computeDepth = [&](size_t idx) -> size_t {
            if (nodes[idx].is_leaf) return 1;
            return 1 + std::max(computeDepth(nodes[idx].left), computeDepth(nodes[idx].right));
        };
        size_t maxDepth = computeDepth(root);
        if (maxDepth > 126) {
            fprintf(stderr, "Warning: BVH depth %zu exceeds stack size 128. Results may be incorrect.\n", maxDepth);
        }

        // ----- Prepare GPU data -----
        std::vector<TriangleDevRefl> triDev(N);
        for (size_t i = 0; i < N; ++i) {
            triDev[i].v[0][0] = static_cast<float>(triangles[i].v1_x);
            triDev[i].v[0][1] = static_cast<float>(triangles[i].v1_y);
            triDev[i].v[0][2] = static_cast<float>(triangles[i].v1_z);
            triDev[i].v[1][0] = static_cast<float>(triangles[i].v2_x);
            triDev[i].v[1][1] = static_cast<float>(triangles[i].v2_y);
            triDev[i].v[1][2] = static_cast<float>(triangles[i].v2_z);
            triDev[i].v[2][0] = static_cast<float>(triangles[i].v3_x);
            triDev[i].v[2][1] = static_cast<float>(triangles[i].v3_y);
            triDev[i].v[2][2] = static_cast<float>(triangles[i].v3_z);
            triDev[i].normal[0] = static_cast<float>(triangles[i].normal_x);
            triDev[i].normal[1] = static_cast<float>(triangles[i].normal_y);
            triDev[i].normal[2] = static_cast<float>(triangles[i].normal_z);
            triDev[i].reflectance = static_cast<float>(triangles[i].reflectance);
            triDev[i].specularity = static_cast<float>(triangles[i].specularity);
            triDev[i].emissivity  = static_cast<float>(triangles[i].emissivity);
            triDev[i].area        = static_cast<float>(triangles[i].area);
        }

        std::vector<float> centers_flat(N * 3);
        for (size_t i = 0; i < N; ++i) {
            centers_flat[i * 3]     = static_cast<float>(centers_cpu[i][0]);
            centers_flat[i * 3 + 1] = static_cast<float>(centers_cpu[i][1]);
            centers_flat[i * 3 + 2] = static_cast<float>(centers_cpu[i][2]);
        }

        std::vector<BVHNodeDevRefl> nodesDev(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            nodesDev[i].bounds.min[0] = static_cast<float>(nodes[i].bounds.min_x);
            nodesDev[i].bounds.min[1] = static_cast<float>(nodes[i].bounds.min_y);
            nodesDev[i].bounds.min[2] = static_cast<float>(nodes[i].bounds.min_z);
            nodesDev[i].bounds.max[0] = static_cast<float>(nodes[i].bounds.max_x);
            nodesDev[i].bounds.max[1] = static_cast<float>(nodes[i].bounds.max_y);
            nodesDev[i].bounds.max[2] = static_cast<float>(nodes[i].bounds.max_z);
            nodesDev[i].triangle_idx  = static_cast<unsigned int>(nodes[i].triangle_idx);
            nodesDev[i].left          = static_cast<unsigned int>(nodes[i].left);
            nodesDev[i].right         = static_cast<unsigned int>(nodes[i].right);
            nodesDev[i].is_leaf       = nodes[i].is_leaf;
        }

        // Allocate and copy scene data
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_tri,     N * sizeof(TriangleDevRefl)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_centers, N * 3 * sizeof(float)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_nodes,   nodes.size() * sizeof(BVHNodeDevRefl)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_labels,  N * sizeof(int)));

        REFL_CUDA_CHECK(cudaMemset(g_refl_cache.d_labels, 0, N * sizeof(int)));

        REFL_CUDA_CHECK(cudaMemcpy(g_refl_cache.d_tri,     triDev.data(),       N * sizeof(TriangleDevRefl),               cudaMemcpyHostToDevice));
        REFL_CUDA_CHECK(cudaMemcpy(g_refl_cache.d_centers, centers_flat.data(),  N * 3 * sizeof(float),                     cudaMemcpyHostToDevice));
        REFL_CUDA_CHECK(cudaMemcpy(g_refl_cache.d_nodes,   nodesDev.data(),      nodes.size() * sizeof(BVHNodeDevRefl),     cudaMemcpyHostToDevice));

        // Constant memory path
        size_t nodeSize = sizeof(BVHNodeDevRefl);
        size_t constantMemLimit = 64 * 1024;
        g_refl_cache.useConstant = (nodes.size() * nodeSize <= constantMemLimit);
        if (g_refl_cache.useConstant) {
            REFL_CUDA_CHECK(cudaMemcpyToSymbol(d_refl_constant_nodes, nodesDev.data(), nodes.size() * nodeSize));
        }

        // Allocate reflection ray double buffers (capacity = N)
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_ray_origins_A,     N * 3 * sizeof(float)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_ray_dirs_A,        N * 3 * sizeof(float)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_ray_intensities_A, N * sizeof(float)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_ray_sources_A,     N * sizeof(unsigned int)));

        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_ray_origins_B,     N * 3 * sizeof(float)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_ray_dirs_B,        N * 3 * sizeof(float)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_ray_intensities_B, N * sizeof(float)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_ray_sources_B,     N * sizeof(unsigned int)));

        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_ray_count_A, sizeof(unsigned int)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_ray_count_B, sizeof(unsigned int)));

        // Force/moment accumulators
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_force_accum,  3 * sizeof(float)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_moment_accum, 3 * sizeof(float)));

        // Bounce-level tracking (visualization)
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_bounce_levels, N * sizeof(int)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_incident_dirs, N * 3 * sizeof(float)));
        REFL_CUDA_CHECK(cudaMalloc(&g_refl_cache.d_origin_pts,    N * 3 * sizeof(float)));

        g_refl_cache.N = N;
        g_refl_cache.valid = true;
    }

    // =============== Phase 1: Direct illumination (identical to original) ===============
    int blockSize, minGrid;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, reflShadowKernel_global, 0, 0);
    int gridSize = (int)((N + blockSize - 1) / blockSize);

    reflInitLabelsKernel<<<gridSize, blockSize, 0, g_refl_cache.stream>>>(
        g_refl_cache.d_tri, g_refl_cache.d_labels, N, sunX, sunY, sunZ);
    REFL_CUDA_CHECK(cudaGetLastError());

    if (g_refl_cache.useConstant) {
        reflShadowKernel_constant<<<gridSize, blockSize, 0, g_refl_cache.stream>>>(
            g_refl_cache.d_tri, g_refl_cache.d_centers, g_refl_cache.d_labels, N,
            sunX, sunY, sunZ, g_refl_cache.root_idx);
    }
    else {
        reflShadowKernel_global<<<gridSize, blockSize, 0, g_refl_cache.stream>>>(
            g_refl_cache.d_tri, g_refl_cache.d_nodes, g_refl_cache.d_centers, g_refl_cache.d_labels, N,
            sunX, sunY, sunZ, g_refl_cache.root_idx);
    }
    REFL_CUDA_CHECK(cudaGetLastError());

    // Copy labels back (async)
    REFL_CUDA_CHECK(cudaMemcpyAsync(g_refl_cache.h_pinned_labels, g_refl_cache.d_labels, N * sizeof(int),
        cudaMemcpyDeviceToHost, g_refl_cache.stream));

    // If a primary emitter is specified, restrict direct illumination to that component only.
    // Sync the stream so h_pinned_labels is ready, apply the filter on CPU,
    // then copy the filtered labels back to the device before Phase 2.
    if (!primary_emitter_name.empty()) {
        REFL_CUDA_CHECK(cudaStreamSynchronize(g_refl_cache.stream));
        for (size_t i = 0; i < N; ++i) {
            if (g_refl_cache.h_pinned_labels[i] == 1 &&
                triangles[i].component_name != primary_emitter_name)
                g_refl_cache.h_pinned_labels[i] = 0;
        }
        REFL_CUDA_CHECK(cudaMemcpyAsync(g_refl_cache.d_labels, g_refl_cache.h_pinned_labels,
            N * sizeof(int), cudaMemcpyHostToDevice, g_refl_cache.stream));
    }

    // =============== Phase 2: Multi-bounce reflections ===============
    REFL_CUDA_CHECK(cudaMemsetAsync(g_refl_cache.d_force_accum,  0, 3 * sizeof(float), g_refl_cache.stream));
    REFL_CUDA_CHECK(cudaMemsetAsync(g_refl_cache.d_moment_accum, 0, 3 * sizeof(float), g_refl_cache.stream));

    if (max_reflections > 0) {
        // Zero bounce-tracking arrays only when reflections are actually computed.
        // Skipping these for max_reflections==0 saves ~150 MB of device writes for
        // large models (e.g. 5.4M triangles → 3×N×sizeof = 151 MB).
        // 0xFF → 0xFFFFFFFF = -1 for int32 ("never hit"); float NaN for dirs/origins.
        REFL_CUDA_CHECK(cudaMemsetAsync(g_refl_cache.d_bounce_levels, 0xFF, N * sizeof(int),       g_refl_cache.stream));
        REFL_CUDA_CHECK(cudaMemsetAsync(g_refl_cache.d_incident_dirs, 0,    N * 3 * sizeof(float), g_refl_cache.stream));
        REFL_CUDA_CHECK(cudaMemsetAsync(g_refl_cache.d_origin_pts,    0xFF, N * 3 * sizeof(float), g_refl_cache.stream));
        // Generate first-bounce rays (also marks directly lit polys as bounce=0)
        REFL_CUDA_CHECK(cudaMemsetAsync(g_refl_cache.d_ray_count_A, 0, sizeof(unsigned int), g_refl_cache.stream));

        reflGenerateFirstBounceKernel<<<gridSize, blockSize, 0, g_refl_cache.stream>>>(
            g_refl_cache.d_tri, g_refl_cache.d_labels, g_refl_cache.d_centers, N,
            sunX, sunY, sunZ, default_alpha, default_mu,
            g_refl_cache.d_ray_origins_A, g_refl_cache.d_ray_dirs_A,
            g_refl_cache.d_ray_intensities_A, g_refl_cache.d_ray_sources_A,
            g_refl_cache.d_ray_count_A,
            g_refl_cache.d_bounce_levels,
            g_refl_cache.d_incident_dirs);
        REFL_CUDA_CHECK(cudaGetLastError());

        // Pointers for double buffering
        float* cur_origins      = g_refl_cache.d_ray_origins_A;
        float* cur_dirs         = g_refl_cache.d_ray_dirs_A;
        float* cur_intensities  = g_refl_cache.d_ray_intensities_A;
        unsigned int* cur_sources = g_refl_cache.d_ray_sources_A;
        unsigned int* cur_count   = g_refl_cache.d_ray_count_A;

        float* next_origins     = g_refl_cache.d_ray_origins_B;
        float* next_dirs        = g_refl_cache.d_ray_dirs_B;
        float* next_intensities = g_refl_cache.d_ray_intensities_B;
        unsigned int* next_sources = g_refl_cache.d_ray_sources_B;
        unsigned int* next_count   = g_refl_cache.d_ray_count_B;

        for (int bounce = 0; bounce < max_reflections; ++bounce) {
            // Read current ray count from device
            unsigned int h_ray_count = 0;
            REFL_CUDA_CHECK(cudaMemcpyAsync(&h_ray_count, cur_count, sizeof(unsigned int),
                cudaMemcpyDeviceToHost, g_refl_cache.stream));
            REFL_CUDA_CHECK(cudaStreamSynchronize(g_refl_cache.stream));

            if (h_ray_count == 0) break;

            if (verbose) {

                if (bounce == 0)
                    std::cout << "  GPU reflection bounce 0 (direct): " << h_ray_count << " rays\n";
                else
                    std::cout << "  GPU reflection bounce " << bounce << ": " << h_ray_count << " rays\n";
            }

            // Reset next-bounce counter
            REFL_CUDA_CHECK(cudaMemsetAsync(next_count, 0, sizeof(unsigned int), g_refl_cache.stream));

            // Launch trace-and-bounce kernel
            int reflBlockSize, reflMinGrid;
            cudaOccupancyMaxPotentialBlockSize(&reflMinGrid, &reflBlockSize, reflTraceAndBounceKernel, 0, 0);
            int reflGridSize = (int)((h_ray_count + reflBlockSize - 1) / reflBlockSize);

            reflTraceAndBounceKernel<<<reflGridSize, reflBlockSize, 0, g_refl_cache.stream>>>(
                cur_origins, cur_dirs, cur_intensities, cur_sources,
                h_ray_count,
                g_refl_cache.d_tri, g_refl_cache.d_nodes, g_refl_cache.d_centers, N,
                g_refl_cache.root_idx,
                default_alpha, default_mu,
                g_refl_cache.d_force_accum, g_refl_cache.d_moment_accum,
                next_origins, next_dirs, next_intensities, next_sources, next_count,
                bounce + 1,                      // bounce_level: 1 for first hit, 2 for second, ...
                g_refl_cache.d_bounce_levels,
                g_refl_cache.d_incident_dirs,
                g_refl_cache.d_origin_pts);
            REFL_CUDA_CHECK(cudaGetLastError());

            // Swap buffers
            std::swap(cur_origins, next_origins);
            std::swap(cur_dirs, next_dirs);
            std::swap(cur_intensities, next_intensities);
            std::swap(cur_sources, next_sources);
            std::swap(cur_count, next_count);
        }
    }

    // =============== Collect results ===============
    REFL_CUDA_CHECK(cudaStreamSynchronize(g_refl_cache.stream));

    // Populate visualization globals (bounce levels, incident dirs, emitter origins).
    // When max_reflections==0 there are no bounces: skip the two large GPU→host copies
    // (~150 MB for large models) and build the data from the host labels array directly.
    {
        const double dNaN = std::numeric_limits<double>::quiet_NaN();
        std::vector<int>                   bounce_levels(N);
        std::vector<std::array<double, 3>> incident_dirs(N, {0.0, 0.0, 0.0});
        std::vector<std::array<double, 3>> origin_pts(N, {dNaN, dNaN, dNaN});

        if (max_reflections > 0) {
            // Full GPU→host copy: bounce-tracking arrays were populated by the kernels.
            std::vector<float> h_inc(N * 3);
            std::vector<float> h_orig(N * 3);
            REFL_CUDA_CHECK(cudaMemcpy(bounce_levels.data(), g_refl_cache.d_bounce_levels, N * sizeof(int),       cudaMemcpyDeviceToHost));
            REFL_CUDA_CHECK(cudaMemcpy(h_inc.data(),         g_refl_cache.d_incident_dirs, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
            REFL_CUDA_CHECK(cudaMemcpy(h_orig.data(),        g_refl_cache.d_origin_pts,    N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < N; ++i) {
                incident_dirs[i] = { static_cast<double>(h_inc[i*3]),   static_cast<double>(h_inc[i*3+1]),  static_cast<double>(h_inc[i*3+2])  };
                origin_pts[i]    = { static_cast<double>(h_orig[i*3]),  static_cast<double>(h_orig[i*3+1]), static_cast<double>(h_orig[i*3+2]) };
            }
        } else {
            // No reflections: derive bounce_levels from labels already on the host.
            // Lit triangles are direct (bounce=0); shadowed triangles are -1.
            // incident_dirs stays zero-initialized; origin_pts stays NaN.
            // h_pinned_labels is a raw int* (pinned host memory), already synced above.
            for (size_t i = 0; i < N; ++i) {
                bounce_levels[i] = (g_refl_cache.h_pinned_labels[i] == 1) ? 0 : -1;
            }
        }
        set_bounce_globals(std::move(bounce_levels), std::move(incident_dirs), std::move(origin_pts));
    }

    std::vector<int> labels_host(g_refl_cache.h_pinned_labels, g_refl_cache.h_pinned_labels + N);

    // Compute direct SRP on host (same as original)
    SRPResult result = compute_srp_forces(triangles, std::move(labels_host), d_sunX, d_sunY, d_sunZ);

    // Add reflection contributions
    if (max_reflections > 0) {
        float h_force[3]  = {0, 0, 0};
        float h_moment[3] = {0, 0, 0};
        REFL_CUDA_CHECK(cudaMemcpy(h_force,  g_refl_cache.d_force_accum,  3 * sizeof(float), cudaMemcpyDeviceToHost));
        REFL_CUDA_CHECK(cudaMemcpy(h_moment, g_refl_cache.d_moment_accum, 3 * sizeof(float), cudaMemcpyDeviceToHost));

        // GPU kernel accumulated forces with phi0=1 implicitly; scale by g_srp_phi0 here.
        result.total_force[0]  += g_srp_phi0 * static_cast<double>(h_force[0]);
        result.total_force[1]  += g_srp_phi0 * static_cast<double>(h_force[1]);
        result.total_force[2]  += g_srp_phi0 * static_cast<double>(h_force[2]);
        result.total_moment[0] += g_srp_phi0 * static_cast<double>(h_moment[0]);
        result.total_moment[1] += g_srp_phi0 * static_cast<double>(h_moment[1]);
        result.total_moment[2] += g_srp_phi0 * static_cast<double>(h_moment[2]);
    }

    return result;
}
