// ---------------------------------------------------------------------------
// gpu_reflection_raytracer_bench.cu
// Benchmark-only lean copy of gpu_reflection_raytracer.cu.
//
// Removed vs original:
//   - d_bounce_levels / d_incident_dirs / d_origin_pts GPU arrays (never allocated)
//   - bounce tracking parameters from both Phase 2 kernels (no atomicCAS)
//   - set_bounce_globals() call
//   - primary_emitter_name parameter
//   - Reflection ray buffers NOT allocated when max_reflections==0 (saves ~346 MB
//     for 5.4M triangle model)
//
// When max_reflections==0 cost = GPU shadow pass + reflDirectForceBenchKernel.
// No CPU force computation; only 48 bytes (6×double) transferred to host.
//
// All __global__ kernels have "Bench" suffix to avoid linker conflicts.
// __constant__ symbol renamed to d_refl_bench_constant_nodes.
//
// Exports: calculate_labels_reflections_gpu_bench()
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "SatelliteDataset.h"
#include "ShadowAlgorithms.h"
#include "bench_methods.h"
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <functional>

#define REFL_BENCH_CUDA_CHECK(ans) { reflBenchGpuAssert((ans), __FILE__, __LINE__); }
inline void reflBenchGpuAssert(cudaError_t code, const char* file, int line, bool abort_=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort_) exit(code);
    }
}

// ---------------------- GPU structures (float, same as original) ----------------------
struct TriangleDevReflBench {
    float v[3][3];
    float normal[3];
    float reflectance;
    float specularity;
    float emissivity;
    float area;
};

struct AABBDevReflBench {
    float min[3];
    float max[3];
};

struct BVHNodeDevReflBench {
    AABBDevReflBench bounds;
    unsigned int triangle_idx;
    unsigned int left, right;
    bool is_leaf;
};

// Renamed constant to avoid symbol conflict with gpu_reflection_raytracer.cu
#define MAX_REFL_BENCH_CONSTANT_NODES 1024
__constant__ BVHNodeDevReflBench d_refl_bench_constant_nodes[MAX_REFL_BENCH_CONSTANT_NODES];

// ---------------------- Device helper functions (static = TU-local) ----------------------
__device__ static inline bool reflBenchRayAABBIntersect(
    const float origin[3], const float dir[3],
    const AABBDevReflBench& box, float& tmin, float& tmax)
{
    tmin = -1e30f; tmax = 1e30f;
    for (int i = 0; i < 3; ++i) {
        if (fabsf(dir[i]) < 1e-10f) {
            if (origin[i] < box.min[i] || origin[i] > box.max[i]) return false;
        } else {
            float invD = 1.0f / dir[i];
            float t1 = (box.min[i] - origin[i]) * invD;
            float t2 = (box.max[i] - origin[i]) * invD;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            tmin = fmaxf(tmin, t1);
            tmax = fminf(tmax, t2);
            if (tmin > tmax) return false;
        }
    }
    return tmax > 0.0f;
}

__device__ static inline bool reflBenchRayTriangleIntersect(
    const float origin[3], const float dir[3],
    const float v0[3], const float v1[3], const float v2[3], float& t)
{
    const float EPS = 1e-8f;
    float e1[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
    float e2[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
    float h[3] = { dir[1]*e2[2]-dir[2]*e2[1], dir[2]*e2[0]-dir[0]*e2[2], dir[0]*e2[1]-dir[1]*e2[0] };
    float a = e1[0]*h[0] + e1[1]*h[1] + e1[2]*h[2];
    if (fabsf(a) < EPS) return false;
    float f = 1.0f / a;
    float s[3] = { origin[0]-v0[0], origin[1]-v0[1], origin[2]-v0[2] };
    float u = f * (s[0]*h[0] + s[1]*h[1] + s[2]*h[2]);
    if (u < 0.0f || u > 1.0f) return false;
    float q[3] = { s[1]*e1[2]-s[2]*e1[1], s[2]*e1[0]-s[0]*e1[2], s[0]*e1[1]-s[1]*e1[0] };
    float v = f * (dir[0]*q[0] + dir[1]*q[1] + dir[2]*q[2]);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = f * (e2[0]*q[0] + e2[1]*q[1] + e2[2]*q[2]);
    return t > EPS;
}

// ===================== Phase 1 kernels (renamed) =====================

__global__ void reflInitLabelsBenchKernel(
    const TriangleDevReflBench* __restrict__ triangles,
    int* __restrict__ labels, size_t N,
    float sunX, float sunY, float sunZ)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float dot_norm = triangles[i].normal[0]*sunX +
                     triangles[i].normal[1]*sunY +
                     triangles[i].normal[2]*sunZ;
    labels[i] = (dot_norm > 1e-5f) ? 1 : 0;   // 1e-5f matches RTX backface_filter  //PHY
}

__global__ void reflShadowBenchKernel_global(
    const TriangleDevReflBench* __restrict__ triangles,
    const BVHNodeDevReflBench* __restrict__ nodes,
    const float* __restrict__ centers,
    int* __restrict__ labels, size_t N,
    float sunX, float sunY, float sunZ,
    unsigned int root_idx)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (labels[i] == 0) return;

    // Offset origin along face normal — prevents false self-shadow from adjacent panels  //PHY
    float snx = triangles[i].normal[0], sny = triangles[i].normal[1], snz = triangles[i].normal[2];
    float origin[3] = { centers[i*3]   + 1e-4f*snx,
                        centers[i*3+1] + 1e-4f*sny,
                        centers[i*3+2] + 1e-4f*snz };
    float dir[3] = { sunX, sunY, sunZ };

    unsigned int stack[128];
    int sp = 0;
    stack[sp++] = root_idx;

    bool shadowed = false;
    while (sp > 0 && !shadowed) {
        unsigned int nodeIdx = stack[--sp];
        const BVHNodeDevReflBench& node = nodes[nodeIdx];
        float tmin, tmax;
        if (!reflBenchRayAABBIntersect(origin, dir, node.bounds, tmin, tmax)) continue;
        if (node.is_leaf) {
            unsigned int j = node.triangle_idx;
            if (i == j) continue;
            float diff[3] = { centers[j*3]-origin[0], centers[j*3+1]-origin[1], centers[j*3+2]-origin[2] };
            float proj = diff[0]*dir[0]+diff[1]*dir[1]+diff[2]*dir[2];
            if (proj <= 1e-5f) continue;
            float v0[3]={triangles[j].v[0][0],triangles[j].v[0][1],triangles[j].v[0][2]};
            float v1[3]={triangles[j].v[1][0],triangles[j].v[1][1],triangles[j].v[1][2]};
            float v2[3]={triangles[j].v[2][0],triangles[j].v[2][1],triangles[j].v[2][2]};
            float t;
            if (reflBenchRayTriangleIntersect(origin, dir, v0, v1, v2, t)) { labels[i]=0; shadowed=true; }
        } else {
            if (sp < 126) { stack[sp++]=node.left; stack[sp++]=node.right; }
        }
    }
}

__global__ void reflShadowBenchKernel_constant(
    const TriangleDevReflBench* __restrict__ triangles,
    const float* __restrict__ centers,
    int* __restrict__ labels, size_t N,
    float sunX, float sunY, float sunZ,
    unsigned int root_idx)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (labels[i] == 0) return;

    // Offset origin along face normal — prevents false self-shadow from adjacent panels  //PHY
    float snx = triangles[i].normal[0], sny = triangles[i].normal[1], snz = triangles[i].normal[2];
    float origin[3] = { centers[i*3]   + 1e-4f*snx,
                        centers[i*3+1] + 1e-4f*sny,
                        centers[i*3+2] + 1e-4f*snz };
    float dir[3] = { sunX, sunY, sunZ };

    unsigned int stack[128];
    int sp = 0;
    stack[sp++] = root_idx;

    bool shadowed = false;
    while (sp > 0 && !shadowed) {
        unsigned int nodeIdx = stack[--sp];
        const BVHNodeDevReflBench& node = d_refl_bench_constant_nodes[nodeIdx];
        float tmin, tmax;
        if (!reflBenchRayAABBIntersect(origin, dir, node.bounds, tmin, tmax)) continue;
        if (node.is_leaf) {
            unsigned int j = node.triangle_idx;
            if (i == j) continue;
            float diff[3] = { centers[j*3]-origin[0], centers[j*3+1]-origin[1], centers[j*3+2]-origin[2] };
            float proj = diff[0]*dir[0]+diff[1]*dir[1]+diff[2]*dir[2];
            if (proj <= 1e-5f) continue;
            float v0[3]={triangles[j].v[0][0],triangles[j].v[0][1],triangles[j].v[0][2]};
            float v1[3]={triangles[j].v[1][0],triangles[j].v[1][1],triangles[j].v[1][2]};
            float v2[3]={triangles[j].v[2][0],triangles[j].v[2][1],triangles[j].v[2][2]};
            float t;
            if (reflBenchRayTriangleIntersect(origin, dir, v0, v1, v2, t)) { labels[i]=0; shadowed=true; }
        } else {
            if (sp < 126) { stack[sp++]=node.left; stack[sp++]=node.right; }
        }
    }
}

// ===================== Phase 2 kernels (lean - no bounce tracking) =====================

__global__ void reflGenerateFirstBounceBenchKernel(
    const TriangleDevReflBench* __restrict__ triangles,
    const int* __restrict__ labels,
    const float* __restrict__ centers,
    size_t N,
    float sunX, float sunY, float sunZ,
    float default_alpha, float default_mu,
    float* __restrict__ ray_origins,
    float* __restrict__ ray_dirs,
    float* __restrict__ ray_intensities,
    unsigned int* __restrict__ ray_sources,
    unsigned int* __restrict__ ray_count)
    // NOTE: no d_bounce_levels / d_incident_dirs params
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (labels[i] == 0) return;

    float alpha_i = triangles[i].reflectance;
    float mu_i    = triangles[i].specularity;
    if (alpha_i <= 0.0f && mu_i <= 0.0f) { alpha_i = default_alpha; mu_i = default_mu; }

    float nx = triangles[i].normal[0], ny = triangles[i].normal[1], nz = triangles[i].normal[2];
    float cos_theta = nx*sunX + ny*sunY + nz*sunZ;
    if (cos_theta <= 1e-5f) return;   // 1e-5f matches RTX backface_filter  //PHY

    float rx = 2.0f*cos_theta*nx - sunX;
    float ry = 2.0f*cos_theta*ny - sunY;
    float rz = 2.0f*cos_theta*nz - sunZ;
    float rlen = sqrtf(rx*rx + ry*ry + rz*rz);
    if (rlen < 1e-12f) return;
    rx/=rlen; ry/=rlen; rz/=rlen;

    // NO bounce tracking writes
    float A_i = triangles[i].area;
    if (A_i <= 0.0f) {
        float e1[3] = { triangles[i].v[1][0]-triangles[i].v[0][0],
                        triangles[i].v[1][1]-triangles[i].v[0][1],
                        triangles[i].v[1][2]-triangles[i].v[0][2] };
        float e2[3] = { triangles[i].v[2][0]-triangles[i].v[0][0],
                        triangles[i].v[2][1]-triangles[i].v[0][1],
                        triangles[i].v[2][2]-triangles[i].v[0][2] };
        float cr[3] = { e1[1]*e2[2]-e1[2]*e2[1],
                        e1[2]*e2[0]-e1[0]*e2[2],
                        e1[0]*e2[1]-e1[1]*e2[0] };
        A_i = 0.5f * sqrtf(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
    }
    unsigned int write_idx = atomicAdd(ray_count, 1u);
    // Offset first-bounce origin by 1e-4 along source normal to avoid self-intersection  //FIX-A
    ray_origins[write_idx*3]   = centers[i*3]   + 1e-4f*nx;
    ray_origins[write_idx*3+1] = centers[i*3+1] + 1e-4f*ny;
    ray_origins[write_idx*3+2] = centers[i*3+2] + 1e-4f*nz;
    ray_dirs[write_idx*3]   = rx;
    ray_dirs[write_idx*3+1] = ry;
    ray_dirs[write_idx*3+2] = rz;
    // Beam cross-section A_i*cos_theta_i is conserved through specular bounces  //PHY
    ray_intensities[write_idx] = A_i * cos_theta * alpha_i * mu_i;               //PHY
    ray_sources[write_idx] = (unsigned int)i;
}

__global__ void reflTraceAndBounceBenchKernel(
    const float* __restrict__ ray_origins,
    const float* __restrict__ ray_dirs,
    const float* __restrict__ ray_intensities,
    const unsigned int* __restrict__ ray_sources,
    unsigned int num_rays,
    const TriangleDevReflBench* __restrict__ triangles,
    const BVHNodeDevReflBench* __restrict__ nodes,
    const float* __restrict__ centers,
    size_t N,
    unsigned int root_idx,
    float default_alpha, float default_mu,
    double* __restrict__ force_accum,
    double* __restrict__ moment_accum,
    float* __restrict__ next_ray_origins,
    float* __restrict__ next_ray_dirs,
    float* __restrict__ next_ray_intensities,
    unsigned int* __restrict__ next_ray_sources,
    unsigned int* __restrict__ next_ray_count)
    // NOTE: no bounce_level / d_bounce_levels / d_incident_dirs / d_origin_pts params
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) return;

    float origin[3] = { ray_origins[idx*3], ray_origins[idx*3+1], ray_origins[idx*3+2] };
    float dir[3]    = { ray_dirs[idx*3],    ray_dirs[idx*3+1],    ray_dirs[idx*3+2] };
    float intensity = ray_intensities[idx];
    unsigned int source = ray_sources[idx];

    unsigned int stack[128];
    int sp = 0;
    stack[sp++] = root_idx;
    int   closest_idx = -1;
    float closest_t   = 1e30f;

    while (sp > 0) {
        unsigned int nodeIdx = stack[--sp];
        const BVHNodeDevReflBench& node = nodes[nodeIdx];
        float tmin, tmax;
        if (!reflBenchRayAABBIntersect(origin, dir, node.bounds, tmin, tmax)) continue;
        if (tmin > closest_t) continue;
        if (node.is_leaf) {
            unsigned int j = node.triangle_idx;
            if (j == source) continue;
            float v0[3]={triangles[j].v[0][0],triangles[j].v[0][1],triangles[j].v[0][2]};
            float v1[3]={triangles[j].v[1][0],triangles[j].v[1][1],triangles[j].v[1][2]};
            float v2[3]={triangles[j].v[2][0],triangles[j].v[2][1],triangles[j].v[2][2]};
            float t;
            if (reflBenchRayTriangleIntersect(origin, dir, v0, v1, v2, t))
                if (t < closest_t) { closest_t=t; closest_idx=(int)j; }
        } else {
            if (sp < 126) { stack[sp++]=node.left; stack[sp++]=node.right; }
        }
    }

    if (closest_idx < 0) return;

    unsigned int j = (unsigned int)closest_idx;
    float s_eq[3] = { -dir[0], -dir[1], -dir[2] };
    float nj[3] = { triangles[j].normal[0], triangles[j].normal[1], triangles[j].normal[2] };
    float cos_theta_j = nj[0]*s_eq[0] + nj[1]*s_eq[1] + nj[2]*s_eq[2];
    if (cos_theta_j <= 1e-6f) return;

    // NO atomicCAS bounce tracking
    float alpha_j = triangles[j].reflectance;
    float mu_j    = triangles[j].specularity;
    if (alpha_j <= 0.0f && mu_j <= 0.0f) { alpha_j = default_alpha; mu_j = default_mu; }

    // intensity already carries A_i*cos_theta_i*alpha_i*mu_i chain - beam cross-section conserved  //PHY
    float base    = intensity;                                                                        //PHY
    float s_coeff = -base * ((1.0f - alpha_j) + alpha_j*(1.0f - mu_j));                             //PHY
    float n_coeff = -base * (2.0f*alpha_j*mu_j*cos_theta_j + (2.0f/3.0f)*alpha_j*(1.0f - mu_j));   //PHY
    float fx = s_coeff*s_eq[0] + n_coeff*nj[0];   //PHY
    float fy = s_coeff*s_eq[1] + n_coeff*nj[1];   //PHY
    float fz = s_coeff*s_eq[2] + n_coeff*nj[2];   //PHY
    atomicAdd(&force_accum[0], (double)fx); atomicAdd(&force_accum[1], (double)fy); atomicAdd(&force_accum[2], (double)fz);

    // Actual hit point - correct moment arm and next-bounce origin  //PHY
    float hit_j[3] = { origin[0] + closest_t*dir[0],
                       origin[1] + closest_t*dir[1],
                       origin[2] + closest_t*dir[2] };
    atomicAdd(&moment_accum[0], (double)(hit_j[1]*fz - hit_j[2]*fy));   //PHY
    atomicAdd(&moment_accum[1], (double)(hit_j[2]*fx - hit_j[0]*fz));   //PHY
    atomicAdd(&moment_accum[2], (double)(hit_j[0]*fy - hit_j[1]*fx));   //PHY

    float new_intensity = intensity * alpha_j * mu_j;
    if (new_intensity < 1e-10f) return;
    float rnx=2.0f*cos_theta_j*nj[0]-s_eq[0], rny=2.0f*cos_theta_j*nj[1]-s_eq[1], rnz=2.0f*cos_theta_j*nj[2]-s_eq[2];
    float rnlen=sqrtf(rnx*rnx+rny*rny+rnz*rnz);
    if (rnlen < 1e-12f) return;
    rnx/=rnlen; rny/=rnlen; rnz/=rnlen;
    unsigned int write_idx = atomicAdd(next_ray_count, 1u);
    // Offset next-bounce origin by 1e-4 along hit-triangle normal to avoid self-intersection  //FIX-B
    next_ray_origins[write_idx*3]  =hit_j[0]+1e-4f*nj[0];
    next_ray_origins[write_idx*3+1]=hit_j[1]+1e-4f*nj[1];
    next_ray_origins[write_idx*3+2]=hit_j[2]+1e-4f*nj[2];
    next_ray_dirs[write_idx*3]=rnx; next_ray_dirs[write_idx*3+1]=rny; next_ray_dirs[write_idx*3+2]=rnz;
    next_ray_intensities[write_idx] = new_intensity;
    next_ray_sources[write_idx] = j;
}

// ===================== Direct-force GPU kernel (bench replacement for CPU compute_srp_forces) =====================
// Computes SRP force/moment for all directly-lit triangles (labels[i]==1) on GPU.
// Uses double atomicAdd to match the precision of the reflection accumulator.
// SRP formula (Kenneally): area_w = area*cos_theta
//   sc = -area_w * ((1-alpha) + alpha*(1-mu))      (absorbed + diffuse)
//   nc = -area_w * (2*alpha*mu*cos_theta + 2/3*alpha*(1-mu))
//   F = sc*sun + nc*normal;  M = center × F
__global__ void reflDirectForceBenchKernel(
    const TriangleDevReflBench* __restrict__ triangles,
    const float* __restrict__ centers,
    const int*   __restrict__ labels,
    size_t N,
    float sunX, float sunY, float sunZ,
    float default_alpha, float default_mu,
    double* __restrict__ force_accum,
    double* __restrict__ moment_accum)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || labels[i] == 0) return;

    float nx = triangles[i].normal[0], ny = triangles[i].normal[1], nz = triangles[i].normal[2];
    float cos_theta = nx*sunX + ny*sunY + nz*sunZ;
    if (cos_theta <= 1e-5f) return;   // 1e-5f matches RTX backface_filter  //PHY

    float alpha = triangles[i].reflectance;
    float mu    = triangles[i].specularity;
    if (alpha <= 0.0f && mu <= 0.0f) { alpha = default_alpha; mu = default_mu; }

    float area = triangles[i].area;
    if (area <= 0.0f) {
        float e1[3] = { triangles[i].v[1][0]-triangles[i].v[0][0],
                        triangles[i].v[1][1]-triangles[i].v[0][1],
                        triangles[i].v[1][2]-triangles[i].v[0][2] };
        float e2[3] = { triangles[i].v[2][0]-triangles[i].v[0][0],
                        triangles[i].v[2][1]-triangles[i].v[0][1],
                        triangles[i].v[2][2]-triangles[i].v[0][2] };
        float cr[3] = { e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0] };
        area = 0.5f * sqrtf(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
    }

    float area_w = area * cos_theta;
    float sc = -area_w * ((1.0f - alpha) + alpha*(1.0f - mu));
    float nc = -area_w * (2.0f*alpha*mu*cos_theta + (2.0f/3.0f)*alpha*(1.0f - mu));
    float fx = sc*sunX + nc*nx;
    float fy = sc*sunY + nc*ny;
    float fz = sc*sunZ + nc*nz;

    atomicAdd(&force_accum[0], (double)fx);
    atomicAdd(&force_accum[1], (double)fy);
    atomicAdd(&force_accum[2], (double)fz);

    float cx = centers[i*3], cy = centers[i*3+1], cz = centers[i*3+2];
    atomicAdd(&moment_accum[0], (double)(cy*fz - cz*fy));
    atomicAdd(&moment_accum[1], (double)(cz*fx - cx*fz));
    atomicAdd(&moment_accum[2], (double)(cx*fy - cy*fx));
}

// ===================== Host GPU cache =====================

struct ReflGPUCacheBench {
    TriangleDevReflBench* d_tri     = nullptr;
    float*                d_centers = nullptr;
    BVHNodeDevReflBench*  d_nodes   = nullptr;
    int*                  d_labels  = nullptr;
    unsigned int root_idx  = 0;
    size_t N               = 0;
    size_t numNodes        = 0;
    bool useConstant       = false;
    bool valid             = false;
    bool has_refl_buffers  = false;   // true iff max_reflections>0 when allocated
    cudaStream_t stream    = nullptr;

    // Base accumulators (always allocated with geometry cache)
    double* d_force_accum  = nullptr;
    double* d_moment_accum = nullptr;

    // Reflection ray buffers - only allocated when max_reflections>0
    float* d_ray_origins_A     = nullptr;
    float* d_ray_dirs_A        = nullptr;
    float* d_ray_intensities_A = nullptr;
    unsigned int* d_ray_sources_A  = nullptr;
    float* d_ray_origins_B     = nullptr;
    float* d_ray_dirs_B        = nullptr;
    float* d_ray_intensities_B = nullptr;
    unsigned int* d_ray_sources_B  = nullptr;
    unsigned int* d_ray_count_A    = nullptr;
    unsigned int* d_ray_count_B    = nullptr;
};

static ReflGPUCacheBench g_refl_bench_cache;

static void refl_bench_free_refl_buffers(ReflGPUCacheBench& c)
{
    auto cf = [](auto*& p){ if(p){ cudaFree(p); p=nullptr; } };
    cf(c.d_ray_origins_A); cf(c.d_ray_dirs_A); cf(c.d_ray_intensities_A); cf(c.d_ray_sources_A);
    cf(c.d_ray_origins_B); cf(c.d_ray_dirs_B); cf(c.d_ray_intensities_B); cf(c.d_ray_sources_B);
    cf(c.d_ray_count_A); cf(c.d_ray_count_B);
    c.has_refl_buffers = false;
}

static void refl_bench_alloc_refl_buffers(ReflGPUCacheBench& c, size_t N)
{
    REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_ray_origins_A,     N*3*sizeof(float)));
    REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_ray_dirs_A,        N*3*sizeof(float)));
    REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_ray_intensities_A, N*sizeof(float)));
    REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_ray_sources_A,     N*sizeof(unsigned int)));
    REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_ray_origins_B,     N*3*sizeof(float)));
    REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_ray_dirs_B,        N*3*sizeof(float)));
    REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_ray_intensities_B, N*sizeof(float)));
    REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_ray_sources_B,     N*sizeof(unsigned int)));
    REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_ray_count_A,       sizeof(unsigned int)));
    REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_ray_count_B,       sizeof(unsigned int)));
    c.has_refl_buffers = true;
}

// ===================== Main exported function =====================

SRPResult calculate_labels_reflections_gpu_bench(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    int max_reflections,
    bool verbose)
{
    if (triangles.empty()) return {};

    const float default_alpha = 0.5f, default_mu = 0.5f;

    double len = sqrt(sun_vector[0]*sun_vector[0] + sun_vector[1]*sun_vector[1] + sun_vector[2]*sun_vector[2]);
    if (len < 1e-12) throw std::runtime_error("Sun vector is zero");
    float sunX=(float)(sun_vector[0]/len), sunY=(float)(sun_vector[1]/len), sunZ=(float)(sun_vector[2]/len);
    size_t N = triangles.size();

    auto& c = g_refl_bench_cache;

    // Rebuild geometry cache if triangle set changed
    if (!c.valid || c.N != N) {
        if (c.d_tri)           REFL_BENCH_CUDA_CHECK(cudaFree(c.d_tri));
        if (c.d_centers)       REFL_BENCH_CUDA_CHECK(cudaFree(c.d_centers));
        if (c.d_nodes)         REFL_BENCH_CUDA_CHECK(cudaFree(c.d_nodes));
        if (c.d_labels)        REFL_BENCH_CUDA_CHECK(cudaFree(c.d_labels));
        if (c.d_force_accum)   REFL_BENCH_CUDA_CHECK(cudaFree(c.d_force_accum));
        if (c.d_moment_accum)  REFL_BENCH_CUDA_CHECK(cudaFree(c.d_moment_accum));
        if (c.stream)          REFL_BENCH_CUDA_CHECK(cudaStreamDestroy(c.stream));
        refl_bench_free_refl_buffers(c);

        c = ReflGPUCacheBench{};

        REFL_BENCH_CUDA_CHECK(cudaStreamCreate(&c.stream));

        // Build BVH
        std::vector<std::array<double,3>> centers_cpu(N);
        for (size_t i=0;i<N;++i) centers_cpu[i]={(triangles[i].v1_x+triangles[i].v2_x+triangles[i].v3_x)/3.0,
                                                   (triangles[i].v1_y+triangles[i].v2_y+triangles[i].v3_y)/3.0,
                                                   (triangles[i].v1_z+triangles[i].v2_z+triangles[i].v3_z)/3.0};
        struct AABB { double min_x,min_y,min_z,max_x,max_y,max_z; };
        std::vector<AABB> tb(N);
        for (size_t i=0;i<N;++i) tb[i]={std::min({triangles[i].v1_x,triangles[i].v2_x,triangles[i].v3_x}),
                                         std::min({triangles[i].v1_y,triangles[i].v2_y,triangles[i].v3_y}),
                                         std::min({triangles[i].v1_z,triangles[i].v2_z,triangles[i].v3_z}),
                                         std::max({triangles[i].v1_x,triangles[i].v2_x,triangles[i].v3_x}),
                                         std::max({triangles[i].v1_y,triangles[i].v2_y,triangles[i].v3_y}),
                                         std::max({triangles[i].v1_z,triangles[i].v2_z,triangles[i].v3_z})};
        struct BVHNode { AABB bounds; size_t triangle_idx,left,right; bool is_leaf; };
        std::vector<BVHNode> nodes;
        std::vector<size_t> indices(N); std::iota(indices.begin(),indices.end(),0);
        std::function<size_t(size_t,size_t,size_t)> build_bvh;
        build_bvh=[&](size_t start,size_t end,size_t depth)->size_t{
            BVHNode node{};
            if (end-start==1){node.is_leaf=true;node.triangle_idx=indices[start];node.bounds=tb[indices[start]];nodes.push_back(node);return nodes.size()-1;}
            double x_min=1e30,x_max=-1e30,y_min=1e30,y_max=-1e30,z_min=1e30,z_max=-1e30;
            for(size_t i=start;i<end;++i){size_t id=indices[i];x_min=std::min(x_min,tb[id].min_x);x_max=std::max(x_max,tb[id].max_x);y_min=std::min(y_min,tb[id].min_y);y_max=std::max(y_max,tb[id].max_y);z_min=std::min(z_min,tb[id].min_z);z_max=std::max(z_max,tb[id].max_z);}
            int axis=0; double dx=x_max-x_min,dy=y_max-y_min,dz=z_max-z_min;
            if(dy>dx&&dy>dz)axis=1;else if(dz>dx)axis=2;
            std::sort(indices.begin()+start,indices.begin()+end,[&](size_t a,size_t b){return centers_cpu[a][axis]<centers_cpu[b][axis];});
            size_t mid=(start+end)/2;
            node.is_leaf=false;
            node.left=build_bvh(start,mid,depth+1); node.right=build_bvh(mid,end,depth+1);
            node.bounds={std::min(nodes[node.left].bounds.min_x,nodes[node.right].bounds.min_x),
                         std::min(nodes[node.left].bounds.min_y,nodes[node.right].bounds.min_y),
                         std::min(nodes[node.left].bounds.min_z,nodes[node.right].bounds.min_z),
                         std::max(nodes[node.left].bounds.max_x,nodes[node.right].bounds.max_x),
                         std::max(nodes[node.left].bounds.max_y,nodes[node.right].bounds.max_y),
                         std::max(nodes[node.left].bounds.max_z,nodes[node.right].bounds.max_z)};
            nodes.push_back(node); return nodes.size()-1;
        };
        size_t root=build_bvh(0,N,0);
        c.root_idx=(unsigned int)root; c.numNodes=nodes.size();

        std::vector<TriangleDevReflBench> triDev(N);
        std::vector<float> centers_flat(N*3);
        for(size_t i=0;i<N;++i){
            triDev[i].v[0][0]=(float)triangles[i].v1_x; triDev[i].v[0][1]=(float)triangles[i].v1_y; triDev[i].v[0][2]=(float)triangles[i].v1_z;
            triDev[i].v[1][0]=(float)triangles[i].v2_x; triDev[i].v[1][1]=(float)triangles[i].v2_y; triDev[i].v[1][2]=(float)triangles[i].v2_z;
            triDev[i].v[2][0]=(float)triangles[i].v3_x; triDev[i].v[2][1]=(float)triangles[i].v3_y; triDev[i].v[2][2]=(float)triangles[i].v3_z;
            triDev[i].normal[0]=(float)triangles[i].normal_x; triDev[i].normal[1]=(float)triangles[i].normal_y; triDev[i].normal[2]=(float)triangles[i].normal_z;
            triDev[i].reflectance=(float)triangles[i].reflectance; triDev[i].specularity=(float)triangles[i].specularity;
            triDev[i].emissivity=(float)triangles[i].emissivity; triDev[i].area=(float)triangles[i].area;
            centers_flat[i*3]=(float)centers_cpu[i][0]; centers_flat[i*3+1]=(float)centers_cpu[i][1]; centers_flat[i*3+2]=(float)centers_cpu[i][2];
        }
        std::vector<BVHNodeDevReflBench> nodesDev(nodes.size());
        for(size_t i=0;i<nodes.size();++i){
            nodesDev[i].bounds.min[0]=(float)nodes[i].bounds.min_x; nodesDev[i].bounds.min[1]=(float)nodes[i].bounds.min_y; nodesDev[i].bounds.min[2]=(float)nodes[i].bounds.min_z;
            nodesDev[i].bounds.max[0]=(float)nodes[i].bounds.max_x; nodesDev[i].bounds.max[1]=(float)nodes[i].bounds.max_y; nodesDev[i].bounds.max[2]=(float)nodes[i].bounds.max_z;
            nodesDev[i].triangle_idx=(unsigned int)nodes[i].triangle_idx; nodesDev[i].left=(unsigned int)nodes[i].left; nodesDev[i].right=(unsigned int)nodes[i].right; nodesDev[i].is_leaf=nodes[i].is_leaf;
        }
        REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_tri,     N*sizeof(TriangleDevReflBench)));
        REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_centers, N*3*sizeof(float)));
        REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_nodes,   nodes.size()*sizeof(BVHNodeDevReflBench)));
        REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_labels,  N*sizeof(int)));
        REFL_BENCH_CUDA_CHECK(cudaMemcpy(c.d_tri,     triDev.data(),    N*sizeof(TriangleDevReflBench),             cudaMemcpyHostToDevice));
        REFL_BENCH_CUDA_CHECK(cudaMemcpy(c.d_centers, centers_flat.data(), N*3*sizeof(float),                      cudaMemcpyHostToDevice));
        REFL_BENCH_CUDA_CHECK(cudaMemcpy(c.d_nodes,   nodesDev.data(),  nodes.size()*sizeof(BVHNodeDevReflBench),  cudaMemcpyHostToDevice));
        size_t nodeSize=sizeof(BVHNodeDevReflBench);
        c.useConstant = (nodes.size()*nodeSize <= 64*1024);
        if (c.useConstant)
            REFL_BENCH_CUDA_CHECK(cudaMemcpyToSymbol(d_refl_bench_constant_nodes, nodesDev.data(), nodes.size()*nodeSize));

        // Allocate double-precision force/moment accumulators as part of base geometry cache
        REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_force_accum,  3*sizeof(double)));
        REFL_BENCH_CUDA_CHECK(cudaMalloc(&c.d_moment_accum, 3*sizeof(double)));

        c.N=N; c.valid=true;
    }

    // Allocate reflection ray buffers lazily when max_reflections>0 first appears
    if (max_reflections > 0 && !c.has_refl_buffers)
        refl_bench_alloc_refl_buffers(c, N);

    // =============== Phase 1: Direct illumination ===============
    int blockSize, minGrid;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, reflShadowBenchKernel_global, 0, 0);
    int gridSize = (int)((N+blockSize-1)/blockSize);

    // Zero accumulators unconditionally before shadow pass
    REFL_BENCH_CUDA_CHECK(cudaMemsetAsync(c.d_force_accum,  0, 3*sizeof(double), c.stream));
    REFL_BENCH_CUDA_CHECK(cudaMemsetAsync(c.d_moment_accum, 0, 3*sizeof(double), c.stream));

    reflInitLabelsBenchKernel<<<gridSize,blockSize,0,c.stream>>>(c.d_tri, c.d_labels, N, sunX,sunY,sunZ);
    REFL_BENCH_CUDA_CHECK(cudaGetLastError());

    if (c.useConstant)
        reflShadowBenchKernel_constant<<<gridSize,blockSize,0,c.stream>>>(c.d_tri,c.d_centers,c.d_labels,N,sunX,sunY,sunZ,c.root_idx);
    else
        reflShadowBenchKernel_global<<<gridSize,blockSize,0,c.stream>>>(c.d_tri,c.d_nodes,c.d_centers,c.d_labels,N,sunX,sunY,sunZ,c.root_idx);
    REFL_BENCH_CUDA_CHECK(cudaGetLastError());

    // Compute direct SRP force on GPU (replaces CPU compute_srp_forces)
    reflDirectForceBenchKernel<<<gridSize,blockSize,0,c.stream>>>(
        c.d_tri, c.d_centers, c.d_labels, N, sunX, sunY, sunZ,
        default_alpha, default_mu, c.d_force_accum, c.d_moment_accum);
    REFL_BENCH_CUDA_CHECK(cudaGetLastError());

    // =============== Phase 2: Reflections (skipped when max_reflections==0) ===============
    if (max_reflections > 0) {
        REFL_BENCH_CUDA_CHECK(cudaMemsetAsync(c.d_ray_count_A,  0, sizeof(unsigned int), c.stream));

        reflGenerateFirstBounceBenchKernel<<<gridSize,blockSize,0,c.stream>>>(
            c.d_tri, c.d_labels, c.d_centers, N, sunX,sunY,sunZ, default_alpha,default_mu,
            c.d_ray_origins_A, c.d_ray_dirs_A, c.d_ray_intensities_A, c.d_ray_sources_A, c.d_ray_count_A);
        REFL_BENCH_CUDA_CHECK(cudaGetLastError());

        float* cur_origins=c.d_ray_origins_A, *cur_dirs=c.d_ray_dirs_A, *cur_int=c.d_ray_intensities_A;
        unsigned int *cur_src=c.d_ray_sources_A, *cur_cnt=c.d_ray_count_A;
        float* nxt_origins=c.d_ray_origins_B, *nxt_dirs=c.d_ray_dirs_B, *nxt_int=c.d_ray_intensities_B;
        unsigned int *nxt_src=c.d_ray_sources_B, *nxt_cnt=c.d_ray_count_B;

        for (int bounce=0; bounce<max_reflections; ++bounce) {
            unsigned int h_ray_count=0;
            REFL_BENCH_CUDA_CHECK(cudaMemcpyAsync(&h_ray_count,cur_cnt,sizeof(unsigned int),cudaMemcpyDeviceToHost,c.stream));
            REFL_BENCH_CUDA_CHECK(cudaStreamSynchronize(c.stream));
            if (h_ray_count==0) break;
            REFL_BENCH_CUDA_CHECK(cudaMemsetAsync(nxt_cnt,0,sizeof(unsigned int),c.stream));
            int rbs,rmg; cudaOccupancyMaxPotentialBlockSize(&rmg,&rbs,reflTraceAndBounceBenchKernel,0,0);
            int rgs=(int)((h_ray_count+rbs-1)/rbs);
            reflTraceAndBounceBenchKernel<<<rgs,rbs,0,c.stream>>>(
                cur_origins,cur_dirs,cur_int,cur_src,h_ray_count,
                c.d_tri,c.d_nodes,c.d_centers,N,c.root_idx,default_alpha,default_mu,
                c.d_force_accum,c.d_moment_accum,
                nxt_origins,nxt_dirs,nxt_int,nxt_src,nxt_cnt);
            REFL_BENCH_CUDA_CHECK(cudaGetLastError());
            std::swap(cur_origins,nxt_origins); std::swap(cur_dirs,nxt_dirs);
            std::swap(cur_int,nxt_int); std::swap(cur_src,nxt_src); std::swap(cur_cnt,nxt_cnt);
        }
    }

    REFL_BENCH_CUDA_CHECK(cudaStreamSynchronize(c.stream));

    // Read back unified double-precision accumulators (direct + reflection)
    double h_force[3]={}, h_moment[3]={};
    REFL_BENCH_CUDA_CHECK(cudaMemcpy(h_force,  c.d_force_accum,  3*sizeof(double), cudaMemcpyDeviceToHost));
    REFL_BENCH_CUDA_CHECK(cudaMemcpy(h_moment, c.d_moment_accum, 3*sizeof(double), cudaMemcpyDeviceToHost));

    // NO set_bounce_globals call
    SRPResult result;
    result.labels       = {};   // bench: labels not transferred to host
    result.total_force  = { g_srp_phi0*h_force[0],  g_srp_phi0*h_force[1],  g_srp_phi0*h_force[2]  };
    result.total_moment = { g_srp_phi0*h_moment[0], g_srp_phi0*h_moment[1], g_srp_phi0*h_moment[2] };
    return result;
}
