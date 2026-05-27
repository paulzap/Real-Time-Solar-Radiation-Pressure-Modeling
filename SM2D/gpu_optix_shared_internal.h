// gpu_optix_shared_internal.h
// ============================================================
// Internal header: shared OptiX/CUDA types, error macros, and
// RTXSharedState (context + BVH + per-triangle arrays).
// Included by gpu_optix_shared.cu, gpu_optix_center_raytracer.cu
// and gpu_optix_pixel_grid_raytracer.cu.
// DO NOT include from public headers.
// ============================================================
#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <string>
#include "ShadowAlgorithms.h"

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------
#define OPTIX_CHECK(call) { \
    OptixResult _res = (call); \
    if (_res != OPTIX_SUCCESS) { \
        throw std::runtime_error(std::string("OptiX error ") + optixGetErrorName(_res) \
            + " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } \
}

#define CUDA_CHECK(call) { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err) \
            + " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } \
}

// ---------------------------------------------------------------------------
// SBT record types  (header-only, used by both pipelines)
// ---------------------------------------------------------------------------
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord   { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord     { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

// ---------------------------------------------------------------------------
// Helper: allocate and upload one SBT record on device
// ---------------------------------------------------------------------------
static inline void make_sbt_record(OptixProgramGroup pg, size_t rec_size, CUdeviceptr& out_ptr)
{
    std::vector<char> host_rec(rec_size, 0);
    OPTIX_CHECK(optixSbtRecordPackHeader(pg, host_rec.data()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&out_ptr), rec_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(out_ptr), host_rec.data(),
                          rec_size, cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// RTXSharedState -- OptiX context, stream, BVH, per-triangle arrays,
// and per-call output buffers shared between both RTX methods.
// ---------------------------------------------------------------------------
struct RTXSharedState {
    bool valid             = false;   // geometry built, arrays allocated
    bool optix_initialized = false;   // context + stream created

    OptixDeviceContext context = nullptr;
    CUstream           stream  = nullptr;

    // Compacted acceleration structure
    OptixTraversableHandle bvh       = 0;
    CUdeviceptr            d_bvh_buf = 0;

    // Per-triangle attribute arrays (float, read-only per launch)
    CUdeviceptr d_centroids   = 0;   // float3 [N]
    CUdeviceptr d_normals     = 0;   // float3 [N]
    CUdeviceptr d_areas       = 0;   // float  [N]
    CUdeviceptr d_reflectance = 0;   // float  [N]
    CUdeviceptr d_specularity = 0;   // float  [N]

    // Per-call output buffers (zeroed / sentinel-filled before each launch)
    CUdeviceptr d_labels        = 0;  // int   [N]
    CUdeviceptr d_bounce_levels = 0;  // int   [N], sentinel 0x7F7F7F7F = "not hit"
    CUdeviceptr d_incident_dirs = 0;  // float [N*3], sentinel 0xFF = NaN
    CUdeviceptr d_origin_pts    = 0;  // float [N*3], sentinel 0xFF = NaN

    size_t N = 0;
};

extern RTXSharedState g_rtx_shared;

// ---------------------------------------------------------------------------
// Declared in gpu_optix_shared.cu, called by both method files
// ---------------------------------------------------------------------------

// Ensure context exists and BVH is up to date for the given triangle set.
void ensure_rtx_common(const std::vector<Triangle>& triangles);

// Zero labels, set bounce_levels/incident_dirs/origin_pts sentinels.
// Call immediately before optixLaunch.
void reset_rtx_per_call_buffers();
