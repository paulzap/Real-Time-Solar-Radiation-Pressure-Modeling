// gpu_optix_shared.cu
// ============================================================
// Shared OptiX infrastructure: device context, CUDA stream,
// BVH (GAS) build, and per-triangle buffer management.
// Shared between gpu_optix_center_raytracer.cu and
//               gpu_optix_pixel_grid_raytracer.cu.
//
// NOTE: optix_function_table_definition.h is included here
// (defines OPTIX_FUNCTION_TABLE_SYMBOL) - must appear in
// exactly ONE translation unit across the entire project.
// ============================================================

#include "gpu_optix_shared_internal.h"
#include <optix_function_table_definition.h>
#include <cuda.h>                // cuCtxGetCurrent
#include <cmath>
#include <iostream>
#include <algorithm>

// ---------------------------------------------------------------------------
// Global shared state definition
// ---------------------------------------------------------------------------
RTXSharedState g_rtx_shared;

// ---------------------------------------------------------------------------
// OptiX log callback (warnings/errors to stderr)
// ---------------------------------------------------------------------------
static void optix_log_cb(unsigned int level, const char* tag,
                         const char* message, void* /*cbdata*/)
{
    if (level <= 3)
        std::cerr << "[OptiX][" << tag << "] " << message << "\n";
}

// ---------------------------------------------------------------------------
// init_optix_context  (called once)
// ---------------------------------------------------------------------------
static void init_optix_context()
{
    CUDA_CHECK(cudaFree(0));   // force CUDA runtime initialization

    CUcontext cu_ctx = nullptr;
    if (cuCtxGetCurrent(&cu_ctx) != CUDA_SUCCESS || cu_ctx == nullptr)
        throw std::runtime_error(
            "RTX/OptiX not available: no active CUDA context. "
            "Make sure a CUDA-capable GPU is present.");

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions ctx_opts = {};
    ctx_opts.logCallbackFunction = optix_log_cb;
    ctx_opts.logCallbackLevel    = 3;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &ctx_opts, &g_rtx_shared.context));

    CUDA_CHECK(cudaStreamCreate(&g_rtx_shared.stream));
}

// ---------------------------------------------------------------------------
// free_geometry_buffers
// ---------------------------------------------------------------------------
static void free_geometry_buffers()
{
    auto cf = [](CUdeviceptr& p) {
        if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; }
    };
    cf(g_rtx_shared.d_bvh_buf);
    cf(g_rtx_shared.d_centroids);
    cf(g_rtx_shared.d_normals);
    cf(g_rtx_shared.d_areas);
    cf(g_rtx_shared.d_reflectance);
    cf(g_rtx_shared.d_specularity);
    cf(g_rtx_shared.d_labels);
    cf(g_rtx_shared.d_bounce_levels);
    cf(g_rtx_shared.d_incident_dirs);
    cf(g_rtx_shared.d_origin_pts);
    g_rtx_shared.bvh   = 0;
    g_rtx_shared.N     = 0;
    g_rtx_shared.valid = false;
}

// ---------------------------------------------------------------------------
// build_gas_and_buffers  -- BVH build + per-triangle arrays
// ---------------------------------------------------------------------------
static void build_gas_and_buffers(const std::vector<Triangle>& triangles)
{
    const size_t N = triangles.size();

    // ---- Pack vertices (3 float3 per triangle, sequential - no index buffer) ----
    std::vector<float3> h_verts;
    h_verts.reserve(N * 3);
    for (const auto& t : triangles) {
        h_verts.push_back({ (float)t.v1_x, (float)t.v1_y, (float)t.v1_z });
        h_verts.push_back({ (float)t.v2_x, (float)t.v2_y, (float)t.v2_z });
        h_verts.push_back({ (float)t.v3_x, (float)t.v3_y, (float)t.v3_z });
    }
    CUdeviceptr d_verts = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_verts),
                          h_verts.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_verts), h_verts.data(),
                          h_verts.size() * sizeof(float3), cudaMemcpyHostToDevice));

    // ---- Build input ----
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    OptixBuildInputTriangleArray& tri_arr = build_input.triangleArray;
    tri_arr.vertexBuffers             = &d_verts;
    tri_arr.numVertices               = (unsigned int)(N * 3);
    tri_arr.vertexFormat              = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri_arr.vertexStrideInBytes       = sizeof(float3);
    tri_arr.indexBuffer               = 0;
    tri_arr.numIndexTriplets          = 0;
    tri_arr.indexFormat               = OPTIX_INDICES_FORMAT_NONE;
    tri_arr.numSbtRecords             = 1;
    tri_arr.sbtIndexOffsetBuffer      = 0;
    tri_arr.sbtIndexOffsetSizeInBytes = 0;
    tri_arr.primitiveIndexOffset      = 0;
    static const unsigned int geom_flags = OPTIX_GEOMETRY_FLAG_NONE;
    tri_arr.flags = &geom_flags;

    // ---- Accel build options ----
    OptixAccelBuildOptions accel_opts = {};
    accel_opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                            OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buf_sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        g_rtx_shared.context, &accel_opts, &build_input, 1, &buf_sizes));

    CUdeviceptr d_temp = 0, d_output = 0, d_compact_size = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp),         buf_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output),       buf_sizes.outputSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compact_size), sizeof(size_t)));

    OptixAccelEmitDesc emit_desc = {};
    emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = d_compact_size;

    OptixTraversableHandle temp_bvh = 0;
    OPTIX_CHECK(optixAccelBuild(
        g_rtx_shared.context, g_rtx_shared.stream,
        &accel_opts, &build_input, 1,
        d_temp, buf_sizes.tempSizeInBytes,
        d_output, buf_sizes.outputSizeInBytes,
        &temp_bvh, &emit_desc, 1));
    CUDA_CHECK(cudaStreamSynchronize(g_rtx_shared.stream));

    size_t compact_size = 0;
    CUDA_CHECK(cudaMemcpy(&compact_size, reinterpret_cast<void*>(d_compact_size),
                          sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_rtx_shared.d_bvh_buf), compact_size));
    OPTIX_CHECK(optixAccelCompact(g_rtx_shared.context, g_rtx_shared.stream,
                                  temp_bvh, g_rtx_shared.d_bvh_buf, compact_size,
                                  &g_rtx_shared.bvh));
    CUDA_CHECK(cudaStreamSynchronize(g_rtx_shared.stream));

    cudaFree(reinterpret_cast<void*>(d_temp));
    cudaFree(reinterpret_cast<void*>(d_output));
    cudaFree(reinterpret_cast<void*>(d_compact_size));
    cudaFree(reinterpret_cast<void*>(d_verts));

    // ---- Per-triangle attribute arrays ----
    std::vector<float3> h_centroids(N), h_normals(N);
    std::vector<float>  h_areas(N), h_refl(N), h_spec(N);

    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = triangles[i];

        if (std::abs(t.centroid_x) > 1e-30 ||
            std::abs(t.centroid_y) > 1e-30 ||
            std::abs(t.centroid_z) > 1e-30) {
            h_centroids[i] = { (float)t.centroid_x,
                               (float)t.centroid_y,
                               (float)t.centroid_z };
        } else {
            h_centroids[i] = {
                (float)((t.v1_x + t.v2_x + t.v3_x) / 3.0),
                (float)((t.v1_y + t.v2_y + t.v3_y) / 3.0),
                (float)((t.v1_z + t.v2_z + t.v3_z) / 3.0)
            };
        }

        h_normals[i] = { (float)t.normal_x, (float)t.normal_y, (float)t.normal_z };

        if (t.area > 1e-30) {
            h_areas[i] = (float)t.area;
        } else {
            float e1x = (float)(t.v2_x - t.v1_x);
            float e1y = (float)(t.v2_y - t.v1_y);
            float e1z = (float)(t.v2_z - t.v1_z);
            float e2x = (float)(t.v3_x - t.v1_x);
            float e2y = (float)(t.v3_y - t.v1_y);
            float e2z = (float)(t.v3_z - t.v1_z);
            float cx = e1y*e2z - e1z*e2y;
            float cy = e1z*e2x - e1x*e2z;
            float cz = e1x*e2y - e1y*e2x;
            h_areas[i] = 0.5f * std::sqrt(cx*cx + cy*cy + cz*cz);
        }

        h_refl[i] = (float)t.reflectance;
        h_spec[i] = (float)t.specularity;
    }

    auto upload = [](CUdeviceptr& ptr, const void* data, size_t bytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), bytes));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(ptr), data, bytes,
                              cudaMemcpyHostToDevice));
    };
    upload(g_rtx_shared.d_centroids,   h_centroids.data(), N * sizeof(float3));
    upload(g_rtx_shared.d_normals,     h_normals.data(),   N * sizeof(float3));
    upload(g_rtx_shared.d_areas,       h_areas.data(),     N * sizeof(float));
    upload(g_rtx_shared.d_reflectance, h_refl.data(),      N * sizeof(float));
    upload(g_rtx_shared.d_specularity, h_spec.data(),      N * sizeof(float));

    // Per-call output buffers
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_rtx_shared.d_labels),
                          N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_rtx_shared.d_bounce_levels),
                          N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_rtx_shared.d_incident_dirs),
                          N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_rtx_shared.d_origin_pts),
                          N * 3 * sizeof(float)));

    g_rtx_shared.N     = N;
    g_rtx_shared.valid = true;
}

// ---------------------------------------------------------------------------
// ensure_rtx_common  (public - called by both method files)
// ---------------------------------------------------------------------------
void ensure_rtx_common(const std::vector<Triangle>& triangles)
{
    if (!g_rtx_shared.optix_initialized) {
        init_optix_context();
        g_rtx_shared.optix_initialized = true;
    }

    if (!g_rtx_shared.valid || g_rtx_shared.N != triangles.size()) {
        if (g_rtx_shared.valid) free_geometry_buffers();
        build_gas_and_buffers(triangles);
    }
}

// ---------------------------------------------------------------------------
// is_optix_runtime_available  (public - callable from plain .cpp files)
// ---------------------------------------------------------------------------
bool is_optix_runtime_available()
{
    try {
        CUDA_CHECK(cudaFree(0));  // ensure CUDA runtime is up
        return (optixInit() == OPTIX_SUCCESS);
    } catch (...) {
        return false;
    }
}

// ---------------------------------------------------------------------------
// reset_rtx_per_call_buffers  (public - called before each optixLaunch)
// ---------------------------------------------------------------------------
void reset_rtx_per_call_buffers()
{
    const size_t N = g_rtx_shared.N;
    const CUstream s = g_rtx_shared.stream;
    // 0x7F7F7F7F (~2.1e9) sentinel for bounce_levels; 0xFF = float NaN for dirs.
    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_rtx_shared.d_labels),        0x00,
                               N * sizeof(int), s));
    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_rtx_shared.d_bounce_levels), 0x7F,
                               N * sizeof(int), s));
    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_rtx_shared.d_incident_dirs), 0xFF,
                               N * 3 * sizeof(float), s));
    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_rtx_shared.d_origin_pts),    0xFF,
                               N * 3 * sizeof(float), s));
}
