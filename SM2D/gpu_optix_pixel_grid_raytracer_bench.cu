// gpu_optix_pixel_grid_raytracer_bench.cu
// ============================================================
// Benchmark-only RTX Pixel Grid method (method 8 lean copy).
// No bounce tracking, no set_bounce_globals.
// double accumulators preserved (fine grids exceed float saturation limit).
//
// Differences vs gpu_optix_pixel_grid_raytracer.cu:
//   - reset_bench_labels_only_grid(): only zeros d_labels
//   - No bounce array GPU→host copies
//   - No set_bounce_globals call
//   - Loads optix_shaders_pixel_grid_bench.ptx
//
// Exports: calculate_labels_pixel_grid_rtx_bench()
// ============================================================

#include "gpu_optix_shared_internal.h"
#include "bench_methods.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <functional>
#include <array>

// (CPU BVH for centroid reflection pass removed - reflections are now computed
//  in the OptiX shader with area_w = s², matching the classical pixel-grid method.)

// ---------------------------------------------------------------------------
// RTXParamsGridBench -- MUST exactly match optix_shaders_pixel_grid_bench.cu
// ---------------------------------------------------------------------------
struct RTXParamsGridBench {
    OptixTraversableHandle bvh;
    float3  sun;
    int*    d_labels;
    double* d_force;
    double* d_moment;
    float3* d_normals;
    float*  d_areas;
    float*  d_reflectance;
    float*  d_specularity;
    int     N;
    int     max_reflections;
    // NO bounce tracking fields
    float   u_min, v_min, grid_step;
    float3  u_axis, v_axis;
    float   t_start;
};

// ---------------------------------------------------------------------------
// Pipeline state
// ---------------------------------------------------------------------------
struct RTXGridBenchState {
    bool built = false;

    OptixModule       module    = nullptr;
    OptixProgramGroup pg_raygen = nullptr;
    OptixProgramGroup pg_miss   = nullptr;
    OptixProgramGroup pg_hit    = nullptr;
    OptixPipeline     pipeline  = nullptr;

    OptixShaderBindingTable sbt = {};
    CUdeviceptr d_sbt_rg = 0, d_sbt_ms = 0, d_sbt_hg = 0;

    CUdeviceptr d_force  = 0;  // double[3]
    CUdeviceptr d_moment = 0;  // double[3]
    CUdeviceptr d_params = 0;  // sizeof(RTXParamsGridBench)
};

static RTXGridBenchState g_grid_bench;

// ---------------------------------------------------------------------------
// load_ptx_grid_bench
// ---------------------------------------------------------------------------
static std::string load_ptx_grid_bench()
{
    const char* candidates[] = {
        "optix_shaders_pixel_grid_bench.ptx",
        "Debug/optix_shaders_pixel_grid_bench.ptx",
        "Release/optix_shaders_pixel_grid_bench.ptx"
    };
    for (const char* path : candidates) {
        std::ifstream f(path, std::ios::binary);
        if (f.good()) {
            std::ostringstream ss;
            ss << f.rdbuf();
            std::string content = ss.str();
            if (!content.empty()) return content;
        }
    }
    throw std::runtime_error(
        "RTX bench not available: cannot find optix_shaders_pixel_grid_bench.ptx "
        "in the working directory, Debug/, or Release/. "
        "Compile optix_shaders_pixel_grid_bench.cu to PTX first:\n"
        "  nvcc -ptx -arch=<OptixPtxArch> "
        "-I\"<OptixRoot>/include\" "
        "optix_shaders_pixel_grid_bench.cu -o optix_shaders_pixel_grid_bench.ptx\n"
        "(OptixRoot and OptixPtxArch are defined in LocalPaths.props)");
}

// ---------------------------------------------------------------------------
// build_grid_bench_pipeline
// ---------------------------------------------------------------------------
static void build_grid_bench_pipeline()
{
    std::string ptx_source = load_ptx_grid_bench();

    OptixModuleCompileOptions mco = {};
    mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    mco.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pco = {};
    pco.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pco.numPayloadValues                 = 2;
    pco.numAttributeValues               = 2;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags           = (unsigned int)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char log[4096]; size_t log_size = sizeof(log);

    OPTIX_CHECK(optixModuleCreate(
        g_rtx_shared.context, &mco, &pco,
        ptx_source.c_str(), ptx_source.size(),
        log, &log_size, &g_grid_bench.module));

    OptixProgramGroupOptions pg_opts = {};

    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      d.raygen.module = g_grid_bench.module;
      d.raygen.entryFunctionName = "__raygen__pixel_grid_bench";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_grid_bench.pg_raygen)); }

    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      d.miss.module = g_grid_bench.module;
      d.miss.entryFunctionName = "__miss__pixel_grid_bench";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_grid_bench.pg_miss)); }

    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      d.hitgroup.moduleCH            = g_grid_bench.module;
      d.hitgroup.entryFunctionNameCH = "__closesthit__pixel_grid_bench";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_grid_bench.pg_hit)); }

    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;
    OptixProgramGroup pgs[] = { g_grid_bench.pg_raygen, g_grid_bench.pg_miss, g_grid_bench.pg_hit };
    log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(g_rtx_shared.context, &pco, &plo,
                                    pgs, 3, log, &log_size, &g_grid_bench.pipeline));
    OPTIX_CHECK(optixPipelineSetStackSize(g_grid_bench.pipeline, 0, 0, 2048, 1));

    make_sbt_record(g_grid_bench.pg_raygen, sizeof(RaygenRecord),   g_grid_bench.d_sbt_rg);
    make_sbt_record(g_grid_bench.pg_miss,   sizeof(MissRecord),     g_grid_bench.d_sbt_ms);
    make_sbt_record(g_grid_bench.pg_hit,    sizeof(HitgroupRecord), g_grid_bench.d_sbt_hg);

    g_grid_bench.sbt = {};
    g_grid_bench.sbt.raygenRecord                = g_grid_bench.d_sbt_rg;
    g_grid_bench.sbt.missRecordBase              = g_grid_bench.d_sbt_ms;
    g_grid_bench.sbt.missRecordStrideInBytes     = sizeof(MissRecord);
    g_grid_bench.sbt.missRecordCount             = 1;
    g_grid_bench.sbt.hitgroupRecordBase          = g_grid_bench.d_sbt_hg;
    g_grid_bench.sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    g_grid_bench.sbt.hitgroupRecordCount         = 1;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_grid_bench.d_force),  3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_grid_bench.d_moment), 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_grid_bench.d_params), sizeof(RTXParamsGridBench)));

    g_grid_bench.built = true;
}

// ---------------------------------------------------------------------------
// ensure_grid_bench
// ---------------------------------------------------------------------------
static void ensure_grid_bench(const std::vector<Triangle>& triangles)
{
    ensure_rtx_common(triangles);
    if (!g_grid_bench.built) build_grid_bench_pipeline();
}

// ---------------------------------------------------------------------------
// reset_bench_labels_only_grid -- zeros only d_labels
// ---------------------------------------------------------------------------
static void reset_bench_labels_only_grid()
{
    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_rtx_shared.d_labels), 0x00,
                               g_rtx_shared.N * sizeof(int), g_rtx_shared.stream));
}

// ---------------------------------------------------------------------------
// rtx_grid_bench_backface_filter
// GPU kernel: same purpose as rtx_center_bench_backface_filter - eliminates
// stripe artifacts from float32 normal·sun rounding on near-edge-on panels.
// ---------------------------------------------------------------------------
__global__ void rtx_grid_bench_backface_filter(
    int* d_labels, const float3* d_normals, float3 sun, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !d_labels[i]) return;
    float3 n = d_normals[i];
    if (n.x*sun.x + n.y*sun.y + n.z*sun.z <= 1e-5f) d_labels[i] = 0;
}

// ---------------------------------------------------------------------------
// calculate_labels_pixel_grid_rtx_bench  (benchmark-only method 8)
// ---------------------------------------------------------------------------
SRPResult calculate_labels_pixel_grid_rtx_bench(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections,
    bool verbose)
{
    try {
        if (triangles.empty()) return {};

        double slen = std::sqrt(sun_vector[0]*sun_vector[0] +
                                sun_vector[1]*sun_vector[1] +
                                sun_vector[2]*sun_vector[2]);
        if (slen < 1e-12)
            throw std::runtime_error("Sun vector is zero");
        float sx = (float)(sun_vector[0] / slen);
        float sy = (float)(sun_vector[1] / slen);
        float sz = (float)(sun_vector[2] / slen);
        float3 sun_f = { sx, sy, sz };

        // ---- Build orthonormal basis {u, v} perpendicular to sun ----
        float3 up = { 0.0f, 0.0f, 1.0f };
        if (std::abs(sz) > 0.9f) up = { 1.0f, 0.0f, 0.0f };

        float3 u_axis;
        u_axis.x = up.y*sz - up.z*sy;
        u_axis.y = up.z*sx - up.x*sz;
        u_axis.z = up.x*sy - up.y*sx;
        float u_len = std::sqrt(u_axis.x*u_axis.x + u_axis.y*u_axis.y + u_axis.z*u_axis.z);
        if (u_len < 1e-12f) u_len = 1.0f;
        u_axis.x /= u_len; u_axis.y /= u_len; u_axis.z /= u_len;

        float3 v_axis;
        v_axis.x = sy*u_axis.z - sz*u_axis.y;
        v_axis.y = sz*u_axis.x - sx*u_axis.z;
        v_axis.z = sx*u_axis.y - sy*u_axis.x;

        // ---- Grid extents ----
        float u_min =  1e30f, u_max = -1e30f;
        float v_min_ =  1e30f, v_max_ = -1e30f;
        float s_max_ = -1e30f;

        for (const auto& t : triangles) {
            float verts[3][3] = {
                { (float)t.v1_x, (float)t.v1_y, (float)t.v1_z },
                { (float)t.v2_x, (float)t.v2_y, (float)t.v2_z },
                { (float)t.v3_x, (float)t.v3_y, (float)t.v3_z }
            };
            for (int vi = 0; vi < 3; ++vi) {
                float pu = verts[vi][0]*u_axis.x + verts[vi][1]*u_axis.y + verts[vi][2]*u_axis.z;
                float pv = verts[vi][0]*v_axis.x + verts[vi][1]*v_axis.y + verts[vi][2]*v_axis.z;
                float ps = verts[vi][0]*sx        + verts[vi][1]*sy        + verts[vi][2]*sz;
                u_min  = std::min(u_min,  pu); u_max  = std::max(u_max,  pu);
                v_min_ = std::min(v_min_, pv); v_max_ = std::max(v_max_, pv);
                s_max_ = std::max(s_max_, ps);
            }
        }

        float fstep = (float)grid_step;
        const float u_center = 0.5f * (u_min  + u_max);
        const float v_center = 0.5f * (v_min_ + v_max_);
        const float u_half   = 0.5f * (u_max  - u_min)  + fstep;
        const float v_half   = 0.5f * (v_max_ - v_min_) + fstep;
        float t_start = s_max_ + fstep;

        unsigned int nu = (unsigned int)std::ceil(2.0f * u_half / fstep);
        unsigned int nv = (unsigned int)std::ceil(2.0f * v_half / fstep);
        if (nu == 0) nu = 1;
        if (nv == 0) nv = 1;
        u_min  = u_center - (nu * fstep) * 0.5f;
        v_min_ = v_center - (nv * fstep) * 0.5f;

        if (verbose) {
            std::cout << "RTX pixel grid bench: " << nu << " x " << nv
                      << " = " << (unsigned long long)nu * nv
                      << " rays, step=" << fstep << "\n";
        }

        ensure_grid_bench(triangles);
        const size_t N = g_rtx_shared.N;
        const CUstream s = g_rtx_shared.stream;

        // Zero labels only - bounce arrays untouched (bench optimization)
        reset_bench_labels_only_grid();
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_grid_bench.d_force),  0,
                                   3 * sizeof(double), s));
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_grid_bench.d_moment), 0,
                                   3 * sizeof(double), s));

        RTXParamsGridBench h_params = {};
        h_params.bvh             = g_rtx_shared.bvh;
        h_params.sun             = sun_f;
        h_params.d_labels        = reinterpret_cast<int*>(g_rtx_shared.d_labels);
        h_params.d_force         = reinterpret_cast<double*>(g_grid_bench.d_force);
        h_params.d_moment        = reinterpret_cast<double*>(g_grid_bench.d_moment);
        h_params.d_normals       = reinterpret_cast<float3*>(g_rtx_shared.d_normals);
        h_params.d_areas         = reinterpret_cast<float*>(g_rtx_shared.d_areas);
        h_params.d_reflectance   = reinterpret_cast<float*>(g_rtx_shared.d_reflectance);
        h_params.d_specularity   = reinterpret_cast<float*>(g_rtx_shared.d_specularity);
        h_params.N               = (int)N;
        h_params.max_reflections = max_reflections;
        h_params.u_min           = u_min;
        h_params.v_min           = v_min_;
        h_params.grid_step       = fstep;
        h_params.u_axis          = u_axis;
        h_params.v_axis          = v_axis;
        h_params.t_start         = t_start;

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(g_grid_bench.d_params), &h_params,
                                   sizeof(RTXParamsGridBench), cudaMemcpyHostToDevice, s));

        OPTIX_CHECK(optixLaunch(g_grid_bench.pipeline, s,
                                g_grid_bench.d_params, sizeof(RTXParamsGridBench),
                                &g_grid_bench.sbt,
                                nu, nv, 1));

        // Post-filter on GPU: queued on stream s after optixLaunch, no extra sync needed.
        {
            const int thr = 256;
            const int blk = ((int)N + thr - 1) / thr;
            rtx_grid_bench_backface_filter<<<blk, thr, 0, s>>>(
                reinterpret_cast<int*>(g_rtx_shared.d_labels),
                reinterpret_cast<const float3*>(g_rtx_shared.d_normals),
                { sx, sy, sz }, (int)N);
        }
        CUDA_CHECK(cudaStreamSynchronize(s));

        // Readback force/moment - NO labels D2H, NO bounce array downloads
        double h_force[3]  = {};
        double h_moment[3] = {};
        CUDA_CHECK(cudaMemcpy(h_force,
                              reinterpret_cast<void*>(g_grid_bench.d_force),
                              3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_moment,
                              reinterpret_cast<void*>(g_grid_bench.d_moment),
                              3 * sizeof(double), cudaMemcpyDeviceToHost));

        // Reflections are now handled entirely in the shader (area_w = s², classical method).
        // NO CPU centroid pass; NO set_bounce_globals call.
        SRPResult res;
        res.labels       = {};   // bench: labels not transferred to host
        res.total_force  = { g_srp_phi0*h_force[0],  g_srp_phi0*h_force[1],  g_srp_phi0*h_force[2]  };
        res.total_moment = { g_srp_phi0*h_moment[0], g_srp_phi0*h_moment[1], g_srp_phi0*h_moment[2] };
        return res;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("RTX bench not available (pixel_grid_rtx_bench): ") + e.what());
    }
}
