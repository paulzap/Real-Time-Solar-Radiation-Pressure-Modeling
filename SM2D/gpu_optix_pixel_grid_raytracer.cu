// gpu_optix_pixel_grid_raytracer.cu
// ============================================================
// RTX Pixel Grid method (method 8): one ray per grid cell.
// Accumulates force/moment into double[3] - required because
// fine grids produce >8.4M rays, exceeding the float-atomicAdd
// saturation limit (float eps ~1.2e-7).
//
// Shader entry points live in optix_shaders_pixel_grid.cu;
// compile that file to PTX before running:
//
//   nvcc -ptx -arch=compute_75 ^
//        -I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0\include" ^
//        optix_shaders_pixel_grid.cu -o optix_shaders_pixel_grid.ptx
// ============================================================

#include "gpu_optix_shared_internal.h"
#include "gpu_optix_raytracer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <array>

// ---------------------------------------------------------------------------
// RTXParamsGrid -- MUST exactly match the struct in optix_shaders_pixel_grid.cu
// double* d_force / d_moment: avoids saturation for grids up to ~4.5e15 rays
// ---------------------------------------------------------------------------
struct RTXParamsGrid {
    OptixTraversableHandle bvh;
    float3  sun;
    int*    d_labels;
    double* d_force;          // double: no saturation up to ~4.5e15 rays
    double* d_moment;
    float3* d_normals;
    float*  d_areas;
    float*  d_reflectance;
    float*  d_specularity;
    int     N;
    int     max_reflections;
    int*    d_bounce_levels;
    float*  d_incident_dirs;
    float*  d_origin_pts;
    // pixel-grid specific
    float   u_min, v_min, grid_step;
    float3  u_axis, v_axis;
    float   t_start;
};

// ---------------------------------------------------------------------------
// Pipeline state (private to this TU)
// ---------------------------------------------------------------------------
struct RTXGridState {
    bool built = false;

    OptixModule       module   = nullptr;
    OptixProgramGroup pg_raygen = nullptr;
    OptixProgramGroup pg_miss   = nullptr;
    OptixProgramGroup pg_hit    = nullptr;
    OptixPipeline     pipeline  = nullptr;

    OptixShaderBindingTable sbt = {};
    CUdeviceptr d_sbt_rg = 0, d_sbt_ms = 0, d_sbt_hg = 0;

    CUdeviceptr d_force  = 0;  // double[3]
    CUdeviceptr d_moment = 0;  // double[3]
    CUdeviceptr d_params = 0;  // sizeof(RTXParamsGrid)
};

static RTXGridState g_grid;

// ---------------------------------------------------------------------------
// load_ptx_grid
// ---------------------------------------------------------------------------
static std::string load_ptx_grid()
{
    const char* candidates[] = {
        "optix_shaders_pixel_grid.ptx",
        "Debug/optix_shaders_pixel_grid.ptx",
        "Release/optix_shaders_pixel_grid.ptx"
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
        "RTX/OptiX not available: cannot find optix_shaders_pixel_grid.ptx "
        "in the working directory, Debug/, or Release/. "
        "Compile optix_shaders_pixel_grid.cu to PTX first (see file header for the command).");
}

// ---------------------------------------------------------------------------
// build_grid_pipeline
// ---------------------------------------------------------------------------
static void build_grid_pipeline()
{
    std::string ptx_source = load_ptx_grid();

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
        log, &log_size, &g_grid.module));

    OptixProgramGroupOptions pg_opts = {};

    // raygen
    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      d.raygen.module = g_grid.module;
      d.raygen.entryFunctionName = "__raygen__pixel_grid";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_grid.pg_raygen)); }
    // miss
    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      d.miss.module = g_grid.module;
      d.miss.entryFunctionName = "__miss__pixel_grid";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_grid.pg_miss)); }
    // closesthit
    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      d.hitgroup.moduleCH            = g_grid.module;
      d.hitgroup.entryFunctionNameCH = "__closesthit__pixel_grid";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_grid.pg_hit)); }

    // Pipeline
    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;
    OptixProgramGroup pgs[] = { g_grid.pg_raygen, g_grid.pg_miss, g_grid.pg_hit };
    log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(g_rtx_shared.context, &pco, &plo,
                                    pgs, 3, log, &log_size, &g_grid.pipeline));
    OPTIX_CHECK(optixPipelineSetStackSize(g_grid.pipeline, 0, 0, 2048, 1));

    // SBT
    make_sbt_record(g_grid.pg_raygen, sizeof(RaygenRecord),   g_grid.d_sbt_rg);
    make_sbt_record(g_grid.pg_miss,   sizeof(MissRecord),     g_grid.d_sbt_ms);
    make_sbt_record(g_grid.pg_hit,    sizeof(HitgroupRecord), g_grid.d_sbt_hg);

    g_grid.sbt = {};
    g_grid.sbt.raygenRecord                = g_grid.d_sbt_rg;
    g_grid.sbt.missRecordBase              = g_grid.d_sbt_ms;
    g_grid.sbt.missRecordStrideInBytes     = sizeof(MissRecord);
    g_grid.sbt.missRecordCount             = 1;
    g_grid.sbt.hitgroupRecordBase          = g_grid.d_sbt_hg;
    g_grid.sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    g_grid.sbt.hitgroupRecordCount         = 1;

    // Force / moment / params buffers
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_grid.d_force),  3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_grid.d_moment), 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_grid.d_params), sizeof(RTXParamsGrid)));

    g_grid.built = true;
}

// ---------------------------------------------------------------------------
// ensure_grid
// ---------------------------------------------------------------------------
static void ensure_grid(const std::vector<Triangle>& triangles)
{
    ensure_rtx_common(triangles);
    if (!g_grid.built) build_grid_pipeline();
}

// ---------------------------------------------------------------------------
// calculate_labels_pixel_grid_rtx  (method 8)
// ---------------------------------------------------------------------------
SRPResult calculate_labels_pixel_grid_rtx(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections,
    bool verbose,
    const std::string& primary_emitter_name)
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

        // Center the grid symmetrically to avoid a spurious moment bias
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
            std::cout << "RTX pixel grid: " << nu << " x " << nv
                      << " = " << (unsigned long long)nu * nv
                      << " rays, step=" << fstep << "\n";
        }

        ensure_grid(triangles);
        const size_t N = g_rtx_shared.N;
        const CUstream s = g_rtx_shared.stream;

        // Reset per-call buffers
        reset_rtx_per_call_buffers();
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_grid.d_force),  0,
                                   3 * sizeof(double), s));
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_grid.d_moment), 0,
                                   3 * sizeof(double), s));

        // Build and upload params
        RTXParamsGrid h_params = {};
        h_params.bvh             = g_rtx_shared.bvh;
        h_params.sun             = sun_f;
        h_params.d_labels        = reinterpret_cast<int*>(g_rtx_shared.d_labels);
        h_params.d_bounce_levels = reinterpret_cast<int*>(g_rtx_shared.d_bounce_levels);
        h_params.d_incident_dirs = reinterpret_cast<float*>(g_rtx_shared.d_incident_dirs);
        h_params.d_origin_pts    = reinterpret_cast<float*>(g_rtx_shared.d_origin_pts);
        h_params.d_force         = reinterpret_cast<double*>(g_grid.d_force);
        h_params.d_moment        = reinterpret_cast<double*>(g_grid.d_moment);
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

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(g_grid.d_params), &h_params,
                                   sizeof(RTXParamsGrid), cudaMemcpyHostToDevice, s));

        // Launch
        OPTIX_CHECK(optixLaunch(g_grid.pipeline, s,
                                g_grid.d_params, sizeof(RTXParamsGrid),
                                &g_grid.sbt,
                                nu, nv, 1));
        CUDA_CHECK(cudaStreamSynchronize(s));

        // Readback
        std::vector<int> labels(N);
        double h_force[3]  = {};
        double h_moment[3] = {};
        CUDA_CHECK(cudaMemcpy(labels.data(),
                              reinterpret_cast<void*>(g_rtx_shared.d_labels),
                              N * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_force,
                              reinterpret_cast<void*>(g_grid.d_force),
                              3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_moment,
                              reinterpret_cast<void*>(g_grid.d_moment),
                              3 * sizeof(double), cudaMemcpyDeviceToHost));
        h_force[0]  *= g_srp_phi0; h_force[1]  *= g_srp_phi0; h_force[2]  *= g_srp_phi0;
        h_moment[0] *= g_srp_phi0; h_moment[1] *= g_srp_phi0; h_moment[2] *= g_srp_phi0;

        // Post-process: double-precision back-face threshold, matching GPU Shadow (dot > 1e-6).
        // Eliminates stripe artifacts on panels with near-grazing sun angles (float32 rounding).
        double sx_d = sun_vector[0] / slen, sy_d = sun_vector[1] / slen, sz_d = sun_vector[2] / slen;
        for (size_t i = 0; i < N; ++i) {
            if (labels[i] != 0) {
                double cos_dp = triangles[i].normal_x * sx_d +
                                triangles[i].normal_y * sy_d +
                                triangles[i].normal_z * sz_d;
                if (cos_dp <= 1e-6) labels[i] = 0;
            }
        }

        std::vector<int>   bounce_levels(N);
        std::vector<float> h_inc(N * 3);
        std::vector<float> h_orig(N * 3);
        CUDA_CHECK(cudaMemcpy(bounce_levels.data(),
                              reinterpret_cast<void*>(g_rtx_shared.d_bounce_levels),
                              N * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_inc.data(),
                              reinterpret_cast<void*>(g_rtx_shared.d_incident_dirs),
                              N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_orig.data(),
                              reinterpret_cast<void*>(g_rtx_shared.d_origin_pts),
                              N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < N; ++i)
            if (bounce_levels[i] > 100) bounce_levels[i] = -1;

        // Primary emitter filter
        if (!primary_emitter_name.empty()) {
            for (size_t i = 0; i < N; ++i) {
                if (bounce_levels[i] == 0 &&
                    triangles[i].component_name != primary_emitter_name) {
                    labels[i]        = 0;
                    bounce_levels[i] = -1;
                }
            }
            SRPResult filtered = compute_srp_forces(triangles, labels,
                sx_d, sy_d, sz_d);
            h_force[0]  = filtered.total_force[0];
            h_force[1]  = filtered.total_force[1];
            h_force[2]  = filtered.total_force[2];
            h_moment[0] = filtered.total_moment[0];
            h_moment[1] = filtered.total_moment[1];
            h_moment[2] = filtered.total_moment[2];
        }

        {
            const double dNaN = std::numeric_limits<double>::quiet_NaN();
            std::vector<std::array<double,3>> incident_dirs(N, {0.0, 0.0, 0.0});
            std::vector<std::array<double,3>> origin_pts(N, {dNaN, dNaN, dNaN});
            for (size_t i = 0; i < N; ++i) {
                incident_dirs[i] = { (double)h_inc[i*3],   (double)h_inc[i*3+1],  (double)h_inc[i*3+2]  };
                origin_pts[i]    = { (double)h_orig[i*3],  (double)h_orig[i*3+1], (double)h_orig[i*3+2] };
            }
            set_bounce_globals(std::move(bounce_levels),
                               std::move(incident_dirs),
                               std::move(origin_pts));
        }

        SRPResult res;
        res.labels       = std::move(labels);
        res.total_force  = { h_force[0],  h_force[1],  h_force[2]  };
        res.total_moment = { h_moment[0], h_moment[1], h_moment[2] };
        return res;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("RTX/OptiX not available (pixel_grid_rtx): ") + e.what());
    }
}
