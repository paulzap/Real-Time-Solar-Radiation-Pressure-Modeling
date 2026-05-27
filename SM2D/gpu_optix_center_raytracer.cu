// gpu_optix_center_raytracer.cu
// ============================================================
// RTX Shadow method (method 7): one ray per triangle centroid.
// Accumulates force/moment into float[3] - safe because
// N_triangles << 8.4M (the float-atomicAdd saturation limit).
//
// Shader entry points live in optix_shaders_center.cu;
// compile that file to PTX before running:
//
//   nvcc -ptx -arch=compute_75 ^
//        -I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0\include" ^
//        optix_shaders_center.cu -o optix_shaders_center.ptx
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
// RTXParamsCenter -- MUST exactly match the struct in optix_shaders_center.cu
// float* d_force / d_moment: safe for N_triangles up to ~8 million
// ---------------------------------------------------------------------------
struct RTXParamsCenter {
    OptixTraversableHandle bvh;
    float3  sun;
    int*    d_labels;
    float*  d_force;          // float: direct-illumination SRP only
    float*  d_moment;
    float*  d_refl_force;     // float: reflection-bounce SRP only
    float*  d_refl_moment;
    float3* d_centroids;
    float3* d_normals;
    float*  d_areas;
    float*  d_reflectance;
    float*  d_specularity;
    int     N;
    int     max_reflections;
    int*    d_bounce_levels;
    float*  d_incident_dirs;
    float*  d_origin_pts;
};

// ---------------------------------------------------------------------------
// Pipeline state (private to this TU)
// ---------------------------------------------------------------------------
struct RTXCenterState {
    bool built = false;

    OptixModule       module   = nullptr;
    OptixProgramGroup pg_raygen = nullptr;
    OptixProgramGroup pg_miss   = nullptr;
    OptixProgramGroup pg_hit    = nullptr;
    OptixPipeline     pipeline  = nullptr;

    OptixShaderBindingTable sbt = {};
    CUdeviceptr d_sbt_rg = 0, d_sbt_ms = 0, d_sbt_hg = 0;

    CUdeviceptr d_force       = 0;  // float[3] - direct illumination SRP
    CUdeviceptr d_moment      = 0;  // float[3]
    CUdeviceptr d_refl_force  = 0;  // float[3] - reflection-bounce SRP
    CUdeviceptr d_refl_moment = 0;  // float[3]
    CUdeviceptr d_params      = 0;  // sizeof(RTXParamsCenter)
};

static RTXCenterState g_center;

// ---------------------------------------------------------------------------
// load_ptx_center
// ---------------------------------------------------------------------------
static std::string load_ptx_center()
{
    const char* candidates[] = {
        "optix_shaders_center.ptx",
        "Debug/optix_shaders_center.ptx",
        "Release/optix_shaders_center.ptx"
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
        "RTX/OptiX not available: cannot find optix_shaders_center.ptx "
        "in the working directory, Debug/, or Release/. "
        "Compile optix_shaders_center.cu to PTX first (see file header for the command).");
}

// ---------------------------------------------------------------------------
// build_center_pipeline  (called once after context exists)
// ---------------------------------------------------------------------------
static void build_center_pipeline()
{
    std::string ptx_source = load_ptx_center();

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
        log, &log_size, &g_center.module));

    OptixProgramGroupOptions pg_opts = {};

    // raygen
    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      d.raygen.module = g_center.module;
      d.raygen.entryFunctionName = "__raygen__center";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_center.pg_raygen)); }
    // miss
    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      d.miss.module = g_center.module;
      d.miss.entryFunctionName = "__miss__center";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_center.pg_miss)); }
    // closesthit
    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      d.hitgroup.moduleCH            = g_center.module;
      d.hitgroup.entryFunctionNameCH = "__closesthit__center";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_center.pg_hit)); }

    // Pipeline
    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;
    OptixProgramGroup pgs[] = { g_center.pg_raygen, g_center.pg_miss, g_center.pg_hit };
    log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(g_rtx_shared.context, &pco, &plo,
                                    pgs, 3, log, &log_size, &g_center.pipeline));
    OPTIX_CHECK(optixPipelineSetStackSize(g_center.pipeline, 0, 0, 2048, 1));

    // SBT
    make_sbt_record(g_center.pg_raygen, sizeof(RaygenRecord),   g_center.d_sbt_rg);
    make_sbt_record(g_center.pg_miss,   sizeof(MissRecord),     g_center.d_sbt_ms);
    make_sbt_record(g_center.pg_hit,    sizeof(HitgroupRecord), g_center.d_sbt_hg);

    g_center.sbt = {};
    g_center.sbt.raygenRecord                = g_center.d_sbt_rg;
    g_center.sbt.missRecordBase              = g_center.d_sbt_ms;
    g_center.sbt.missRecordStrideInBytes     = sizeof(MissRecord);
    g_center.sbt.missRecordCount             = 1;
    g_center.sbt.hitgroupRecordBase          = g_center.d_sbt_hg;
    g_center.sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    g_center.sbt.hitgroupRecordCount         = 1;

    // Force / moment / params buffers
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_center.d_force),       3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_center.d_moment),      3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_center.d_refl_force),  3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_center.d_refl_moment), 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_center.d_params),      sizeof(RTXParamsCenter)));

    g_center.built = true;
}

// ---------------------------------------------------------------------------
// ensure_center
// ---------------------------------------------------------------------------
static void ensure_center(const std::vector<Triangle>& triangles)
{
    ensure_rtx_common(triangles);
    if (!g_center.built) build_center_pipeline();
}

// ---------------------------------------------------------------------------
// calculate_labels_ray_casting_rtx  (method 7)
// ---------------------------------------------------------------------------
SRPResult calculate_labels_ray_casting_rtx(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    int max_reflections,
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

        ensure_center(triangles);
        const size_t N = g_rtx_shared.N;
        const CUstream s = g_rtx_shared.stream;

        // Reset per-call buffers
        reset_rtx_per_call_buffers();
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_center.d_force),       0,
                                   3 * sizeof(float), s));
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_center.d_moment),      0,
                                   3 * sizeof(float), s));
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_center.d_refl_force),  0,
                                   3 * sizeof(float), s));
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_center.d_refl_moment), 0,
                                   3 * sizeof(float), s));

        // Build and upload params
        RTXParamsCenter h_params = {};
        h_params.bvh             = g_rtx_shared.bvh;
        h_params.sun             = { sx, sy, sz };
        h_params.d_labels        = reinterpret_cast<int*>(g_rtx_shared.d_labels);
        h_params.d_bounce_levels = reinterpret_cast<int*>(g_rtx_shared.d_bounce_levels);
        h_params.d_incident_dirs = reinterpret_cast<float*>(g_rtx_shared.d_incident_dirs);
        h_params.d_origin_pts    = reinterpret_cast<float*>(g_rtx_shared.d_origin_pts);
        h_params.d_force         = reinterpret_cast<float*>(g_center.d_force);
        h_params.d_moment        = reinterpret_cast<float*>(g_center.d_moment);
        h_params.d_refl_force    = reinterpret_cast<float*>(g_center.d_refl_force);
        h_params.d_refl_moment   = reinterpret_cast<float*>(g_center.d_refl_moment);
        h_params.d_centroids     = reinterpret_cast<float3*>(g_rtx_shared.d_centroids);
        h_params.d_normals       = reinterpret_cast<float3*>(g_rtx_shared.d_normals);
        h_params.d_areas         = reinterpret_cast<float*>(g_rtx_shared.d_areas);
        h_params.d_reflectance   = reinterpret_cast<float*>(g_rtx_shared.d_reflectance);
        h_params.d_specularity   = reinterpret_cast<float*>(g_rtx_shared.d_specularity);
        h_params.N               = (int)N;
        h_params.max_reflections = max_reflections;

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(g_center.d_params), &h_params,
                                   sizeof(RTXParamsCenter), cudaMemcpyHostToDevice, s));

        // Launch
        OPTIX_CHECK(optixLaunch(g_center.pipeline, s,
                                g_center.d_params, sizeof(RTXParamsCenter),
                                &g_center.sbt,
                                (unsigned int)N, 1, 1));
        CUDA_CHECK(cudaStreamSynchronize(s));

        // Readback labels and bounce arrays - float32 force/moment not downloaded (computed in double on CPU)
        std::vector<int> labels(N);
        CUDA_CHECK(cudaMemcpy(labels.data(),
                              reinterpret_cast<void*>(g_rtx_shared.d_labels),
                              N * sizeof(int), cudaMemcpyDeviceToHost));

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

        // Direct-illumination forces: recomputed in double precision on CPU to
        // eliminate the ~1% float accumulation error present in the GPU shader.
        auto result = compute_srp_forces(triangles, std::move(labels), sx_d, sy_d, sz_d);

        // Reflection forces: accumulated correctly in the GPU shader (separate
        // d_refl_force/d_refl_moment buffers, each bounce uses the actual incident
        // direction and intensity - cannot be replicated on CPU without per-polygon
        // intensity data).  Cast float→double; error is negligible because
        // reflection contributions are small relative to direct illumination.
        float h_refl_force[3] = {}, h_refl_moment[3] = {};
        CUDA_CHECK(cudaMemcpy(h_refl_force,
                              reinterpret_cast<void*>(g_center.d_refl_force),
                              3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_refl_moment,
                              reinterpret_cast<void*>(g_center.d_refl_moment),
                              3 * sizeof(float), cudaMemcpyDeviceToHost));
        result.total_force[0]  += g_srp_phi0*(double)h_refl_force[0];
        result.total_force[1]  += g_srp_phi0*(double)h_refl_force[1];
        result.total_force[2]  += g_srp_phi0*(double)h_refl_force[2];
        result.total_moment[0] += g_srp_phi0*(double)h_refl_moment[0];
        result.total_moment[1] += g_srp_phi0*(double)h_refl_moment[1];
        result.total_moment[2] += g_srp_phi0*(double)h_refl_moment[2];
        return result;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("RTX/OptiX not available (ray_casting_rtx): ") + e.what());
    }
}
