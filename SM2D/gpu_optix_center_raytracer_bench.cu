// gpu_optix_center_raytracer_bench.cu
// ============================================================
// Benchmark-only RTX Shadow method (method 7 lean copy).
// No bounce tracking, no set_bounce_globals.
// Uses the same shared BVH/context as gpu_optix_shared.cu.
//
// Differences vs gpu_optix_center_raytracer.cu:
//   - reset_bench_labels_only(): only zeros d_labels (skips 3 bounce arrays
//     = ~172 MB of memsets for 5.4M triangles)
//   - No bounce array GPU→host copies (~172 MB saved per call)
//   - No set_bounce_globals call
//   - Loads optix_shaders_center_bench.ptx (no bounce writes in shader)
//
// Exports: calculate_labels_ray_casting_rtx_bench()
// ============================================================

#include "gpu_optix_shared_internal.h"
#include "bench_methods.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <array>

// ---------------------------------------------------------------------------
// RTXParamsCenterBench -- MUST exactly match the struct in
// optix_shaders_center_bench.cu. No bounce tracking fields.
// ---------------------------------------------------------------------------
struct RTXParamsCenterBench {
    OptixTraversableHandle bvh;
    float3  sun;
    int*    d_labels;
    double* d_force;
    double* d_moment;
    float3* d_centroids;
    float3* d_normals;
    float*  d_areas;
    float*  d_reflectance;
    float*  d_specularity;
    int     N;
    int     max_reflections;
};

// ---------------------------------------------------------------------------
// Pipeline state (private to this TU)
// ---------------------------------------------------------------------------
struct RTXCenterBenchState {
    bool built = false;

    OptixModule       module    = nullptr;
    OptixProgramGroup pg_raygen = nullptr;
    OptixProgramGroup pg_miss   = nullptr;
    OptixProgramGroup pg_hit    = nullptr;
    OptixPipeline     pipeline  = nullptr;

    OptixShaderBindingTable sbt = {};
    CUdeviceptr d_sbt_rg = 0, d_sbt_ms = 0, d_sbt_hg = 0;

    CUdeviceptr d_force    = 0;  // double[3] — unified accumulator for direct + reflection forces
    CUdeviceptr d_moment   = 0;  // double[3] — unified accumulator for direct + reflection torques
    CUdeviceptr d_params   = 0;  // sizeof(RTXParamsCenterBench)
};

static RTXCenterBenchState g_center_bench;

// ---------------------------------------------------------------------------
// load_ptx_center_bench
// ---------------------------------------------------------------------------
static std::string load_ptx_center_bench()
{
    const char* candidates[] = {
        "optix_shaders_center_bench.ptx",
        "Debug/optix_shaders_center_bench.ptx",
        "Release/optix_shaders_center_bench.ptx"
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
        "RTX bench not available: cannot find optix_shaders_center_bench.ptx "
        "in the working directory, Debug/, or Release/. "
        "Compile optix_shaders_center_bench.cu to PTX first:\n"
        "  nvcc -ptx -arch=<OptixPtxArch> "
        "-I\"<OptixRoot>/include\" "
        "optix_shaders_center_bench.cu -o optix_shaders_center_bench.ptx\n"
        "(OptixRoot and OptixPtxArch are defined in LocalPaths.props)");
}

// ---------------------------------------------------------------------------
// build_center_bench_pipeline  (called once after context exists)
// ---------------------------------------------------------------------------
static void build_center_bench_pipeline()
{
    std::string ptx_source = load_ptx_center_bench();

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
        log, &log_size, &g_center_bench.module));

    OptixProgramGroupOptions pg_opts = {};

    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      d.raygen.module = g_center_bench.module;
      d.raygen.entryFunctionName = "__raygen__center_bench";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_center_bench.pg_raygen)); }

    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      d.miss.module = g_center_bench.module;
      d.miss.entryFunctionName = "__miss__center_bench";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_center_bench.pg_miss)); }

    { OptixProgramGroupDesc d = {};
      d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      d.hitgroup.moduleCH            = g_center_bench.module;
      d.hitgroup.entryFunctionNameCH = "__closesthit__center_bench";
      log_size = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(g_rtx_shared.context, &d, 1,
                                          &pg_opts, log, &log_size, &g_center_bench.pg_hit)); }

    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;
    OptixProgramGroup pgs[] = { g_center_bench.pg_raygen, g_center_bench.pg_miss, g_center_bench.pg_hit };
    log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(g_rtx_shared.context, &pco, &plo,
                                    pgs, 3, log, &log_size, &g_center_bench.pipeline));
    OPTIX_CHECK(optixPipelineSetStackSize(g_center_bench.pipeline, 0, 0, 2048, 1));

    make_sbt_record(g_center_bench.pg_raygen, sizeof(RaygenRecord),   g_center_bench.d_sbt_rg);
    make_sbt_record(g_center_bench.pg_miss,   sizeof(MissRecord),     g_center_bench.d_sbt_ms);
    make_sbt_record(g_center_bench.pg_hit,    sizeof(HitgroupRecord), g_center_bench.d_sbt_hg);

    g_center_bench.sbt = {};
    g_center_bench.sbt.raygenRecord                = g_center_bench.d_sbt_rg;
    g_center_bench.sbt.missRecordBase              = g_center_bench.d_sbt_ms;
    g_center_bench.sbt.missRecordStrideInBytes     = sizeof(MissRecord);
    g_center_bench.sbt.missRecordCount             = 1;
    g_center_bench.sbt.hitgroupRecordBase          = g_center_bench.d_sbt_hg;
    g_center_bench.sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    g_center_bench.sbt.hitgroupRecordCount         = 1;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_center_bench.d_force),    3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_center_bench.d_moment),   3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_center_bench.d_params),   sizeof(RTXParamsCenterBench)));

    g_center_bench.built = true;
}

// ---------------------------------------------------------------------------
// ensure_center_bench
// ---------------------------------------------------------------------------
static void ensure_center_bench(const std::vector<Triangle>& triangles)
{
    ensure_rtx_common(triangles);
    if (!g_center_bench.built) build_center_bench_pipeline();
}

// ---------------------------------------------------------------------------
// reset_bench_labels_only -- zeros only d_labels; does NOT touch the
// bounce-tracking arrays (d_bounce_levels, d_incident_dirs, d_origin_pts).
// For 5.4M triangles this saves ~172 MB of async memsets per call.
// ---------------------------------------------------------------------------
static void reset_bench_labels_only()
{
    CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_rtx_shared.d_labels), 0x00,
                               g_rtx_shared.N * sizeof(int), g_rtx_shared.stream));
}

// ---------------------------------------------------------------------------
// double atomicAdd compatibility shim for SM < 6.0 (compute_52 gencode target).
// SM 6.0+ has native double atomicAdd; older architectures need CAS emulation.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ static double atomicAdd(double* addr, double val)
{
    unsigned long long int* addr_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int  old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// ---------------------------------------------------------------------------
// rtx_center_bench_backface_filter
// GPU kernel: re-checks labels after optixLaunch with a tighter threshold
// (1e-5f >> float32 eps ~1.2e-7) to eliminate stripe artifacts where
// float32 normal·sun rounding returns a tiny positive value for edge-on
// triangles. Queued on the same stream as optixLaunch - near-zero overhead.
// ---------------------------------------------------------------------------
__global__ void rtx_center_bench_backface_filter(
    int* d_labels, const float3* d_normals, float3 sun, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !d_labels[i]) return;
    float3 n = d_normals[i];
    if (n.x*sun.x + n.y*sun.y + n.z*sun.z <= 1e-5f) d_labels[i] = 0;
}

// ---------------------------------------------------------------------------
// rtx_center_bench_compute_force
// GPU kernel: computes SRP force/moment in double precision from the
// post-filtered labels. Runs on the same stream directly after
// rtx_center_bench_backface_filter so it reads the corrected labels.
// Uses double atomicAdd (requires CC ≥ 6.0; this hardware is CC 8.x).
// SRP: area_w=area*cos_th; sc=-area_w*(1-alpha*mu); nc=-area_w*(2*alpha*mu*cos_th+2/3*alpha*(1-mu))
// F = sc*sun + nc*normal;  M = centroid × F
// ---------------------------------------------------------------------------
__global__ void rtx_center_bench_compute_force(
    const int*    d_labels,
    const float3* d_normals,
    const float3* d_centroids,
    const float*  d_areas,
    const float*  d_reflectance,
    const float*  d_specularity,
    float3        sun,
    double*       d_force_out,
    double*       d_moment_out,
    int           N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !d_labels[i]) return;

    float3 n     = d_normals[i];
    double cos_th = (double)n.x*(double)sun.x + (double)n.y*(double)sun.y + (double)n.z*(double)sun.z;
    if (cos_th <= 0.0) return;

    double area  = (double)d_areas[i];
    double alpha = (double)d_reflectance[i];
    double mu    = (double)d_specularity[i];
    if (alpha == 0.0 && mu == 0.0) { alpha = 0.5; mu = 0.5; }  // legacy CSV fallback

    double area_w = area * cos_th;
    double sc = -area_w * ((1.0 - alpha) + alpha * (1.0 - mu));
    double nc = -area_w * (2.0 * alpha * mu * cos_th + (2.0/3.0) * alpha * (1.0 - mu));

    double fx = sc * (double)sun.x + nc * (double)n.x;
    double fy = sc * (double)sun.y + nc * (double)n.y;
    double fz = sc * (double)sun.z + nc * (double)n.z;

    atomicAdd(&d_force_out[0], fx);
    atomicAdd(&d_force_out[1], fy);
    atomicAdd(&d_force_out[2], fz);

    float3 c = d_centroids[i];
    atomicAdd(&d_moment_out[0], (double)c.y * fz - (double)c.z * fy);
    atomicAdd(&d_moment_out[1], (double)c.z * fx - (double)c.x * fz);
    atomicAdd(&d_moment_out[2], (double)c.x * fy - (double)c.y * fx);
}

// ---------------------------------------------------------------------------
// calculate_labels_ray_casting_rtx_bench  (benchmark-only method 7)
// ---------------------------------------------------------------------------
SRPResult calculate_labels_ray_casting_rtx_bench(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    int max_reflections)
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

        ensure_center_bench(triangles);
        const size_t N = g_rtx_shared.N;
        const CUstream s = g_rtx_shared.stream;

        // Zero labels and force accumulators - bounce arrays untouched (bench optimization)
        reset_bench_labels_only();
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_center_bench.d_force),  0,
                                   3 * sizeof(double), s));
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(g_center_bench.d_moment), 0,
                                   3 * sizeof(double), s));

        RTXParamsCenterBench h_params = {};
        h_params.bvh           = g_rtx_shared.bvh;
        h_params.sun           = { sx, sy, sz };
        h_params.d_labels      = reinterpret_cast<int*>(g_rtx_shared.d_labels);
        h_params.d_force       = reinterpret_cast<double*>(g_center_bench.d_force);
        h_params.d_moment      = reinterpret_cast<double*>(g_center_bench.d_moment);
        h_params.d_centroids   = reinterpret_cast<float3*>(g_rtx_shared.d_centroids);
        h_params.d_normals     = reinterpret_cast<float3*>(g_rtx_shared.d_normals);
        h_params.d_areas       = reinterpret_cast<float*>(g_rtx_shared.d_areas);
        h_params.d_reflectance = reinterpret_cast<float*>(g_rtx_shared.d_reflectance);
        h_params.d_specularity = reinterpret_cast<float*>(g_rtx_shared.d_specularity);
        h_params.N             = (int)N;
        h_params.max_reflections = max_reflections;

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(g_center_bench.d_params), &h_params,
                                   sizeof(RTXParamsCenterBench), cudaMemcpyHostToDevice, s));

        OPTIX_CHECK(optixLaunch(g_center_bench.pipeline, s,
                                g_center_bench.d_params, sizeof(RTXParamsCenterBench),
                                &g_center_bench.sbt,
                                (unsigned int)N, 1, 1));

        // Post-process on GPU - both kernels queued on stream s, no extra sync needed.
        // backface_filter runs first (fixes float32 rounding stripes), then compute_force
        // reads the corrected labels and accumulates double-precision force/moment.
        {
            const int thr = 256;
            const int blk = ((int)N + thr - 1) / thr;
            rtx_center_bench_backface_filter<<<blk, thr, 0, s>>>(
                reinterpret_cast<int*>(g_rtx_shared.d_labels),
                reinterpret_cast<const float3*>(g_rtx_shared.d_normals),
                { sx, sy, sz }, (int)N);
            rtx_center_bench_compute_force<<<blk, thr, 0, s>>>(
                reinterpret_cast<const int*>(g_rtx_shared.d_labels),
                reinterpret_cast<const float3*>(g_rtx_shared.d_normals),
                reinterpret_cast<const float3*>(g_rtx_shared.d_centroids),
                reinterpret_cast<const float*>(g_rtx_shared.d_areas),
                reinterpret_cast<const float*>(g_rtx_shared.d_reflectance),
                reinterpret_cast<const float*>(g_rtx_shared.d_specularity),
                { sx, sy, sz },
                reinterpret_cast<double*>(g_center_bench.d_force),
                reinterpret_cast<double*>(g_center_bench.d_moment),
                (int)N);
        }
        CUDA_CHECK(cudaStreamSynchronize(s));

        // Readback unified double-precision force (direct + reflection) - NO labels D2H, NO bounce array downloads
        double h_force[3]  = {};
        double h_moment[3] = {};
        CUDA_CHECK(cudaMemcpy(h_force,
                              reinterpret_cast<void*>(g_center_bench.d_force),
                              3 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_moment,
                              reinterpret_cast<void*>(g_center_bench.d_moment),
                              3 * sizeof(double), cudaMemcpyDeviceToHost));

        // NO set_bounce_globals call
        SRPResult res;
        res.labels       = {};   // bench: labels not transferred to host
        res.total_force  = { g_srp_phi0*h_force[0],
                             g_srp_phi0*h_force[1],
                             g_srp_phi0*h_force[2] };
        res.total_moment = { g_srp_phi0*h_moment[0],
                             g_srp_phi0*h_moment[1],
                             g_srp_phi0*h_moment[2] };
        return res;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("RTX bench not available (ray_casting_rtx_bench): ") + e.what());
    }
}
