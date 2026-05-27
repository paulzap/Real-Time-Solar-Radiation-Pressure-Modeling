// optix_shaders_center_bench.cu
// ============================================================
// Lean OptiX device programs for RTX Shadow bench (benchmark mode).
// No bounce tracking, no incident_dirs, no origin_pts.
// Designed for maximum throughput force/torque measurement.
//
// COMPILE TO PTX:
//   nvcc -ptx -arch=compute_75 ^
//        -I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0\include" ^
//        optix_shaders_center_bench.cu -o optix_shaders_center_bench.ptx
//
// DO NOT add this file to the regular CUDA compilation list.
// ============================================================

#include <optix.h>
#include <cuda_runtime.h>

// ---- minimal float3 helpers ----
inline __device__ float  dot3f(float3 a, float3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline __device__ float3 scale3f(float s, float3 v) { return {s*v.x, s*v.y, s*v.z}; }
inline __device__ float3 add3f(float3 a, float3 b)  { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline __device__ float3 neg3f(float3 v)             { return {-v.x, -v.y, -v.z}; }

// ---- params layout -- MUST match RTXParamsCenterBench in gpu_optix_center_raytracer_bench.cu ----
struct RTXParamsCenterBench {
    OptixTraversableHandle bvh;
    float3  sun;
    int*    d_labels;
    double* d_force;          // double: unified accumulator for direct + reflection forces
    double* d_moment;
    float3* d_centroids;
    float3* d_normals;
    float*  d_areas;
    float*  d_reflectance;
    float*  d_specularity;
    int     N;
    int     max_reflections;
    // NO bounce tracking fields (no d_bounce_levels, d_incident_dirs, d_origin_pts)
};

extern "C" __constant__ RTXParamsCenterBench params;

// ---------------------------------------------------------------------------
// SRP accumulation helper
// light_dir : unit vector FROM surface TOWARD light source
// cos_th    : dot(normal, light_dir) > 0
// area_w    : effective projected area weight
// n, c      : normal and moment-arm point of the receiving triangle
// ---------------------------------------------------------------------------
__device__ inline void accumulate_srp(float3 light_dir, float cos_th,
                                       float area_w, float alpha, float mu,
                                       float3 n, float3 c)
{
    const float sc  = -area_w * ((1.0f - alpha) + alpha*(1.0f - mu));                   //PHY
    const float nc  = -area_w * (2.0f*alpha*mu*cos_th + (2.0f/3.0f)*alpha*(1.0f - mu)); //PHY
    const float dfx = sc*light_dir.x + nc*n.x;                                           //PHY
    const float dfy = sc*light_dir.y + nc*n.y;                                           //PHY
    const float dfz = sc*light_dir.z + nc*n.z;                                           //PHY
    // double atomicAdd: consistent with GPU direct-force kernel; native on sm_75+
    atomicAdd(&params.d_force[0], (double)dfx);                                           //PHY
    atomicAdd(&params.d_force[1], (double)dfy);                                           //PHY
    atomicAdd(&params.d_force[2], (double)dfz);                                           //PHY
    atomicAdd(&params.d_moment[0], (double)(c.y*dfz - c.z*dfy));                         //PHY
    atomicAdd(&params.d_moment[1], (double)(c.z*dfx - c.x*dfz));                         //PHY
    atomicAdd(&params.d_moment[2], (double)(c.x*dfy - c.y*dfx));                         //PHY
}

// ============================================================
// Pipeline A -- RTX Shadow bench (center method, benchmark mode)
// Launch: (N, 1, 1) -- one thread per triangle
//
// Payload layout (2 registers):
//   p0: primitive index of hit, or 0xFFFFFFFF if missed
//   p1: hit t value (bits reinterpreted as float), valid when p0 != 0xFFFFFFFF
// ============================================================

extern "C" __global__ void __miss__center_bench() {
    optixSetPayload_0(0xFFFFFFFFu);
    optixSetPayload_1(0u);
}

extern "C" __global__ void __closesthit__center_bench() {
    optixSetPayload_0((unsigned int)optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

extern "C" __global__ void __raygen__center_bench() {
    const int tri = (int)optixGetLaunchIndex().x;
    if (tri >= params.N) return;

    const float3 n_tri  = params.d_normals[tri];
    const float  cos_th = dot3f(n_tri, params.sun);
    if (cos_th <= 0.0f) return;  // back face

    // Shadow ray: from centroid toward sun
    const float3 orig = add3f(params.d_centroids[tri],
                               scale3f(1e-4f, n_tri));   // offset to avoid self-hit
    unsigned int prim = 0xFFFFFFFFu, t_bits = 0u;
    optixTrace(
        params.bvh, orig, params.sun,
        1e-4f, 1e30f, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0u, 1u, 0u,
        prim, t_bits);

    if (prim != 0xFFFFFFFFu) return;  // shadowed

    // Lit: record label (bounce tracking removed)
    atomicOr(&params.d_labels[tri], 1);

    // Direct SRP is computed in double precision by the GPU kernel after launch.
    // d_force/d_moment here accumulates ONLY reflection forces.

    if (params.max_reflections <= 0) return;

    // ---- Reflection chain ----
    // Convention: s_eq = unit vector FROM surface TOWARD light source.
    //   refl = 2*(n . s_eq)*n - s_eq  (specular reflection direction)
    //   When refl hits triangle j: s_eq_j = -refl (light arrives from refl direction).
    float3 s_eq     = params.sun;
    float3 cur_n    = n_tri;
    float3 cur_orig = params.d_centroids[tri];
    float  intensity = params.d_reflectance[tri] * params.d_specularity[tri];

    for (int b = 0; b < params.max_reflections && intensity > 1e-10f; ++b) {
        const float cos_cur = dot3f(cur_n, s_eq);
        if (cos_cur <= 1e-6f) break;

        float3 refl = {
            2.0f*cos_cur*cur_n.x - s_eq.x,
            2.0f*cos_cur*cur_n.y - s_eq.y,
            2.0f*cos_cur*cur_n.z - s_eq.z
        };
        const float rlen = sqrtf(refl.x*refl.x + refl.y*refl.y + refl.z*refl.z);
        if (rlen < 1e-12f) break;
        refl = scale3f(1.0f / rlen, refl);

        float3 refl_orig = add3f(cur_orig, scale3f(1e-4f, cur_n));
        unsigned int j_prim = 0xFFFFFFFFu, j_t_bits = 0u;
        optixTrace(
            params.bvh, refl_orig, refl,
            1e-4f, 1e30f, 0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0u, 1u, 0u,
            j_prim, j_t_bits);

        if (j_prim == 0xFFFFFFFFu) break;

        const int    j      = (int)j_prim;
        const float3 j_n    = params.d_normals[j];
        const float3 s_eq_j = neg3f(refl);
        const float  cos_j  = dot3f(j_n, s_eq_j);
        if (cos_j <= 1e-6f) break;

        // Actual ray-triangle intersection point on j (not j's centroid) - used as
        // the moment arm for reflected-light force, mirroring the pixel-grid method.
        const float  t_j   = __uint_as_float(j_t_bits);
        const float3 hit_j = add3f(refl_orig, scale3f(t_j, refl));

        // Reflection contributes to force - do NOT set label on j (would corrupt label semantics).
        // area_w = A_tri * cos_th * intensity (emitter's projected area, conserved
        // through specular bounces). Using A_j * cos_j was wrong.
        // Accumulate into d_force/d_moment (reflection only; direct is handled by GPU kernel).
        accumulate_srp(s_eq_j, cos_j,
                       params.d_areas[tri] * cos_th * intensity,
                       params.d_reflectance[j], params.d_specularity[j],
                       j_n, hit_j);

        intensity *= params.d_reflectance[j] * params.d_specularity[j];
        cur_n     = j_n;
        cur_orig  = hit_j;
        s_eq      = s_eq_j;
    }
}
