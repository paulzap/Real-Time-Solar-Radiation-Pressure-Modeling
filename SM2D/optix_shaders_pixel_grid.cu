// optix_shaders_pixel_grid.cu
// ============================================================
// OptiX device programs for RTX Pixel Grid method (method 8).
// Force/moment accumulators are double* - required because
// fine grids can exceed 8.4M rays, which saturates float atomicAdd.
// double atomicAdd is hardware-native on sm_75+ (all RTX GPUs).
//
// COMPILE TO PTX:
//   nvcc -ptx -arch=compute_75 ^
//        -I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0\include" ^
//        optix_shaders_pixel_grid.cu -o optix_shaders_pixel_grid.ptx
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

// ---- params layout -- MUST match RTXParamsGrid in gpu_optix_pixel_grid_raytracer.cu ----
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

extern "C" __constant__ RTXParamsGrid params;

// ---------------------------------------------------------------------------
// SRP accumulation helper
// Computes in float, accumulates into double to prevent saturation.
// ---------------------------------------------------------------------------
__device__ inline void accumulate_srp(float3 light_dir, float cos_th,
                                       float area_w, float alpha, float mu,
                                       float3 n, float3 c)
{
    const float sc  = -area_w * ((1.0f - alpha) + alpha*(1.0f - mu));                    //PHY
    const float nc  = -area_w * (2.0f*alpha*mu*cos_th + (2.0f/3.0f)*alpha*(1.0f - mu)); //PHY
    const float dfx = sc*light_dir.x + nc*n.x;                                           //PHY
    const float dfy = sc*light_dir.y + nc*n.y;                                           //PHY
    const float dfz = sc*light_dir.z + nc*n.z;                                           //PHY
    // double atomicAdd: prevents saturation when N_rays > ~8.4M (sm_75+ native)
    atomicAdd(&params.d_force[0], (double)dfx);                                           //PHY
    atomicAdd(&params.d_force[1], (double)dfy);                                           //PHY
    atomicAdd(&params.d_force[2], (double)dfz);                                           //PHY
    atomicAdd(&params.d_moment[0], (double)(c.y*dfz - c.z*dfy));                         //PHY
    atomicAdd(&params.d_moment[1], (double)(c.z*dfx - c.x*dfz));                         //PHY
    atomicAdd(&params.d_moment[2], (double)(c.x*dfy - c.y*dfx));                         //PHY
}

// ============================================================
// Pipeline B -- RTX Grid (pixel-grid method, method 8)
// Launch: (nu, nv, 1) -- one thread per grid cell
//
// Payload layout (2 registers):
//   p0: primitive index, or 0xFFFFFFFF if no hit
//   p1: hit t value (bits reinterpreted as float), valid when p0 != 0xFFFFFFFF
// ============================================================

extern "C" __global__ void __miss__pixel_grid() {
    optixSetPayload_0(0xFFFFFFFFu);
    optixSetPayload_1(0u);
}

extern "C" __global__ void __closesthit__pixel_grid() {
    optixSetPayload_0((unsigned int)optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

extern "C" __global__ void __raygen__pixel_grid() {
    const int iu = (int)optixGetLaunchIndex().x;
    const int iv = (int)optixGetLaunchIndex().y;

    const float uc = params.u_min + (iu + 0.5f) * params.grid_step;
    const float vc = params.v_min + (iv + 0.5f) * params.grid_step;

    const float3 ray_o = add3f(add3f(scale3f(uc, params.u_axis),
                                     scale3f(vc, params.v_axis)),
                               scale3f(params.t_start, params.sun));
    const float3 ray_d = neg3f(params.sun);

    // Primary ray
    unsigned int prim = 0xFFFFFFFFu, t_bits = 0u;
    optixTrace(
        params.bvh, ray_o, ray_d,
        1e-4f, 1e30f, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0u, 1u, 0u,
        prim, t_bits);

    if (prim == 0xFFFFFFFFu) return;

    const int    k      = (int)prim;
    const float3 n_k    = params.d_normals[k];
    const float  cos_th = dot3f(n_k, params.sun);
    if (cos_th <= 0.0f) return;

    atomicOr(&params.d_labels[k], 1);
    atomicMin(&params.d_bounce_levels[k], 0);

    int prev_k = atomicCAS(reinterpret_cast<int*>(&params.d_incident_dirs[k * 3]),
                           -1, __float_as_int(params.sun.x));
    if (prev_k == -1) {
        params.d_incident_dirs[k * 3 + 1] = params.sun.y;
        params.d_incident_dirs[k * 3 + 2] = params.sun.z;
    }

    const float s2 = params.grid_step * params.grid_step;

    // Use the ACTUAL HIT POINT as the moment arm (not the centroid).
    // The centered grid ensures symmetric shapes produce zero net moment.
    const float  t_k   = __uint_as_float(t_bits);
    const float3 hit_k = add3f(ray_o, scale3f(t_k, ray_d));
    accumulate_srp(params.sun, cos_th, s2,
                   params.d_reflectance[k], params.d_specularity[k],
                   n_k, hit_k);

    if (params.max_reflections <= 0) return;

    // ---- Reflection bounces ----
    float3 s_eq     = params.sun;
    float3 cur_n    = n_k;
    float3 cur_orig = hit_k;
    float  intensity = params.d_reflectance[k] * params.d_specularity[k];

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

        atomicOr(&params.d_labels[j], 1);
        atomicMin(&params.d_bounce_levels[j], b + 1);

        int prev_j = atomicCAS(reinterpret_cast<int*>(&params.d_incident_dirs[j * 3]),
                               -1, __float_as_int(s_eq_j.x));
        if (prev_j == -1) {
            params.d_incident_dirs[j * 3 + 1] = s_eq_j.y;
            params.d_incident_dirs[j * 3 + 2] = s_eq_j.z;
            params.d_origin_pts[j * 3 + 0]    = cur_orig.x;
            params.d_origin_pts[j * 3 + 1]    = cur_orig.y;
            params.d_origin_pts[j * 3 + 2]    = cur_orig.z;
        }

        const float  t_j   = __uint_as_float(j_t_bits);
        const float3 hit_j = add3f(refl_orig, scale3f(t_j, refl));

        accumulate_srp(s_eq_j, cos_j, s2 * intensity,
                       params.d_reflectance[j], params.d_specularity[j],
                       j_n, hit_j);

        intensity *= params.d_reflectance[j] * params.d_specularity[j];
        cur_n     = j_n;
        cur_orig  = hit_j;
        s_eq      = s_eq_j;
    }
}
