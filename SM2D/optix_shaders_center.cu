// optix_shaders_center.cu
// ============================================================
// OptiX device programs for RTX Shadow method (method 7).
// Force/moment accumulators are float* - safe because the
// number of rays equals the triangle count, which is far below
// the ~8.4M float-atomicAdd saturation limit.
//
// COMPILE TO PTX:
//   nvcc -ptx -arch=compute_75 ^
//        -I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0\include" ^
//        optix_shaders_center.cu -o optix_shaders_center.ptx
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

// ---- params layout -- MUST match RTXParamsCenter in gpu_optix_center_raytracer.cu ----
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

extern "C" __constant__ RTXParamsCenter params;

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
    // float atomicAdd: safe for N_triangles << 8.4M
    atomicAdd(&params.d_force[0], dfx);                                                   //PHY
    atomicAdd(&params.d_force[1], dfy);                                                   //PHY
    atomicAdd(&params.d_force[2], dfz);                                                   //PHY
    atomicAdd(&params.d_moment[0], c.y*dfz - c.z*dfy);                                   //PHY
    atomicAdd(&params.d_moment[1], c.z*dfx - c.x*dfz);                                   //PHY
    atomicAdd(&params.d_moment[2], c.x*dfy - c.y*dfx);                                   //PHY
}

// ============================================================
// Pipeline A -- RTX Shadow (center method, method 7)
// Launch: (N, 1, 1) -- one thread per triangle
//
// Payload layout (2 registers):
//   p0: primitive index of hit, or 0xFFFFFFFF if missed
//   p1: hit t value (bits reinterpreted as float), valid when p0 != 0xFFFFFFFF
// ============================================================

extern "C" __global__ void __miss__center() {
    optixSetPayload_0(0xFFFFFFFFu);
    optixSetPayload_1(0u);
}

extern "C" __global__ void __closesthit__center() {
    optixSetPayload_0((unsigned int)optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

extern "C" __global__ void __raygen__center() {
    const int tri = (int)optixGetLaunchIndex().x;
    if (tri >= params.N) return;

    const float3 n_tri  = params.d_normals[tri];
    const float  cos_th = dot3f(n_tri, params.sun);
    if (cos_th <= 0.0f) return;  // back face

    // Shadow ray: from centroid toward sun.
    // tmin matches the origin offset so that occluders immediately adjacent
    // to the surface are not skipped.  Using 1e-3f here was 10× larger than
    // the 1e-4f origin offset and caused nearby panels to be missed as
    // occluders, making triangles appear directly lit when they are actually
    // shadowed (wrong yellow instead of correct 2nd-reflection red).
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

    // Lit: record label and bounce info
    atomicOr(&params.d_labels[tri], 1);
    atomicMin(&params.d_bounce_levels[tri], 0);

    // First-hit-wins: write sun direction as incident dir
    int prev = atomicCAS(reinterpret_cast<int*>(&params.d_incident_dirs[tri * 3]),
                         -1, __float_as_int(params.sun.x));
    if (prev == -1) {
        params.d_incident_dirs[tri * 3 + 1] = params.sun.y;
        params.d_incident_dirs[tri * 3 + 2] = params.sun.z;
        // d_origin_pts stays NaN - directly illuminated
    }

    // Direct SRP contribution.
    // area_w = A * cos_theta (projected area) - matches compute_srp_forces on CPU:
    //   base = phi0 * A * cos_theta; s_coeff/n_coeff both scale from base.
    accumulate_srp(params.sun, cos_th,
                   params.d_areas[tri] * cos_th,
                   params.d_reflectance[tri], params.d_specularity[tri],
                   n_tri, params.d_centroids[tri]);

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

        // Actual ray-triangle intersection point on j (not j's centroid).
        // Used as the moment arm for reflected-light force, mirroring the
        // pixel-grid method.  This is the load-bearing fix for reflection-
        // induced moment errors in the centroid pipeline.
        const float  t_j   = __uint_as_float(j_t_bits);
        const float3 hit_j = add3f(refl_orig, scale3f(t_j, refl));

        // ---- Visualization tracking (before cos_j gate) ----------------------
        // Record bounce level for EVERY hit, including back-facing surfaces.
        // This ensures panels that are struck from behind (cos_j <= 0) still
        // appear orange in the visualizer.  Force computation is gated below.
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

        // Physical check: back-facing or grazing surfaces receive no force
        // and cannot continue the specular reflection chain.
        if (cos_j <= 1e-6f) break;

        // area_w = A_tri * cos_th * intensity (emitter's projected area, conserved
        // through specular bounces - beam cross-section is invariant).
        // Using the RECEIVER's A_j * cos_j was wrong: it changes the magnitude of
        // the force with each bounce even for a perfect mirror (alpha=mu=1).
        // Accumulated into d_refl_force/d_refl_moment, kept separate from direct
        // illumination so the host can combine CPU-double direct forces with
        // GPU-float reflection forces for maximum accuracy.
        {
            const float3 ldir  = s_eq_j;
            const float  aw    = params.d_areas[tri] * cos_th * intensity;
            const float  alpha = params.d_reflectance[j];
            const float  mu    = params.d_specularity[j];
            const float  sc    = -aw * ((1.0f - alpha) + alpha*(1.0f - mu));               //PHY
            const float  nc    = -aw * (2.0f*alpha*mu*cos_j + (2.0f/3.0f)*alpha*(1.0f - mu)); //PHY
            const float  dfx   = sc*ldir.x + nc*j_n.x;                                    //PHY
            const float  dfy   = sc*ldir.y + nc*j_n.y;                                    //PHY
            const float  dfz   = sc*ldir.z + nc*j_n.z;                                    //PHY
            const float3 jc    = hit_j;
            atomicAdd(&params.d_refl_force[0],  dfx);                                      //PHY
            atomicAdd(&params.d_refl_force[1],  dfy);                                      //PHY
            atomicAdd(&params.d_refl_force[2],  dfz);                                      //PHY
            atomicAdd(&params.d_refl_moment[0], jc.y*dfz - jc.z*dfy);                     //PHY
            atomicAdd(&params.d_refl_moment[1], jc.z*dfx - jc.x*dfz);                     //PHY
            atomicAdd(&params.d_refl_moment[2], jc.x*dfy - jc.y*dfx);                     //PHY
        }

        intensity *= params.d_reflectance[j] * params.d_specularity[j];
        cur_n     = j_n;
        cur_orig  = hit_j;
        s_eq      = s_eq_j;
    }
}
