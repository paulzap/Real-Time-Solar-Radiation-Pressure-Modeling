// ---------------------------------------------------------------------------
// gpu_pixel_grid_raytracer_bench.cu
// Benchmark-only lean copy of gpu_pixel_grid_raytracer.cu.
//
// Removed vs original:
//   - bounce_levels / incident_dirs / origin_pts vectors (CPU-side, N×24 bytes)
//   - set_bounce_globals() call
//   - Verbose/debug prints from bounce pass
//   - primary_emitter_name parameter (not used in benchmarks)
//
// GPU kernel and BVH infrastructure are identical to the original; only the
// kernel is renamed to pixelGridKernelBench to avoid a linker conflict.
//
// Exports: calculate_labels_pixel_grid_gpu_bench()
// ---------------------------------------------------------------------------

#include "ShadowAlgorithms.h"
#include "bench_methods.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <functional>

#define PGB_CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            char buf[512]; \
            snprintf(buf, sizeof(buf), "CUDA error %s:%d - %s", \
                     __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(buf); \
        } \
    } while(0)

// ---------------------------------------------------------------------------
// Device-side structs (identical to original, TU-local)
// ---------------------------------------------------------------------------
struct PGBTriDev {
    double v[3][3];
    double normal[3];
    double center[3];
    float reflectance;
    float specularity;
};

struct PGBBVHNodeDev {
    double  bounds[6];
    unsigned int left;
    unsigned int right;
    unsigned int triangle_idx;
    int    is_leaf;
};

// ---------------------------------------------------------------------------
// Custom atomicAdd for double - required for compute_52 (Maxwell) targets
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// ---------------------------------------------------------------------------
// Device BVH traversal helpers (static = TU-local, no name conflict)
// ---------------------------------------------------------------------------
__device__ static inline bool pgb_ray_aabb_dev(
    const double* __restrict__ o, const double* __restrict__ d,
    const double* __restrict__ b, double& tmin_out, double& tmax_out)
{
    double tmin = 0.0, tmax = 1e30;
    for (int ax = 0; ax < 3; ++ax) {
        double inv = 1.0 / d[ax];
        double lo = (b[ax] - o[ax]) * inv;
        double hi = (b[ax+3] - o[ax]) * inv;
        if (lo > hi) { double tmp = lo; lo = hi; hi = tmp; }
        tmin = fmax(tmin, lo);
        tmax = fmin(tmax, hi);
        if (tmin > tmax) return false;
    }
    tmin_out = tmin; tmax_out = tmax;
    return tmax > 1e-8;
}

__device__ static bool pgb_ray_tri_dev(
    const double* __restrict__ o, const double* __restrict__ d,
    const double* v0, const double* v1, const double* v2,
    double& t_out)
{
    const double EPS = 1e-8;
    double e1[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
    double e2[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
    double h[3] = { d[1]*e2[2]-d[2]*e2[1], d[2]*e2[0]-d[0]*e2[2], d[0]*e2[1]-d[1]*e2[0] };
    double a = e1[0]*h[0] + e1[1]*h[1] + e1[2]*h[2];
    if (fabs(a) < EPS) return false;
    double f = 1.0 / a;
    double s[3] = { o[0]-v0[0], o[1]-v0[1], o[2]-v0[2] };
    double u = f * (s[0]*h[0] + s[1]*h[1] + s[2]*h[2]);
    if (u < 0.0 || u > 1.0) return false;
    double q[3] = { s[1]*e1[2]-s[2]*e1[1], s[2]*e1[0]-s[0]*e1[2], s[0]*e1[1]-s[1]*e1[0] };
    double v = f * (d[0]*q[0] + d[1]*q[1] + d[2]*q[2]);
    if (v < 0.0 || u + v > 1.0) return false;
    double t = f * (e2[0]*q[0] + e2[1]*q[1] + e2[2]*q[2]);
    if (t < EPS) return false;
    t_out = t;
    return true;
}

// ---------------------------------------------------------------------------
// Kernel (renamed to avoid linker conflict with pixelGridKernel)
// ---------------------------------------------------------------------------
__global__ void pixelGridKernelBench(
    const PGBTriDev* __restrict__ triangles,
    const PGBBVHNodeDev* __restrict__ nodes,
    unsigned int root_idx,
    double ux, double uy, double uz,
    double vx, double vy, double vz,
    double sx, double sy, double sz,
    double u_min, double v_min,
    double grid_step,
    int nu, int nv,
    double t_start,
    double step2,
    double* __restrict__ d_force,
    double* __restrict__ d_moment,
    // per-pixel bounce data (nullptr if max_reflections==0)
    int*    d_hit_tri_out,
    double* d_hit_pt_out,
    double* d_hit_intensity_out,
    double* d_hit_dir_out)
{
    int iu = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int iv = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (iu >= nu || iv >= nv) return;

    double u_c = u_min + (iu + 0.5) * grid_step;
    double v_c = v_min + (iv + 0.5) * grid_step;

    double ray_o[3] = {
        u_c*ux + v_c*vx + t_start*sx,
        u_c*uy + v_c*vy + t_start*sy,
        u_c*uz + v_c*vz + t_start*sz
    };
    double ray_d[3] = { -sx, -sy, -sz };

    unsigned int stack[128];
    int sp = 0;
    stack[sp++] = root_idx;

    int    closest   = -1;
    double closest_t = 1e30;

    while (sp > 0) {
        unsigned int node_idx = stack[--sp];
        const PGBBVHNodeDev& node = nodes[node_idx];

        double tmin, tmax;
        if (!pgb_ray_aabb_dev(ray_o, ray_d, node.bounds, tmin, tmax)) continue;
        if (tmin > closest_t) continue;

        if (node.is_leaf) {
            unsigned int j = node.triangle_idx;
            double t;
            if (pgb_ray_tri_dev(ray_o, ray_d,
                    triangles[j].v[0], triangles[j].v[1], triangles[j].v[2], t)) {
                if (t < closest_t) { closest_t = t; closest = (int)j; }
            }
        }
        else {
            if (sp < 126) { stack[sp++] = node.left; stack[sp++] = node.right; }
        }
    }

    if (closest < 0) return;

    double cos_theta = triangles[closest].normal[0]*sx
                     + triangles[closest].normal[1]*sy
                     + triangles[closest].normal[2]*sz;
    if (cos_theta <= 1e-6) return;

    double alpha_t = (double)triangles[closest].reflectance;
    double mu_t    = (double)triangles[closest].specularity;

    double sc_c = -step2 * ((1.0 - alpha_t) + alpha_t*(1.0 - mu_t));
    double n_a  =  step2 * 2.0 * alpha_t * mu_t;
    double n_b  =  step2 * (2.0/3.0) * alpha_t * (1.0 - mu_t);
    double nc_c = -(n_a * cos_theta + n_b);
    double dfx  = sc_c*sx + nc_c*triangles[closest].normal[0];
    double dfy  = sc_c*sy + nc_c*triangles[closest].normal[1];
    double dfz  = sc_c*sz + nc_c*triangles[closest].normal[2];

    atomicAdd(&d_force[0], dfx);
    atomicAdd(&d_force[1], dfy);
    atomicAdd(&d_force[2], dfz);

    double hit_x = ray_o[0] + closest_t*ray_d[0];
    double hit_y = ray_o[1] + closest_t*ray_d[1];
    double hit_z = ray_o[2] + closest_t*ray_d[2];
    atomicAdd(&d_moment[0], hit_y*dfz - hit_z*dfy);
    atomicAdd(&d_moment[1], hit_z*dfx - hit_x*dfz);
    atomicAdd(&d_moment[2], hit_x*dfy - hit_y*dfx);

    // Per-pixel hit data for bounce pass
    if (d_hit_tri_out) {
        int idx = iv * nu + iu;
        d_hit_tri_out[idx]       = closest;
        d_hit_pt_out[idx*3+0]    = hit_x;
        d_hit_pt_out[idx*3+1]    = hit_y;
        d_hit_pt_out[idx*3+2]    = hit_z;
        double alpha_t = (double)triangles[closest].reflectance;
        double mu_t    = (double)triangles[closest].specularity;
        d_hit_intensity_out[idx] = step2 * alpha_t * mu_t;
        d_hit_dir_out[idx*3+0]   = sx;
        d_hit_dir_out[idx*3+1]   = sy;
        d_hit_dir_out[idx*3+2]   = sz;
    }
}

// ---------------------------------------------------------------------------
// Bounce kernel (bench variant - no bounce tracking, uses PGB types)
// ---------------------------------------------------------------------------
__global__ void pixelGridBounceKernelBench(
    const PGBTriDev* __restrict__ triangles,
    const PGBBVHNodeDev* __restrict__ nodes,
    unsigned int root_idx,
    int nu, int nv,
    const int*    __restrict__ d_in_tri,
    const double* __restrict__ d_in_pt,
    const double* __restrict__ d_in_intensity,
    const double* __restrict__ d_in_dir,
    int*    d_out_tri,
    double* d_out_pt,
    double* d_out_intensity,
    double* d_out_dir,
    double* d_force,
    double* d_moment)
{
    int iu = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int iv = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (iu >= nu || iv >= nv) return;

    int idx = iv * nu + iu;
    d_out_tri[idx] = -1;

    int src_tri = d_in_tri[idx];
    if (src_tri < 0) return;

    double intensity = d_in_intensity[idx];
    if (intensity < 1e-30) return;

    double seq_x = d_in_dir[idx*3+0];
    double seq_y = d_in_dir[idx*3+1];
    double seq_z = d_in_dir[idx*3+2];

    double nx = triangles[src_tri].normal[0];
    double ny = triangles[src_tri].normal[1];
    double nz = triangles[src_tri].normal[2];

    double cos_src = nx*seq_x + ny*seq_y + nz*seq_z;
    if (cos_src <= 1e-6) return;

    double rx = 2.0*cos_src*nx - seq_x;
    double ry = 2.0*cos_src*ny - seq_y;
    double rz = 2.0*cos_src*nz - seq_z;
    double rlen = sqrt(rx*rx + ry*ry + rz*rz);
    if (rlen < 1e-12) return;
    rx /= rlen; ry /= rlen; rz /= rlen;

    double ox = d_in_pt[idx*3+0] + 1e-4*nx;
    double oy = d_in_pt[idx*3+1] + 1e-4*ny;
    double oz = d_in_pt[idx*3+2] + 1e-4*nz;

    double ray_o[3] = { ox, oy, oz };
    double ray_d[3] = { rx, ry, rz };

    unsigned int stack[128];
    int sp = 0;
    stack[sp++] = root_idx;

    int    hit_tri = -1;
    double hit_t   = 1e30;

    while (sp > 0) {
        unsigned int node_idx = stack[--sp];
        const PGBBVHNodeDev& node = nodes[node_idx];

        double tmin, tmax;
        if (!pgb_ray_aabb_dev(ray_o, ray_d, node.bounds, tmin, tmax)) continue;
        if (tmin > hit_t) continue;

        if (node.is_leaf) {
            unsigned int j = node.triangle_idx;
            if ((int)j == src_tri) continue;
            double t;
            if (pgb_ray_tri_dev(ray_o, ray_d,
                    triangles[j].v[0], triangles[j].v[1], triangles[j].v[2], t)) {
                if (t < hit_t) { hit_t = t; hit_tri = (int)j; }
            }
        } else {
            if (sp < 126) { stack[sp++] = node.left; stack[sp++] = node.right; }
        }
    }

    if (hit_tri < 0) return;

    double n2x = triangles[hit_tri].normal[0];
    double n2y = triangles[hit_tri].normal[1];
    double n2z = triangles[hit_tri].normal[2];
    double s2x = -rx, s2y = -ry, s2z = -rz;
    double cos2 = n2x*s2x + n2y*s2y + n2z*s2z;
    if (cos2 <= 1e-6) return;

    double hit_x = ox + hit_t * rx;
    double hit_y = oy + hit_t * ry;
    double hit_z = oz + hit_t * rz;

    double alpha2 = (double)triangles[hit_tri].reflectance;
    double mu2    = (double)triangles[hit_tri].specularity;

    double sc2 = -intensity * ((1.0 - alpha2) + alpha2*(1.0 - mu2));
    double nc2 = -intensity * (2.0*alpha2*mu2*cos2 + (2.0/3.0)*alpha2*(1.0 - mu2));
    double dfx = sc2*s2x + nc2*n2x;
    double dfy = sc2*s2y + nc2*n2y;
    double dfz = sc2*s2z + nc2*n2z;

    atomicAdd(&d_force[0], dfx);
    atomicAdd(&d_force[1], dfy);
    atomicAdd(&d_force[2], dfz);
    atomicAdd(&d_moment[0], hit_y*dfz - hit_z*dfy);
    atomicAdd(&d_moment[1], hit_z*dfx - hit_x*dfz);
    atomicAdd(&d_moment[2], hit_x*dfy - hit_y*dfx);

    d_out_tri[idx]         = hit_tri;
    d_out_pt[idx*3+0]      = hit_x;
    d_out_pt[idx*3+1]      = hit_y;
    d_out_pt[idx*3+2]      = hit_z;
    d_out_intensity[idx]   = intensity * alpha2 * mu2;
    d_out_dir[idx*3+0]     = s2x;
    d_out_dir[idx*3+1]     = s2y;
    d_out_dir[idx*3+2]     = s2z;
}

// ---------------------------------------------------------------------------
// Host-side BVH helpers - used only during GPU BVH construction, not for traversal.
// ---------------------------------------------------------------------------
struct PGBHostAABB { double mn[3], mx[3]; };
struct PGBHostNode {
    PGBHostAABB b;
    size_t ti, l, r;
    bool   leaf;
};

// ---------------------------------------------------------------------------
// GPU cache (TU-local static, no conflict with g_pg_cache)
// ---------------------------------------------------------------------------
struct PGGPUCacheBench {
    PGBTriDev*     d_tri    = nullptr;
    PGBBVHNodeDev* d_nodes  = nullptr;
    double*        d_force  = nullptr;
    double*        d_moment = nullptr;
    unsigned int   root_idx = 0;
    size_t         N        = 0;
    size_t         numNodes = 0;
    bool           valid    = false;
    cudaStream_t   stream   = nullptr;
    // Per-pixel bounce ping-pong buffers (size = nu*nv each)
    int*    d_bounce_tri_A  = nullptr;
    int*    d_bounce_tri_B  = nullptr;
    double* d_bounce_pt_A   = nullptr;
    double* d_bounce_pt_B   = nullptr;
    double* d_bounce_int_A  = nullptr;
    double* d_bounce_int_B  = nullptr;
    double* d_bounce_dir_A  = nullptr;
    double* d_bounce_dir_B  = nullptr;
    size_t  bounce_px_alloc = 0;
};

static PGGPUCacheBench g_pg_bench_cache;

// ---------------------------------------------------------------------------
// calculate_labels_pixel_grid_gpu_bench
// ---------------------------------------------------------------------------
SRPResult calculate_labels_pixel_grid_gpu_bench(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections,
    bool verbose)
{
    if (triangles.empty()) return {};
    if (grid_step <= 0.0)
        throw std::runtime_error("grid_step must be positive");

    const size_t N = triangles.size();

    double slen = std::sqrt(sun_vector[0]*sun_vector[0] +
                            sun_vector[1]*sun_vector[1] +
                            sun_vector[2]*sun_vector[2]);
    if (slen < 1e-12) throw std::runtime_error("Sun vector is zero");
    double sx = sun_vector[0]/slen, sy = sun_vector[1]/slen, sz = sun_vector[2]/slen;

    // Orthonormal basis
    double hx, hy, hz;
    if (fabs(sx) < 0.9) { hx=1; hy=0; hz=0; } else { hx=0; hy=1; hz=0; }
    double ux = sy*hz-sz*hy, uy = sz*hx-sx*hz, uz = sx*hy-sy*hx;
    double ulen = sqrt(ux*ux + uy*uy + uz*uz);
    ux/=ulen; uy/=ulen; uz/=ulen;
    double vx = sy*uz-sz*uy, vy = sz*ux-sx*uz, vz = sx*uy-sy*ux;

    // Grid extents
    double u_min=1e30, u_max=-1e30, v_min=1e30, v_max=-1e30, s_max=-1e30;
    for (const auto& tri : triangles) {
        for (const auto& pt : std::array<std::array<double,3>,3>{{
            {tri.v1_x,tri.v1_y,tri.v1_z},{tri.v2_x,tri.v2_y,tri.v2_z},{tri.v3_x,tri.v3_y,tri.v3_z}}}) {
            double uc = pt[0]*ux+pt[1]*uy+pt[2]*uz;
            double vc = pt[0]*vx+pt[1]*vy+pt[2]*vz;
            double sc = pt[0]*sx+pt[1]*sy+pt[2]*sz;
            u_min=std::min(u_min,uc); u_max=std::max(u_max,uc);
            v_min=std::min(v_min,vc); v_max=std::max(v_max,vc);
            s_max=std::max(s_max,sc);
        }
    }

    double gs = grid_step;
    const double u_center = 0.5*(u_min+u_max), v_center = 0.5*(v_min+v_max);
    const double u_half   = 0.5*(u_max-u_min)+gs, v_half = 0.5*(v_max-v_min)+gs;
    double t_start = s_max + gs;

    long long nu_ll = (long long)std::ceil(2.0*u_half/gs);
    long long nv_ll = (long long)std::ceil(2.0*v_half/gs);
    u_min = u_center - (nu_ll*gs)*0.5;
    v_min = v_center - (nv_ll*gs)*0.5;

    if (nu_ll > 2'000'000'000LL || nv_ll > 2'000'000'000LL)
        throw std::runtime_error("Pixel grid step too small: grid dimensions exceed CUDA limits.");
    int nu = (int)nu_ll, nv = (int)nv_ll;

    // Rebuild cache if triangle set changed
    if (!g_pg_bench_cache.valid || g_pg_bench_cache.N != N) {
        if (g_pg_bench_cache.d_tri)           PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_tri));
        if (g_pg_bench_cache.d_nodes)         PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_nodes));
        if (g_pg_bench_cache.d_force)         PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_force));
        if (g_pg_bench_cache.d_moment)        PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_moment));
        // Free bounce ping-pong buffers
        if (g_pg_bench_cache.d_bounce_tri_A)  PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_tri_A));
        if (g_pg_bench_cache.d_bounce_tri_B)  PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_tri_B));
        if (g_pg_bench_cache.d_bounce_pt_A)   PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_pt_A));
        if (g_pg_bench_cache.d_bounce_pt_B)   PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_pt_B));
        if (g_pg_bench_cache.d_bounce_int_A)  PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_int_A));
        if (g_pg_bench_cache.d_bounce_int_B)  PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_int_B));
        if (g_pg_bench_cache.d_bounce_dir_A)  PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_dir_A));
        if (g_pg_bench_cache.d_bounce_dir_B)  PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_dir_B));
        if (g_pg_bench_cache.stream)          PGB_CUDA_CHECK(cudaStreamDestroy(g_pg_bench_cache.stream));
        g_pg_bench_cache = PGGPUCacheBench{};

        PGB_CUDA_CHECK(cudaStreamCreate(&g_pg_bench_cache.stream));

        // Build host BVH
        std::vector<PGBHostAABB> tb(N);
        for (size_t i = 0; i < N; ++i) {
            tb[i] = { {std::min({triangles[i].v1_x,triangles[i].v2_x,triangles[i].v3_x}),
                       std::min({triangles[i].v1_y,triangles[i].v2_y,triangles[i].v3_y}),
                       std::min({triangles[i].v1_z,triangles[i].v2_z,triangles[i].v3_z})},
                      {std::max({triangles[i].v1_x,triangles[i].v2_x,triangles[i].v3_x}),
                       std::max({triangles[i].v1_y,triangles[i].v2_y,triangles[i].v3_y}),
                       std::max({triangles[i].v1_z,triangles[i].v2_z,triangles[i].v3_z})} };
        }
        std::vector<PGBHostNode> hn;
        hn.reserve(4*N);
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);

        std::function<size_t(size_t,size_t)> bld = [&](size_t lo, size_t hi) -> size_t {
            PGBHostNode nd{};
            if (hi-lo == 1) {
                nd.leaf=true; nd.ti=idx[lo]; nd.b=tb[idx[lo]];
                hn.push_back(nd); return hn.size()-1;
            }
            PGBHostAABB merged = tb[idx[lo]];
            for (size_t k=lo+1; k<hi; ++k)
                for (int ax=0;ax<3;++ax) {
                    merged.mn[ax]=std::min(merged.mn[ax],tb[idx[k]].mn[ax]);
                    merged.mx[ax]=std::max(merged.mx[ax],tb[idx[k]].mx[ax]);
                }
            int axis=0;
            double dx=merged.mx[0]-merged.mn[0], dy=merged.mx[1]-merged.mn[1], dz=merged.mx[2]-merged.mn[2];
            if (dy>dx && dy>dz) axis=1; else if (dz>dx) axis=2;
            std::sort(idx.begin()+lo, idx.begin()+hi, [&](size_t a,size_t b_) {
                return tb[a].mn[axis]+tb[a].mx[axis] < tb[b_].mn[axis]+tb[b_].mx[axis]; });
            size_t mid=(lo+hi)/2;
            nd.leaf=false; nd.b=merged;
            hn.push_back(nd);
            size_t self=hn.size()-1;
            hn[self].l=bld(lo,mid); hn[self].r=bld(mid,hi);
            return self;
        };
        size_t root = bld(0, N);
        g_pg_bench_cache.root_idx = (unsigned int)root;
        g_pg_bench_cache.numNodes = hn.size();

        // Pack GPU structs
        std::vector<PGBTriDev> triDev(N);
        for (size_t i=0; i<N; ++i) {
            triDev[i].v[0][0]=triangles[i].v1_x; triDev[i].v[0][1]=triangles[i].v1_y; triDev[i].v[0][2]=triangles[i].v1_z;
            triDev[i].v[1][0]=triangles[i].v2_x; triDev[i].v[1][1]=triangles[i].v2_y; triDev[i].v[1][2]=triangles[i].v2_z;
            triDev[i].v[2][0]=triangles[i].v3_x; triDev[i].v[2][1]=triangles[i].v3_y; triDev[i].v[2][2]=triangles[i].v3_z;
            triDev[i].normal[0]=triangles[i].normal_x; triDev[i].normal[1]=triangles[i].normal_y; triDev[i].normal[2]=triangles[i].normal_z;
            triDev[i].center[0]=(triangles[i].v1_x+triangles[i].v2_x+triangles[i].v3_x)/3.0;
            triDev[i].center[1]=(triangles[i].v1_y+triangles[i].v2_y+triangles[i].v3_y)/3.0;
            triDev[i].center[2]=(triangles[i].v1_z+triangles[i].v2_z+triangles[i].v3_z)/3.0;
            triDev[i].reflectance=(float)triangles[i].reflectance;
            triDev[i].specularity=(float)triangles[i].specularity;
        }

        std::vector<PGBBVHNodeDev> nodeDev(hn.size());
        for (size_t i=0; i<hn.size(); ++i) {
            const auto& h=hn[i];
            nodeDev[i].bounds[0]=h.b.mn[0]; nodeDev[i].bounds[1]=h.b.mn[1]; nodeDev[i].bounds[2]=h.b.mn[2];
            nodeDev[i].bounds[3]=h.b.mx[0]; nodeDev[i].bounds[4]=h.b.mx[1]; nodeDev[i].bounds[5]=h.b.mx[2];
            nodeDev[i].is_leaf=(int)h.leaf;
            nodeDev[i].triangle_idx=h.leaf?(unsigned int)h.ti:0u;
            nodeDev[i].left =h.leaf?0u:(unsigned int)h.l;
            nodeDev[i].right=h.leaf?0u:(unsigned int)h.r;
        }

        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_tri,    N*sizeof(PGBTriDev)));
        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_nodes,  hn.size()*sizeof(PGBBVHNodeDev)));
        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_force,  3*sizeof(double)));
        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_moment, 3*sizeof(double)));

        PGB_CUDA_CHECK(cudaMemcpy(g_pg_bench_cache.d_tri,   triDev.data(),   N*sizeof(PGBTriDev),             cudaMemcpyHostToDevice));
        PGB_CUDA_CHECK(cudaMemcpy(g_pg_bench_cache.d_nodes, nodeDev.data(),  hn.size()*sizeof(PGBBVHNodeDev), cudaMemcpyHostToDevice));

        g_pg_bench_cache.N = N;
        g_pg_bench_cache.valid = true;
    }

    // Reset per-call arrays
    PGB_CUDA_CHECK(cudaMemsetAsync(g_pg_bench_cache.d_force,  0, 3*sizeof(double),    g_pg_bench_cache.stream));
    PGB_CUDA_CHECK(cudaMemsetAsync(g_pg_bench_cache.d_moment, 0, 3*sizeof(double),    g_pg_bench_cache.stream));

    double step2 = gs*gs;
    size_t px_count = (size_t)nu * (size_t)nv;

    // Reallocate ping-pong bounce buffers if grid size grew
    if (max_reflections > 0 && px_count > g_pg_bench_cache.bounce_px_alloc) {
        if (g_pg_bench_cache.d_bounce_tri_A) PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_tri_A));
        if (g_pg_bench_cache.d_bounce_tri_B) PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_tri_B));
        if (g_pg_bench_cache.d_bounce_pt_A)  PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_pt_A));
        if (g_pg_bench_cache.d_bounce_pt_B)  PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_pt_B));
        if (g_pg_bench_cache.d_bounce_int_A) PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_int_A));
        if (g_pg_bench_cache.d_bounce_int_B) PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_int_B));
        if (g_pg_bench_cache.d_bounce_dir_A) PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_dir_A));
        if (g_pg_bench_cache.d_bounce_dir_B) PGB_CUDA_CHECK(cudaFree(g_pg_bench_cache.d_bounce_dir_B));
        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_bounce_tri_A, px_count * sizeof(int)));
        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_bounce_tri_B, px_count * sizeof(int)));
        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_bounce_pt_A,  px_count * 3 * sizeof(double)));
        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_bounce_pt_B,  px_count * 3 * sizeof(double)));
        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_bounce_int_A, px_count * sizeof(double)));
        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_bounce_int_B, px_count * sizeof(double)));
        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_bounce_dir_A, px_count * 3 * sizeof(double)));
        PGB_CUDA_CHECK(cudaMalloc(&g_pg_bench_cache.d_bounce_dir_B, px_count * 3 * sizeof(double)));
        g_pg_bench_cache.bounce_px_alloc = px_count;
    }

    const int TW=16, TH=16;
    dim3 blockDim2d(TW,TH);
    dim3 gridDim2d((nu+TW-1)/TW, (nv+TH-1)/TH);

    // Init d_bounce_tri_A to -1 before direct kernel
    if (max_reflections > 0)
        PGB_CUDA_CHECK(cudaMemsetAsync(g_pg_bench_cache.d_bounce_tri_A, 0xFF,
            px_count * sizeof(int), g_pg_bench_cache.stream));

    pixelGridKernelBench<<<gridDim2d, blockDim2d, 0, g_pg_bench_cache.stream>>>(
        g_pg_bench_cache.d_tri,
        g_pg_bench_cache.d_nodes,
        g_pg_bench_cache.root_idx,
        ux,uy,uz, vx,vy,vz, sx,sy,sz,
        u_min, v_min, gs, nu, nv, t_start, step2,
        g_pg_bench_cache.d_force,
        g_pg_bench_cache.d_moment,
        max_reflections > 0 ? g_pg_bench_cache.d_bounce_tri_A : nullptr,
        max_reflections > 0 ? g_pg_bench_cache.d_bounce_pt_A  : nullptr,
        max_reflections > 0 ? g_pg_bench_cache.d_bounce_int_A : nullptr,
        max_reflections > 0 ? g_pg_bench_cache.d_bounce_dir_A : nullptr);
    PGB_CUDA_CHECK(cudaGetLastError());

    // GPU bounce loop - per-pixel specular reflections (area_w = s² each bounce)
    int*    cur_tri = g_pg_bench_cache.d_bounce_tri_A;
    int*    nxt_tri = g_pg_bench_cache.d_bounce_tri_B;
    double* cur_pt  = g_pg_bench_cache.d_bounce_pt_A;
    double* nxt_pt  = g_pg_bench_cache.d_bounce_pt_B;
    double* cur_int = g_pg_bench_cache.d_bounce_int_A;
    double* nxt_int = g_pg_bench_cache.d_bounce_int_B;
    double* cur_dir = g_pg_bench_cache.d_bounce_dir_A;
    double* nxt_dir = g_pg_bench_cache.d_bounce_dir_B;

    for (int b = 0; b < max_reflections; ++b) {
        PGB_CUDA_CHECK(cudaMemsetAsync(nxt_tri, 0xFF, px_count * sizeof(int), g_pg_bench_cache.stream));

        pixelGridBounceKernelBench<<<gridDim2d, blockDim2d, 0, g_pg_bench_cache.stream>>>(
            g_pg_bench_cache.d_tri,
            g_pg_bench_cache.d_nodes,
            g_pg_bench_cache.root_idx,
            nu, nv,
            cur_tri, cur_pt, cur_int, cur_dir,
            nxt_tri, nxt_pt, nxt_int, nxt_dir,
            g_pg_bench_cache.d_force,
            g_pg_bench_cache.d_moment);
        PGB_CUDA_CHECK(cudaGetLastError());

        std::swap(cur_tri, nxt_tri);
        std::swap(cur_pt,  nxt_pt);
        std::swap(cur_int, nxt_int);
        std::swap(cur_dir, nxt_dir);
    }

    double h_force[3]={}, h_moment[3]={};
    PGB_CUDA_CHECK(cudaMemcpyAsync(h_force,  g_pg_bench_cache.d_force,  3*sizeof(double),
        cudaMemcpyDeviceToHost, g_pg_bench_cache.stream));
    PGB_CUDA_CHECK(cudaMemcpyAsync(h_moment, g_pg_bench_cache.d_moment, 3*sizeof(double),
        cudaMemcpyDeviceToHost, g_pg_bench_cache.stream));
    PGB_CUDA_CHECK(cudaStreamSynchronize(g_pg_bench_cache.stream));

    std::array<double,3> total_force  = { g_srp_phi0*h_force[0],  g_srp_phi0*h_force[1],  g_srp_phi0*h_force[2]  };
    std::array<double,3> total_moment = { g_srp_phi0*h_moment[0], g_srp_phi0*h_moment[1], g_srp_phi0*h_moment[2] };

    // NO set_bounce_globals call (bench variant: no visualization tracking)
    SRPResult result;
    result.labels       = {};   // bench: labels not tracked (no h_pinned_labels)
    result.total_force  = total_force;
    result.total_moment = total_moment;
    return result;
}
