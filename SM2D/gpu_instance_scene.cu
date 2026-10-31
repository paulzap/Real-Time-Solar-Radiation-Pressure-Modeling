// SolaRT - solar radiation pressure on GPU RT cores.
// Copyright (c) 2026 Zapevalin P.R. SPDX-License-Identifier: MIT
// If you use this code in published work, please cite:
// https://doi.org/10.1016/j.ast.2026.113714

// ---------------------------------------------------------------------------
// gpu_instance_scene.cu - shared device copy of the body-frame centroids and
// normals for the two-level acceleration structures (see gpu_instance_scene.h).
// ---------------------------------------------------------------------------
#include "gpu_instance_scene.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#define SCENE_CUDA_CHECK(call)                                                         \
    do {                                                                               \
        cudaError_t err_ = (call);                                                     \
        if (err_ != cudaSuccess) {                                                     \
            (void)cudaGetLastError();   /* or the next good call would read it */      \
            throw std::runtime_error(std::string("CUDA error in gpu_instance_scene: ") + \
                                     cudaGetErrorString(err_));                        \
        }                                                                              \
    } while (0)

// Body-frame triangle of a movable part: SatelliteDataset::rebuildTriangles /
// scene_body_triangle with every product and sum rounded separately (no
// fused multiply-add), so the doubles are bit-identical to the host's.
__device__ static inline double sc_xform(const double* T, const double* p)
{
    return __dadd_rn(__dadd_rn(__dadd_rn(__dmul_rn(T[0], p[0]), __dmul_rn(T[1], p[1])),
                               __dmul_rn(T[2], p[2])), T[3]);
}
__device__ static inline double sc_rot(const double* T, const double* p)
{
    return __dadd_rn(__dadd_rn(__dmul_rn(T[0], p[0]), __dmul_rn(T[1], p[1])), __dmul_rn(T[2], p[2]));
}

__global__ void sceneWorldKernel(const double* __restrict__ node_T,      // [nodes * 12]
                                 const double* __restrict__ node_v,      // [M * 9]
                                 const double* __restrict__ node_n,      // [M * 3]
                                 const unsigned int* __restrict__ tri_node,
                                 const unsigned int* __restrict__ tri_global,
                                 unsigned int M,
                                 double* __restrict__ world_c, double* __restrict__ world_n)
{
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= M) return;
    const double* T = node_T + 12 * (size_t)tri_node[t];
    const double* v = node_v + 9 * (size_t)t;
    double w[9];
    for (int k = 0; k < 3; ++k) {
        w[3 * k + 0] = sc_xform(T + 0, v + 3 * k);
        w[3 * k + 1] = sc_xform(T + 4, v + 3 * k);
        w[3 * k + 2] = sc_xform(T + 8, v + 3 * k);
    }
    const size_t g = tri_global[t];
    world_c[3 * g + 0] = __ddiv_rn(__dadd_rn(__dadd_rn(w[0], w[3]), w[6]), 3.0);
    world_c[3 * g + 1] = __ddiv_rn(__dadd_rn(__dadd_rn(w[1], w[4]), w[7]), 3.0);
    world_c[3 * g + 2] = __ddiv_rn(__dadd_rn(__dadd_rn(w[2], w[5]), w[8]), 3.0);
    const double* q = node_n + 3 * (size_t)t;
    double nx = sc_rot(T + 0, q), ny = sc_rot(T + 4, q), nz = sc_rot(T + 8, q);
    const double len = __dsqrt_rn(__dadd_rn(__dadd_rn(__dmul_rn(nx, nx), __dmul_rn(ny, ny)), __dmul_rn(nz, nz)));
    if (len > 1e-15) { nx = __ddiv_rn(nx, len); ny = __ddiv_rn(ny, len); nz = __ddiv_rn(nz, len); }
    world_n[3 * g + 0] = nx;
    world_n[3 * g + 1] = ny;
    world_n[3 * g + 2] = nz;
}

namespace {
struct SceneGpuState {
    SceneGpuView view;
    double*       d_world_c   = nullptr;
    double*       d_world_n   = nullptr;
    double*       d_node_T    = nullptr;
    double*       d_node_v    = nullptr;
    double*       d_node_n    = nullptr;
    unsigned int* d_tri_node  = nullptr;
    unsigned int* d_tri_glob  = nullptr;
    unsigned int  M           = 0;       // movable triangles
    std::size_t   nodes       = 0;
    bool          valid       = false;
    // per part: its movable triangles in the node arrays, and the transform
    // the device data was computed with (only moved parts are recomputed)
    std::vector<unsigned int>           part_off, part_cnt;
    std::vector<std::array<double, 12>> applied;

    void release() {
        auto cf = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        cf(d_world_c); cf(d_world_n); cf(d_node_T); cf(d_node_v); cf(d_node_n);
        cf(d_tri_node); cf(d_tri_glob);
        M = 0; nodes = 0; valid = false; view = SceneGpuView{};
        part_off.clear(); part_cnt.clear(); applied.clear();
    }
};
SceneGpuState g_scene;
}  // namespace

const SceneGpuView& scene_gpu_sync(const InstanceScene& s)
{
    SceneGpuState& g = g_scene;
    if (!g.valid || g.view.structure_version != s.structure_version || g.view.N != s.N) {
        g.release();
        const std::size_t N = s.N;
        // Fixed body: body frame = part frame (bit-identical). Movable parts:
        // filled by the kernel below.
        std::vector<double> c(3 * N, 0.0), n(3 * N, 0.0);
        std::vector<double> nv, nn;
        std::vector<unsigned int> tnode, tglob;
        for (const SceneInstance& part : s.instances) {
            g.part_off.push_back((unsigned int)tglob.size());
            std::size_t t = 0;
            for (std::size_t r = 0; r < part.first.size(); ++r)
                for (std::uint32_t i = part.first[r]; i < part.first[r] + part.count[r]; ++i, ++t) {
                    if (part.node < 0) {
                        const double* v = &s.part_vertices[9 * (std::size_t)i];
                        c[3 * i + 0] = (v[0] + v[3] + v[6]) / 3.0;
                        c[3 * i + 1] = (v[1] + v[4] + v[7]) / 3.0;
                        c[3 * i + 2] = (v[2] + v[5] + v[8]) / 3.0;
                        for (int k = 0; k < 3; ++k) n[3 * i + k] = s.part_normals[3 * (std::size_t)i + k];
                    } else {
                        nv.insert(nv.end(), &part.node_vertices[9 * t], &part.node_vertices[9 * t] + 9);
                        nn.insert(nn.end(), &part.node_normals[3 * t], &part.node_normals[3 * t] + 3);
                        tnode.push_back(part.tri_node[t]);
                        tglob.push_back(i);
                    }
                }
        }
        for (std::size_t k = 0; k < s.instances.size(); ++k)
            g.part_cnt.push_back((k + 1 < s.instances.size() ? g.part_off[k + 1] : (unsigned int)tglob.size())
                                 - g.part_off[k]);
        g.applied.assign(s.instances.size(), std::array<double, 12>{});
        for (auto& a : g.applied) a[0] = std::nan("");   // first sync computes every movable part
        g.M = (unsigned int)tglob.size();
        g.nodes = s.node_T.size();
        SCENE_CUDA_CHECK(cudaMalloc(&g.d_world_c, 3 * N * sizeof(double)));
        SCENE_CUDA_CHECK(cudaMalloc(&g.d_world_n, 3 * N * sizeof(double)));
        SCENE_CUDA_CHECK(cudaMemcpy(g.d_world_c, c.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
        SCENE_CUDA_CHECK(cudaMemcpy(g.d_world_n, n.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
        SCENE_CUDA_CHECK(cudaMalloc(&g.d_node_T, std::max<std::size_t>(g.nodes, 1) * 12 * sizeof(double)));
        if (g.M) {
            SCENE_CUDA_CHECK(cudaMalloc(&g.d_node_v, (std::size_t)g.M * 9 * sizeof(double)));
            SCENE_CUDA_CHECK(cudaMalloc(&g.d_node_n, (std::size_t)g.M * 3 * sizeof(double)));
            SCENE_CUDA_CHECK(cudaMalloc(&g.d_tri_node, (std::size_t)g.M * sizeof(unsigned int)));
            SCENE_CUDA_CHECK(cudaMalloc(&g.d_tri_glob, (std::size_t)g.M * sizeof(unsigned int)));
            SCENE_CUDA_CHECK(cudaMemcpy(g.d_node_v, nv.data(), nv.size() * sizeof(double), cudaMemcpyHostToDevice));
            SCENE_CUDA_CHECK(cudaMemcpy(g.d_node_n, nn.data(), nn.size() * sizeof(double), cudaMemcpyHostToDevice));
            SCENE_CUDA_CHECK(cudaMemcpy(g.d_tri_node, tnode.data(), tnode.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
            SCENE_CUDA_CHECK(cudaMemcpy(g.d_tri_glob, tglob.data(), tglob.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }
        g.view.structure_version = s.structure_version;
        g.view.pose_version      = 0;   // forces the kernel below
        g.view.N                 = N;
        g.view.d_world_c         = g.d_world_c;
        g.view.d_world_n         = g.d_world_n;
        g.valid = true;
    }
    if (g.view.pose_version != s.pose_version) {
        bool uploaded = false;
        for (std::size_t k = 0; k < s.instances.size(); ++k) {
            const SceneInstance& part = s.instances[k];
            if (part.node < 0 || g.part_cnt[k] == 0) continue;
            if (std::memcmp(g.applied[k].data(), part.T.data(), sizeof(part.T)) == 0) continue;
            if (!uploaded) {
                std::vector<double> T(12 * g.nodes);
                for (std::size_t n = 0; n < g.nodes; ++n) std::memcpy(&T[12 * n], s.node_T[n].data(), 12 * sizeof(double));
                SCENE_CUDA_CHECK(cudaMemcpy(g.d_node_T, T.data(), T.size() * sizeof(double), cudaMemcpyHostToDevice));
                uploaded = true;
            }
            const int bs = 256;
            const unsigned int off = g.part_off[k], cnt = g.part_cnt[k];
            // legacy default stream: the modules' streams (blocking) wait for it
            sceneWorldKernel<<<(cnt + bs - 1) / bs, bs>>>(g.d_node_T, g.d_node_v + 9 * (std::size_t)off,
                                                         g.d_node_n + 3 * (std::size_t)off, g.d_tri_node + off,
                                                         g.d_tri_glob + off, cnt, g.d_world_c, g.d_world_n);
            SCENE_CUDA_CHECK(cudaGetLastError());
            g.applied[k] = part.T;
        }
        g.view.pose_version = s.pose_version;
    }
    return g.view;
}

void scene_gpu_download(std::vector<double>& world_c, std::vector<double>& world_n)
{
    const SceneGpuState& g = g_scene;
    world_c.assign(3 * g.view.N, 0.0);
    world_n.assign(3 * g.view.N, 0.0);
    if (!g.valid) return;
    SCENE_CUDA_CHECK(cudaMemcpy(world_c.data(), g.d_world_c, world_c.size() * sizeof(double), cudaMemcpyDeviceToHost));
    SCENE_CUDA_CHECK(cudaMemcpy(world_n.data(), g.d_world_n, world_n.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
void scene_part_inverse(const std::array<double, 12>& T, double Rinv[9], double t[3])
{
    // general 3x3 inverse (the part transform is a rigid motion in practice)
    const double a = T[0], b = T[1], c = T[2], d = T[4], e = T[5], f = T[6], g = T[8], h = T[9], i = T[10];
    const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const double det = a * A + b * B + c * C;
    if (std::fabs(det) < 1e-300) throw std::runtime_error("singular part transform");
    const double s = 1.0 / det;
    Rinv[0] = A * s;  Rinv[1] = -(b * i - c * h) * s;  Rinv[2] = (b * f - c * e) * s;
    Rinv[3] = B * s;  Rinv[4] = (a * i - c * g) * s;   Rinv[5] = -(a * f - c * d) * s;
    Rinv[6] = C * s;  Rinv[7] = -(a * h - b * g) * s;  Rinv[8] = (a * e - b * d) * s;
    t[0] = T[3]; t[1] = T[7]; t[2] = T[11];
}

void scene_part_box(const std::array<double, 12>& T, const double bp[6], double pad, double bb[6])
{
    for (int k = 0; k < 3; ++k) { bb[k] = 1e300; bb[k + 3] = -1e300; }
    for (int m = 0; m < 8; ++m) {
        const double p[3] = {(m & 1) ? bp[3] : bp[0], (m & 2) ? bp[4] : bp[1], (m & 4) ? bp[5] : bp[2]};
        for (int r = 0; r < 3; ++r) {
            const double w = T[4 * r] * p[0] + T[4 * r + 1] * p[1] + T[4 * r + 2] * p[2] + T[4 * r + 3];
            bb[r] = std::min(bb[r], w);
            bb[r + 3] = std::max(bb[r + 3], w);
        }
    }
    for (int k = 0; k < 3; ++k) {
        const double ext = std::max(std::fabs(bb[k]), std::fabs(bb[k + 3]));
        bb[k] -= pad * (1.0 + ext);
        bb[k + 3] += pad * (1.0 + ext);
    }
}

void scene_moved_ranges(const InstanceScene& s, std::vector<std::array<double, 12>>& applied,
                        std::vector<std::uint32_t>& first, std::vector<std::uint32_t>& count)
{
    first.clear();
    count.clear();
    if (applied.size() != s.instances.size()) {
        applied.assign(s.instances.size(), std::array<double, 12>{});
        for (auto& a : applied) a[0] = std::nan("");   // never equal: every movable part counts as moved
    }
    for (std::size_t k = 0; k < s.instances.size(); ++k) {
        const SceneInstance& part = s.instances[k];
        if (part.node < 0) continue;
        if (std::memcmp(applied[k].data(), part.T.data(), sizeof(part.T)) == 0) continue;
        applied[k] = part.T;
        first.insert(first.end(), part.first.begin(), part.first.end());
        count.insert(count.end(), part.count.begin(), part.count.end());
    }
}

void scene_grid_extents(const InstanceScene& s, const double u[3], const double v[3], const double sd[3],
                        double& u_min, double& u_max, double& v_min, double& v_max, double& s_max)
{
    const double ux = u[0], uy = u[1], uz = u[2], vx = v[0], vy = v[1], vz = v[2];
    const double sx = sd[0], sy = sd[1], sz = sd[2];
    u_min = 1e30; u_max = -1e30; v_min = 1e30; v_max = -1e30; s_max = -1e30;
    for (const SceneInstance& part : s.instances) {
        // the axes in the part frame and the offsets of the part origin
        double pu[3] = {ux, uy, uz}, pv[3] = {vx, vy, vz}, ps[3] = {sx, sy, sz};
        double ou = 0.0, ov = 0.0, os = 0.0;
        if (part.node >= 0) {
            const auto& T = part.T;
            for (int k = 0; k < 3; ++k) {
                pu[k] = T[k] * ux + T[4 + k] * uy + T[8 + k] * uz;
                pv[k] = T[k] * vx + T[4 + k] * vy + T[8 + k] * vz;
                ps[k] = T[k] * sx + T[4 + k] * sy + T[8 + k] * sz;
            }
            ou = T[3] * ux + T[7] * uy + T[11] * uz;
            ov = T[3] * vx + T[7] * vy + T[11] * vz;
            os = T[3] * sx + T[7] * sy + T[11] * sz;
        }
        for (std::size_t r = 0; r < part.first.size(); ++r)
            for (std::uint32_t i = part.first[r]; i < part.first[r] + part.count[r]; ++i)
                for (int k = 0; k < 3; ++k) {
                    const double* pt = &s.part_vertices[9 * (std::size_t)i + 3 * k];
                    double uc = pt[0] * pu[0] + pt[1] * pu[1] + pt[2] * pu[2];
                    double vc = pt[0] * pv[0] + pt[1] * pv[1] + pt[2] * pv[2];
                    double sc = pt[0] * ps[0] + pt[1] * ps[1] + pt[2] * ps[2];
                    if (part.node >= 0) { uc += ou; vc += ov; sc += os; }
                    u_min = std::min(u_min, uc); u_max = std::max(u_max, uc);
                    v_min = std::min(v_min, vc); v_max = std::max(v_max, vc);
                    s_max = std::max(s_max, sc);
                }
    }
}

void scene_grid_extents_f(const InstanceScene& s, const float u[3], const float v[3], const float sd[3],
                          float& u_min, float& u_max, float& v_min, float& v_max, float& s_max)
{
    u_min = 1e30f; u_max = -1e30f; v_min = 1e30f; v_max = -1e30f; s_max = -1e30f;
    for (const SceneInstance& part : s.instances) {
        if (part.node < 0) {
            for (std::size_t r = 0; r < part.first.size(); ++r)
                for (std::uint32_t i = part.first[r]; i < part.first[r] + part.count[r]; ++i)
                    for (int k = 0; k < 3; ++k) {
                        const double* p = &s.part_vertices[9 * (std::size_t)i + 3 * k];
                        const float x = (float)p[0], y = (float)p[1], z = (float)p[2];
                        const float pu = x*u[0] + y*u[1] + z*u[2];
                        const float pv = x*v[0] + y*v[1] + z*v[2];
                        const float ps = x*sd[0] + y*sd[1] + z*sd[2];
                        u_min = std::min(u_min, pu); u_max = std::max(u_max, pu);
                        v_min = std::min(v_min, pv); v_max = std::max(v_max, pv);
                        s_max = std::max(s_max, ps);
                    }
            continue;
        }
        // movable part: (R^T a).p + a.t in double, then rounded
        const auto& T = part.T;
        double pa[3][3], off[3];
        const float* ax[3] = {u, v, sd};
        for (int a = 0; a < 3; ++a) {
            for (int k = 0; k < 3; ++k)
                pa[a][k] = T[k] * ax[a][0] + T[4 + k] * ax[a][1] + T[8 + k] * ax[a][2];
            off[a] = T[3] * ax[a][0] + T[7] * ax[a][1] + T[11] * ax[a][2];
        }
        for (std::size_t r = 0; r < part.first.size(); ++r)
            for (std::uint32_t i = part.first[r]; i < part.first[r] + part.count[r]; ++i)
                for (int k = 0; k < 3; ++k) {
                    const double* p = &s.part_vertices[9 * (std::size_t)i + 3 * k];
                    float c[3];
                    for (int a = 0; a < 3; ++a)
                        c[a] = (float)(p[0] * pa[a][0] + p[1] * pa[a][1] + p[2] * pa[a][2] + off[a]);
                    u_min = std::min(u_min, c[0]); u_max = std::max(u_max, c[0]);
                    v_min = std::min(v_min, c[1]); v_max = std::max(v_max, c[1]);
                    s_max = std::max(s_max, c[2]);
                }
    }
}
