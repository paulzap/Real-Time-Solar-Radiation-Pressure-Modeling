#pragma once
// =============================================================================
// gpu_instance_scene.h - internal: two-level acceleration structures on CUDA.
//
// The model is split into rigid parts (instance_scene.h). Each CUDA module
// builds one BVH per part in the part frame and keeps a small part table in
// its own __constant__ memory: a ray in the body frame is moved into the part
// frame by the inverse transform, WITHOUT renormalising the direction, so the
// hit distance t is the same in both frames and the closest hit can be taken
// over all parts with one running t. Triangles keep their global indices.
//
// Body-frame centroids and normals (used for ray origins, forces and moment
// arms) live in one shared device copy, updated on every pose change by
// scene_gpu_sync() with the arithmetic of SatelliteDataset::rebuildTriangles,
// so they are bit-identical to triangles(). The modules copy the values of
// the parts that moved into their own layouts.
// =============================================================================
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include "instance_scene.h"

#define SCENE_MAX_PARTS 64

// Part table entry. Real = float for the centroid modules, double for the
// pixel-grid modules (the precision of their traversal).
template <class Real>
struct ScenePartDev {
    Real         Rinv[9];   // body -> part, linear part (inverse of the part transform)
    Real         t[3];      // part origin in the body frame
    Real         box[6];    // part bounds in the body frame (min xyz, max xyz), conservative
    unsigned int root;      // root of the part's tree in the module's node array
    int          identity;  // 1 = fixed body: the body-frame ray is used unchanged
};

template <class Real>
__device__ inline void scene_ray_to_part(const ScenePartDev<Real>& P, const Real o[3], const Real d[3],
                                         Real ol[3], Real dl[3])
{
    const Real x = o[0] - P.t[0], y = o[1] - P.t[1], z = o[2] - P.t[2];
    ol[0] = P.Rinv[0] * x + P.Rinv[1] * y + P.Rinv[2] * z;
    ol[1] = P.Rinv[3] * x + P.Rinv[4] * y + P.Rinv[5] * z;
    ol[2] = P.Rinv[6] * x + P.Rinv[7] * y + P.Rinv[8] * z;
    dl[0] = P.Rinv[0] * d[0] + P.Rinv[1] * d[1] + P.Rinv[2] * d[2];
    dl[1] = P.Rinv[3] * d[0] + P.Rinv[4] * d[1] + P.Rinv[5] * d[2];
    dl[2] = P.Rinv[6] * d[0] + P.Rinv[7] * d[1] + P.Rinv[8] * d[2];
}

// Slab test of a ray against a part box in the body frame: entry distance in
// tmin_out; false if the box is missed or lies behind the origin.
template <class Real>
__device__ inline bool scene_ray_box(const Real o[3], const Real d[3], const Real* box, Real& tmin_out)
{
    Real tmin = (Real)-1e30, tmax = (Real)1e30;
    for (int a = 0; a < 3; ++a) {
        if (fabs((double)d[a]) < 1e-20) {
            if (o[a] < box[a] || o[a] > box[a + 3]) return false;
            continue;
        }
        const Real inv = (Real)1 / d[a];
        Real t1 = (box[a] - o[a]) * inv, t2 = (box[a + 3] - o[a]) * inv;
        if (t1 > t2) { const Real tmp = t1; t1 = t2; t2 = tmp; }
        tmin = t1 > tmin ? t1 : tmin;
        tmax = t2 < tmax ? t2 : tmax;
        if (tmin > tmax) return false;
    }
    tmin_out = tmin;
    return tmax >= (Real)0;
}

// -----------------------------------------------------------------------------
// Host side
// -----------------------------------------------------------------------------

// Shared body-frame centroids and normals of all triangles for the current pose.
struct SceneGpuView {
    std::uint64_t structure_version = 0;
    std::uint64_t pose_version      = 0;
    std::size_t   N = 0;
    const double* d_world_c = nullptr;   // 3N, bit-identical to triangles() centroids
    const double* d_world_n = nullptr;   // 3N, bit-identical to triangles() normals
};

// Brings the shared copy up to the scene's pose (a kernel over the movable
// triangles on the legacy default stream, which the modules' blocking
// streams wait for; nothing to do when the pose did not change).
const SceneGpuView& scene_gpu_sync(const InstanceScene& s);

// Testing: the shared arrays as stored on the device.
void scene_gpu_download(std::vector<double>& world_c, std::vector<double>& world_n);

// Part transform rows (part -> body, 3x4) as 4x4 inverse pieces, and the
// body-frame box of a part-frame box (8 corners, then widened by pad).
void scene_part_inverse(const std::array<double, 12>& T, double Rinv[9], double t[3]);
void scene_part_box(const std::array<double, 12>& T, const double box_part[6], double pad, double box_body[6]);

template <class Real>
inline ScenePartDev<Real> scene_part_entry(const SceneInstance& part, const double box_part[6],
                                           unsigned int root, double pad)
{
    ScenePartDev<Real> e{};
    double Rinv[9], t[3], box[6];
    scene_part_inverse(part.T, Rinv, t);
    scene_part_box(part.T, box_part, pad, box);
    for (int k = 0; k < 9; ++k) e.Rinv[k] = (Real)Rinv[k];
    for (int k = 0; k < 3; ++k) e.t[k] = (Real)t[k];
    for (int k = 0; k < 6; ++k) e.box[k] = (Real)box[k];
    e.root = root;
    e.identity = part.node < 0 ? 1 : 0;
    return e;
}

// Extents of all vertices along the grid axes u, v and the sun direction s
// (pixel-grid window). Fixed body: the same products as on the body-frame
// triangle list, so bit-identical; a movable part uses (R^T u).p + u.t.
void scene_grid_extents(const InstanceScene& s, const double u[3], const double v[3], const double sd[3],
                        double& u_min, double& u_max, double& v_min, double& v_max, double& s_max);

// The same in single precision, as the RTX pixel-grid window is formed
// (fixed body: float vertices and float products, bit-identical).
void scene_grid_extents_f(const InstanceScene& s, const float u[3], const float v[3], const float sd[3],
                          float& u_min, float& u_max, float& v_min, float& v_max, float& s_max);

// Ranges [first, first + count) of the movable triangles whose part transform
// differs from `applied` (the transforms a module last used); updates `applied`.
void scene_moved_ranges(const InstanceScene& s, std::vector<std::array<double, 12>>& applied,
                        std::vector<std::uint32_t>& first, std::vector<std::uint32_t>& count);
