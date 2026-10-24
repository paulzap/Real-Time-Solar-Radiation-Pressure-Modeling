#pragma once
// =============================================================================
// instance_scene.h - internal header (not part of the public API).
//
// The model split into rigid parts for the two-level GPU acceleration
// structures. A part is the fixed body or one component declared movable in
// the HDF5 file ("revolute" / "prismatic") together with its fixed
// descendants; a movable component nested in another one is a part of its
// own. The split is fixed by the file. Triangles keep their global indices
// (the order of SatelliteDataset::triangles()), so labels and visualisation
// data are indexed exactly as before. The GPU builds one acceleration
// structure per part in the part frame and moves a part by changing its
// 3x4 transform only.
// =============================================================================
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

struct SceneInstance {
    int                        node = -1;  // movable scene-graph node, -1 = fixed body
    std::vector<std::uint32_t> first;      // contiguous ranges of global triangle indices:
    std::vector<std::uint32_t> count;      //   [first[k], first[k] + count[k])
    std::size_t                size = 0;   // number of triangles
    std::array<double, 12>     T{};        // part frame -> body frame, rows of a 3x4 matrix;
                                           // exactly the identity for the fixed body

    // Movable parts only, in range order: data to recompute the body-frame
    // centroids and normals bit-identically to SatelliteDataset::triangles().
    std::vector<std::uint32_t> tri_node;       // scene-graph node of each triangle
    std::vector<double>        node_vertices;  // 9 per triangle, node (file) frame
    std::vector<double>        node_normals;   // 3 per triangle, node frame
};

struct InstanceScene {
    std::uint64_t structure_version = 0;   // changes when a file is read
    std::uint64_t pose_version      = 0;   // changes on every committed joint change
    std::size_t   N = 0;                   // number of triangles

    std::vector<SceneInstance> instances;  // non-empty parts, file order of the movable node

    std::vector<std::uint32_t> tri_instance;   // part of each triangle
    std::vector<double> part_vertices;         // 9 per triangle: v1, v2, v3 in the part frame;
                                               // for the fixed body bit-identical to triangles()
    std::vector<double> part_normals;          // 3 per triangle, part frame

    std::vector<std::array<double, 12>> node_T; // node frame -> body frame, current pose

    bool single() const { return instances.size() <= 1; }
};

// -----------------------------------------------------------------------------
// Body-frame vertices, centroid and normal of triangle t of a movable part,
// with exactly the arithmetic of SatelliteDataset::rebuildTriangles (same
// operation order, no fused multiply-add), so the results are bit-identical
// to triangles(). GPU code reproduces this with __dmul_rn / __dadd_rn.
// -----------------------------------------------------------------------------
inline void scene_node_to_body(const std::array<double, 12>& T, const double* p, double* o) {
    o[0] = T[0] * p[0] + T[1] * p[1] + T[2]  * p[2] + T[3];
    o[1] = T[4] * p[0] + T[5] * p[1] + T[6]  * p[2] + T[7];
    o[2] = T[8] * p[0] + T[9] * p[1] + T[10] * p[2] + T[11];
}

inline void scene_body_triangle(const InstanceScene& s, const SceneInstance& part, std::size_t t,
                                double v[9], double c[3], double n[3]) {
    const std::array<double, 12>& T = s.node_T[part.tri_node[t]];
    for (int k = 0; k < 3; ++k) scene_node_to_body(T, &part.node_vertices[9 * t + 3 * k], &v[3 * k]);
    c[0] = (v[0] + v[3] + v[6]) / 3.0;
    c[1] = (v[1] + v[4] + v[7]) / 3.0;
    c[2] = (v[2] + v[5] + v[8]) / 3.0;
    const double* q = &part.node_normals[3 * t];
    n[0] = T[0] * q[0] + T[1] * q[1] + T[2]  * q[2];
    n[1] = T[4] * q[0] + T[5] * q[1] + T[6]  * q[2];
    n[2] = T[8] * q[0] + T[9] * q[1] + T[10] * q[2];
    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len > 1e-15) { n[0] /= len; n[1] /= len; n[2] /= len; }
}
