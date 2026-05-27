#include "SatelliteDataset.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>
#include <ostream>
#include <numeric>
#include <sstream>

// HighFive for HDF5 reading
#include <highfive/highfive.hpp>

// =========================================================================
// Construction
// =========================================================================
SatelliteDataset::SatelliteDataset(const std::string& folder_path) {
    scanFolder(folder_path);
}

// =========================================================================
// Folder / file list
// =========================================================================
void SatelliteDataset::setFolder(const std::string& folder_path) {
    scanFolder(folder_path);
}

void SatelliteDataset::scanFolder(const std::string& folder_path) {
    file_names_.clear();
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        auto ext = entry.path().extension().string();
        if (ext == ".h5" || ext == ".hdf5")
            file_names_.push_back(entry.path().string());
    }
    if (file_names_.empty())
        throw std::runtime_error("No .h5 / .hdf5 files found in " + folder_path);
    std::sort(file_names_.begin(), file_names_.end());
}

int SatelliteDataset::fileCount() const {
    return static_cast<int>(file_names_.size());
}

const std::vector<std::string>& SatelliteDataset::fileNames() const {
    return file_names_;
}

std::string SatelliteDataset::currentFile() const {
    return current_file_;
}

// =========================================================================
// Loading
//
// load(file)  : parse HDF5 once into cached_nodes_, then build triangles
// load(index) : convenience overload
// reload()    : re-apply joints to cached_nodes_ WITHOUT re-reading HDF5
//
// The cache is invalidated only when the current file changes - articulation
// (setJoint + reload) is then a pure-CPU operation costing matrix math only,
// not file I/O. For a 750k-triangle satellite this drops reload() from
// ~1.5 s (file read) to ~30 ms (transform application).
// =========================================================================
void SatelliteDataset::load(const std::string& file_path) {
    current_file_ = file_path;
    load_h5_into_cache(file_path);   // slow: file read + HDF5 parse, once
    rebuildTriangles(joints_);       // fast: apply joints to cached nodes
}

void SatelliteDataset::load(int index) {
    if (index < 0 || index >= static_cast<int>(file_names_.size()))
        throw std::out_of_range("SatelliteDataset::load: index out of range");
    load(file_names_[index]);
}

void SatelliteDataset::reload() {
    if (current_file_.empty()) return;
    if (cached_nodes_.empty()) {
        // Cache was never populated for this file - fall back to a full load.
        load_h5_into_cache(current_file_);
    }
    rebuildTriangles(joints_);       // fast: no file I/O
}

// =========================================================================
// Triangle access
// =========================================================================
const std::vector<Triangle>& SatelliteDataset::triangles() const {
    return triangles_;
}

int SatelliteDataset::triangleCount() const {
    return static_cast<int>(triangles_.size());
}

// =========================================================================
// Component inspection
// =========================================================================
std::vector<std::string> SatelliteDataset::componentNames() const {
    std::vector<std::string> names;
    for (const auto& t : triangles_) {
        if (std::find(names.begin(), names.end(), t.component_name) == names.end())
            names.push_back(t.component_name);
    }
    return names;
}

std::vector<SatelliteDataset::ComponentInfo> SatelliteDataset::componentInfos() const {
    std::vector<std::string> order;
    std::map<std::string, ComponentInfo> info_map;

    for (const auto& t : triangles_) {
        auto it = info_map.find(t.component_name);
        if (it == info_map.end()) {
            ComponentInfo ci;
            ci.name           = t.component_name;
            ci.type           = t.component_type;
            ci.triangle_count = 1;
            ci.total_area     = t.area;
            info_map[t.component_name] = ci;
            order.push_back(t.component_name);
        } else {
            it->second.triangle_count++;
            it->second.total_area += t.area;
        }
    }

    std::vector<ComponentInfo> result;
    result.reserve(order.size());
    for (const auto& n : order)
        result.push_back(info_map.at(n));
    return result;
}

bool SatelliteDataset::hasComponent(const std::string& name) const {
    for (const auto& t : triangles_)
        if (t.component_name == name) return true;
    return false;
}

std::vector<const Triangle*>
SatelliteDataset::componentTriangles(const std::string& name) const {
    std::vector<const Triangle*> result;
    for (const auto& t : triangles_)
        if (t.component_name == name) result.push_back(&t);
    return result;
}

// =========================================================================
// Articulation - general joints
// =========================================================================
void SatelliteDataset::setJoint(const std::string& name, const JointConfig& jc) {
    joints_[name] = jc;
}

void SatelliteDataset::clearJoint(const std::string& name) {
    joints_.erase(name);
}

void SatelliteDataset::clearAllJoints() {
    joints_.clear();
}

JointConfig SatelliteDataset::getJoint(const std::string& name) const {
    auto it = joints_.find(name);
    return (it != joints_.end()) ? it->second : JointConfig{};
}

bool SatelliteDataset::hasJoint(const std::string& name) const {
    return joints_.count(name) > 0;
}

const std::map<std::string, JointConfig>& SatelliteDataset::joints() const {
    return joints_;
}

void SatelliteDataset::articulate(const std::string& name, const JointConfig& jc) {
    setJoint(name, jc);
    reload();
}

// =========================================================================
// Display helpers
// =========================================================================
void SatelliteDataset::printInfo(std::ostream& out) const {
    namespace fs = std::filesystem;
    out << "File      : "
        << (current_file_.empty() ? "(none)"
                                  : fs::path(current_file_).filename().string())
        << "\n"
        << "Triangles : " << triangles_.size() << "\n"
        << "Components: " << componentNames().size() << "\n";
}

void SatelliteDataset::printComponents(std::ostream& out) const {
    auto infos = componentInfos();
    if (infos.empty()) { out << "(no components loaded)\n"; return; }

    out << "\nComponents:\n"
        << std::left
        << "  " << std::setw(24) << "Name"
        << std::setw(22) << "Type"
        << std::setw(8)  << "Tris"
        << "Area (m2)\n"
        << "  " << std::string(66, '-') << "\n";
    for (const auto& ci : infos) {
        out << "  " << std::setw(24) << ci.name
            << std::setw(22) << ci.type
            << std::setw(8)  << ci.triangle_count
            << std::fixed << std::setprecision(3) << ci.total_area << "\n";
    }
}

void SatelliteDataset::printJoints(std::ostream& out) const {
    if (joints_.empty()) { out << "Joints: (none)\n"; return; }
    const double rad2deg = 180.0 / std::acos(-1.0);
    out << "\nJoints:\n"
        << std::left
        << "  " << std::setw(24) << "Component"
        << std::setw(12) << "Type"
        << std::setw(20) << "Axis"
        << "Value\n"
        << "  " << std::string(66, '-') << "\n";
    for (const auto& [name, jc] : joints_) {
        std::string type_str = (jc.type == JointConfig::Type::REVOLUTE)
                               ? "revolute" : "prismatic";
        std::ostringstream axis_ss;
        axis_ss << std::fixed << std::setprecision(2)
                << "(" << jc.ax << "," << jc.ay << "," << jc.az << ")";
        std::ostringstream val_ss;
        if (jc.type == JointConfig::Type::REVOLUTE)
            val_ss << std::fixed << std::setprecision(4) << jc.value
                   << " rad (" << std::setprecision(2) << (jc.value * rad2deg) << " deg)";
        else
            val_ss << std::fixed << std::setprecision(4) << jc.value << " m";

        out << "  " << std::setw(24) << name
            << std::setw(12) << type_str
            << std::setw(20) << axis_ss.str()
            << val_ss.str() << "\n";
        if (jc.type == JointConfig::Type::REVOLUTE) {
            out << "  " << std::setw(24) << ""
                << "pivot: "
                << (jc.use_auto_pivot ? "auto (mesh centroid)" :
                    "(" + std::to_string(jc.px) + "," +
                          std::to_string(jc.py) + "," +
                          std::to_string(jc.pz) + ")")
                << "\n";
        }
    }
}

// =========================================================================
// Legacy interface
// =========================================================================
void SatelliteDataset::setJointZ(const std::string& name, double angle_rad) {
    setJoint(name, JointConfig::rotZ(angle_rad));
}

void SatelliteDataset::rotateZ(const std::string& name, double angle_rad) {
    articulate(name, JointConfig::rotZ(angle_rad));
}

double SatelliteDataset::getJointZ(const std::string& name) const {
    auto it = joints_.find(name);
    if (it == joints_.end()) return 0.0;
    return (it->second.type == JointConfig::Type::REVOLUTE) ? it->second.value : 0.0;
}

void SatelliteDataset::load_single_file(
        const std::string& file_path,
        const std::map<std::string, double>& z_joint_overrides)
{
    // Convert legacy double-map to JointConfig map, apply for this load only
    std::map<std::string, JointConfig> jc_overrides;
    for (const auto& [name, angle] : z_joint_overrides)
        jc_overrides[name] = JointConfig::rotZ(angle);
    // Merge persistent joints (persistent take lower priority - override wins)
    for (const auto& [name, jc] : joints_)
        if (!jc_overrides.count(name)) jc_overrides[name] = jc;

    current_file_ = file_path;
    load_h5_into_cache(file_path);
    rebuildTriangles(jc_overrides);
}

const std::vector<Triangle>& SatelliteDataset::get_triangles() const {
    return triangles_;
}

const std::vector<std::string>& SatelliteDataset::get_file_names() const {
    return file_names_;
}

// =========================================================================
// 4×4 matrix helpers (row-major, flat array[16])
//
// Layout:  index = row*4 + col
//   [ 0  1  2  3 ]
//   [ 4  5  6  7 ]
//   [ 8  9 10 11 ]
//   [12 13 14 15 ]
// Translation stored in column 3: T[3]=tx, T[7]=ty, T[11]=tz.
// =========================================================================
SatelliteDataset::Mat4 SatelliteDataset::identity4() {
    Mat4 m{};
    m[0] = m[5] = m[10] = m[15] = 1.0;
    return m;
}

SatelliteDataset::Mat4 SatelliteDataset::mul4(const Mat4& A, const Mat4& B) {
    Mat4 C{};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            for (int k = 0; k < 4; ++k)
                C[r*4+c] += A[r*4+k] * B[k*4+c];
    return C;
}

SatelliteDataset::Mat4 SatelliteDataset::makeTrans(double tx, double ty, double tz) {
    Mat4 T = identity4();
    T[3] = tx;  T[7] = ty;  T[11] = tz;
    return T;
}

// Rodrigues' rotation formula for axis (ax,ay,az) - must be a unit vector.
SatelliteDataset::Mat4 SatelliteDataset::makeRotAxis(
        double ax, double ay, double az, double angle) {
    double len = std::sqrt(ax*ax + ay*ay + az*az);
    if (len < 1e-12) return identity4();
    ax /= len; ay /= len; az /= len;

    const double c = std::cos(angle), s = std::sin(angle), t = 1.0 - c;
    Mat4 R = identity4();
    R[0]  = t*ax*ax + c;      R[1]  = t*ax*ay - s*az;  R[2]  = t*ax*az + s*ay;
    R[4]  = t*ax*ay + s*az;   R[5]  = t*ay*ay + c;      R[6]  = t*ay*az - s*ax;
    R[8]  = t*ax*az - s*ay;   R[9]  = t*ay*az + s*ax;   R[10] = t*az*az + c;
    return R;
}

// Rotation around axis (ax,ay,az) through point (cx,cy,cz):
//   Trans(+C) · Rot(axis,angle) · Trans(-C)
SatelliteDataset::Mat4 SatelliteDataset::makeRotAxisAroundPoint(
        double ax, double ay, double az, double angle,
        double cx, double cy, double cz)
{
    Mat4 T_to_origin   = makeTrans(-cx, -cy, -cz);
    Mat4 R             = makeRotAxis(ax, ay, az, angle);
    Mat4 T_from_origin = makeTrans(cx, cy, cz);
    return mul4(T_from_origin, mul4(R, T_to_origin));
}

void SatelliteDataset::transformPoint(const Mat4& T,
    double ix, double iy, double iz,
    double& ox, double& oy, double& oz)
{
    ox = T[0]*ix + T[1]*iy + T[2] *iz + T[3];
    oy = T[4]*ix + T[5]*iy + T[6] *iz + T[7];
    oz = T[8]*ix + T[9]*iy + T[10]*iz + T[11];
}

void SatelliteDataset::transformNormal(const Mat4& T,
    double ix, double iy, double iz,
    double& ox, double& oy, double& oz)
{
    // Rotation only (top-left 3×3), then renormalise
    ox = T[0]*ix + T[1]*iy + T[2] *iz;
    oy = T[4]*ix + T[5]*iy + T[6] *iz;
    oz = T[8]*ix + T[9]*iy + T[10]*iz;
    double len = std::sqrt(ox*ox + oy*oy + oz*oz);
    if (len > 1e-15) { ox /= len; oy /= len; oz /= len; }
}

// =========================================================================
// buildJointTransform - compute the 4×4 T_joint matrix for a JointConfig.
//
// For REVOLUTE:
//   If use_auto_pivot=true: compute centroid of mesh vertices as pivot.
//   Apply Trans(pivot) · Rot(axis,angle) · Trans(-pivot).
//
// For PRISMATIC:
//   Translate along axis by value metres.
// =========================================================================
SatelliteDataset::Mat4 SatelliteDataset::buildJointTransform(
        const JointConfig& jc,
        const std::vector<std::vector<double>>& vertices)
{
    if (jc.type == JointConfig::Type::PRISMATIC) {
        double len = std::sqrt(jc.ax*jc.ax + jc.ay*jc.ay + jc.az*jc.az);
        if (len < 1e-12) return identity4();
        double inv = jc.value / len;
        return makeTrans(jc.ax * inv, jc.ay * inv, jc.az * inv);
    }

    // REVOLUTE - determine pivot
    double cx = jc.px, cy = jc.py, cz = jc.pz;
    if (jc.use_auto_pivot && !vertices.empty()) {
        cx = cy = cz = 0.0;
        for (const auto& v : vertices) { cx += v[0]; cy += v[1]; cz += v[2]; }
        double n = static_cast<double>(vertices.size());
        cx /= n; cy /= n; cz /= n;
    }

    return makeRotAxisAroundPoint(jc.ax, jc.ay, jc.az, jc.value, cx, cy, cz);
}

// =========================================================================
// Helper: read a 1-D or (M,1) HDF5 dataset as flat vector<double>
// =========================================================================
static std::vector<double> readFlatVector(const HighFive::DataSet& ds) {
    auto dims = ds.getDimensions();
    if (dims.size() == 1) {
        std::vector<double> v;
        ds.read(v);
        return v;
    }
    std::vector<std::vector<double>> v2d;
    ds.read(v2d);
    std::vector<double> v(v2d.size());
    for (size_t i = 0; i < v2d.size(); ++i) v[i] = v2d[i][0];
    return v;
}

// =========================================================================
// load_h5_into_cache - recursive traversal of the SatForm3D scene graph.
//
// Parses the file ONCE and stores each component (name, parent index,
// T_local_offset, raw local-frame vertices / normals / triangles / optical
// data) into cached_nodes_. No joints are applied here.
//
// cached_nodes_ is laid out in DFS pre-order, so each entry's parent is
// guaranteed to appear earlier in the vector - this lets rebuildTriangles
// resolve T_body in a single forward pass.
// =========================================================================
void SatelliteDataset::load_h5_into_cache(const std::string& file_path)
{
    cached_nodes_.clear();

    HighFive::File file(file_path, HighFive::File::ReadOnly);

    auto objects = file.listObjectNames();
    if (objects.empty())
        throw std::runtime_error("HDF5 file has no root group: " + file_path);

    auto root_group = file.getGroup(objects[0]);

    std::function<void(const HighFive::Group&, int, const std::string&)> visit;
    visit = [&](const HighFive::Group& group, int parent_idx, const std::string& gname) {
        CachedNode node;
        node.name = gname;
        node.parent_index = parent_idx;
        node.T_local_offset = identity4();

        if (group.hasAttribute("T_local_offset")) {
            std::vector<double> t_flat;
            group.getAttribute("T_local_offset").read(t_flat);
            if (t_flat.size() == 16)
                for (int i = 0; i < 16; ++i) node.T_local_offset[i] = t_flat[i];
        }

        if (group.hasAttribute("component_type"))
            group.getAttribute("component_type").read(node.component_type);

        if (group.exist("mesh")) {
            auto mg = group.getGroup("mesh");
            node.has_mesh = true;
            mg.getDataSet("vertices").read(node.vertices_local);
            mg.getDataSet("normals").read(node.normals_local);
            mg.getDataSet("triangles").read(node.tri_indices);
            node.areas       = readFlatVector(mg.getDataSet("areas"));
            node.reflectance = readFlatVector(mg.getDataSet("reflectance"));
            node.specularity = readFlatVector(mg.getDataSet("specularity"));
            node.emissivity  = readFlatVector(mg.getDataSet("emissivity"));
        }

        cached_nodes_.push_back(std::move(node));
        const int my_idx = static_cast<int>(cached_nodes_.size()) - 1;

        for (const auto& child_name : group.listObjectNames()) {
            if (child_name == "mesh") continue;
            auto child_group = group.getGroup(child_name);
            if (child_group.hasAttribute("joint_type"))
                visit(child_group, my_idx, child_name);
        }
    };

    visit(root_group, -1, objects[0]);
}

// =========================================================================
// rebuildTriangles - apply joints to the cached scene graph and produce
// triangles_ in body frame. Pure CPU math, no file I/O.
//
// For each cached node in DFS pre-order:
//   T_local = T_local_offset · T_joint(if joint configured for this name)
//   T_body  = T_body(parent)  · T_local        (root: parent T_body = identity)
//   if has_mesh: transform each triangle's vertices + normal into body frame
//
// This is the hot path of articulate() - every call to setJoint+reload or
// to engine.dataset().articulate() goes through here. For a 750k-triangle
// satellite the entire rebuild typically costs 20–80 ms (pure CPU math).
// =========================================================================
void SatelliteDataset::rebuildTriangles(
        const std::map<std::string, JointConfig>& joint_overrides)
{
    if (cached_nodes_.empty()) { triangles_.clear(); return; }

    // Pre-count total triangle count.
    size_t total_tris = 0;
    for (const auto& n : cached_nodes_) if (n.has_mesh) total_tris += n.tri_indices.size();

    // Detect first build vs subsequent articulation rebuild.
    // On first build (or when the triangle count changes - different HDF5 file)
    // we must populate the immutable per-triangle fields: ID, component_type,
    // component_name, optical properties, area. On articulation rebuilds the
    // immutable fields are already correct, so we only refresh geometry
    // (vertices, normal, centroid). This avoids ~3*N std::string allocations
    // per articulation - the dominant cost for a 750k-triangle mesh.
    const bool first_build = (triangles_.size() != total_tris);
    triangles_.resize(total_tris);

    // T_body for each cached node; index aligns with cached_nodes_.
    std::vector<Mat4> body_transforms(cached_nodes_.size(), identity4());

    size_t write_idx = 0;
    for (size_t i = 0; i < cached_nodes_.size(); ++i) {
        const CachedNode& node = cached_nodes_[i];

        // T_local = T_local_offset · T_joint (if a joint is registered for this node).
        // buildJointTransform internally inspects jc.use_auto_pivot / jc.type and
        // only reads `vertices` when it computes the auto-pivot centroid; for
        // PRISMATIC or explicit-pivot joints the reference is ignored.
        Mat4 T_local = node.T_local_offset;
        auto it = joint_overrides.find(node.name);
        if (it != joint_overrides.end()) {
            Mat4 T_joint = buildJointTransform(it->second, node.vertices_local);
            T_local = mul4(T_local, T_joint);
        }

        // T_body = T_body(parent) · T_local
        Mat4 T_body = (node.parent_index < 0)
            ? T_local
            : mul4(body_transforms[node.parent_index], T_local);
        body_transforms[i] = T_body;

        if (!node.has_mesh) continue;

        const int M = static_cast<int>(node.tri_indices.size());

        for (int t = 0; t < M; ++t) {
            Triangle& tri = triangles_[write_idx];
            const int i0 = node.tri_indices[t][0];
            const int i1 = node.tri_indices[t][1];
            const int i2 = node.tri_indices[t][2];

            // Immutable structural fields - written only on first build / file switch.
            if (first_build) {
                tri.ID             = node.component_type + "_" + std::to_string(write_idx);
                tri.component_type = node.component_type;
                tri.component_name = node.name;
                tri.reflectance    = (t < (int)node.reflectance.size()) ? node.reflectance[t] : 0.0;
                tri.specularity    = (t < (int)node.specularity.size()) ? node.specularity[t] : 0.0;
                tri.emissivity     = (t < (int)node.emissivity.size())  ? node.emissivity[t]  : 0.0;
                tri.area           = (t < (int)node.areas.size())       ? node.areas[t]       : 0.0;
            }
            tri.label = 1.0;

            // Geometry - recomputed every rebuild (this is the only thing that
            // actually changes when a joint angle moves).
            transformPoint(T_body,
                node.vertices_local[i0][0], node.vertices_local[i0][1], node.vertices_local[i0][2],
                tri.v1_x, tri.v1_y, tri.v1_z);
            transformPoint(T_body,
                node.vertices_local[i1][0], node.vertices_local[i1][1], node.vertices_local[i1][2],
                tri.v2_x, tri.v2_y, tri.v2_z);
            transformPoint(T_body,
                node.vertices_local[i2][0], node.vertices_local[i2][1], node.vertices_local[i2][2],
                tri.v3_x, tri.v3_y, tri.v3_z);

            transformNormal(T_body,
                node.normals_local[t][0], node.normals_local[t][1], node.normals_local[t][2],
                tri.normal_x, tri.normal_y, tri.normal_z);

            tri.centroid_x = (tri.v1_x + tri.v2_x + tri.v3_x) / 3.0;
            tri.centroid_y = (tri.v1_y + tri.v2_y + tri.v3_y) / 3.0;
            tri.centroid_z = (tri.v1_z + tri.v2_z + tri.v3_z) / 3.0;

            ++write_idx;
        }
    }
}
