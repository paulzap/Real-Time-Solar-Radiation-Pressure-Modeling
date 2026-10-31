// SolaRT - solar radiation pressure on GPU RT cores.
// Copyright (c) 2026 Zapevalin P.R. SPDX-License-Identifier: MIT
// If you use this code in published work, please cite:
// https://doi.org/10.1016/j.ast.2026.113714

#include "SatelliteDataset.h"
#include "instance_scene.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>
#include <ostream>
#include <numeric>
#include <sstream>
#include <atomic>
#include <cstring>

// HighFive for HDF5 reading
#include <highfive/highfive.hpp>

// =========================================================================
// Geometry version counter - process-wide, so a version identifies one
// triangle set across all SatelliteDataset objects (the GPU caches it is
// compared against are process-wide too).
// =========================================================================
static std::uint64_t next_geometry_version() {
    static std::atomic<std::uint64_t> counter{0};
    return ++counter;
}

// =========================================================================
// Construction
// =========================================================================
SatelliteDataset::SatelliteDataset(const std::string& folder_path) {
    scanFolder(folder_path);
}

SatelliteDataset::SceneHolder::SceneHolder() : p_(std::make_unique<InstanceScene>()) {}
SatelliteDataset::SceneHolder::~SceneHolder() = default;
SatelliteDataset::SceneHolder::SceneHolder(const SceneHolder& o)
    : p_(std::make_unique<InstanceScene>(*o.p_)) {}
SatelliteDataset::SceneHolder& SatelliteDataset::SceneHolder::operator=(const SceneHolder& o) {
    if (this != &o) *p_ = *o.p_;
    return *this;
}
SatelliteDataset::SceneHolder::SceneHolder(SceneHolder&& o) noexcept
    : p_(std::make_unique<InstanceScene>(std::move(*o.p_))) {}
SatelliteDataset::SceneHolder& SatelliteDataset::SceneHolder::operator=(SceneHolder&& o) noexcept {
    if (this != &o) *p_ = std::move(*o.p_);
    return *this;
}

// =========================================================================
// Geometry version and mass properties
// =========================================================================
std::uint64_t SatelliteDataset::geometryVersion()  const { return geometry_version_; }
std::uint64_t SatelliteDataset::structureVersion() const { return structure_version_; }

bool SatelliteDataset::hasMassProperties() const { return has_mass_; }

static void require_mass(bool has, const std::string& file) {
    if (!has)
        throw std::runtime_error("No mass properties (/inertia group) in " + file);
}
double SatelliteDataset::totalMass() const {
    require_mass(has_mass_, current_file_);
    return mass_;
}
std::array<double, 3> SatelliteDataset::centerOfMass() const {
    require_mass(has_mass_, current_file_);
    return com_;
}
std::array<double, 9> SatelliteDataset::inertiaTensor() const {
    require_mass(has_mass_, current_file_);
    return inertia_;
}

// =========================================================================
// Folder / file list
// =========================================================================
void SatelliteDataset::setFolder(const std::string& folder_path) {
    scanFolder(folder_path);
}

void SatelliteDataset::scanFolder(const std::string& folder_path) {
    // A path to one file is a list of that file.
    if (std::filesystem::is_regular_file(folder_path)) {
        file_names_ = { folder_path };
        return;
    }
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        auto ext = entry.path().extension().string();
        if (ext == ".h5" || ext == ".hdf5")
            names.push_back(entry.path().string());
    }
    if (names.empty())
        throw std::runtime_error("No .h5 / .hdf5 files found in " + folder_path);
    std::sort(names.begin(), names.end());
    file_names_ = std::move(names);
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
// The cache is invalidated only when the current file changes. Articulation
// (setJoint + reload) only commits the joint values: the triangle list is
// rebuilt when triangles() is next called, and the GPU methods move the rigid
// parts of the model without the list at all.
// =========================================================================
// Reads a file and checks the joints against it. If either is refused, the
// dataset stays exactly as it was: the file is only taken once all of it is good.
void SatelliteDataset::switchFile(const std::string& file_path,
                                  const std::map<std::string, JointConfig>& joints) {
    std::vector<CachedNode> nodes = std::move(cached_nodes_);
    const std::string file = current_file_;
    const bool dirty = immutable_dirty_, had_mass = has_mass_;
    const double mass = mass_;
    const std::array<double, 3> com = com_;
    const std::array<double, 9> inertia = inertia_;
    try {
        current_file_ = file_path;
        load_h5_into_cache(file_path);   // slow: file read + HDF5 parse, once
        for (const auto& [name, jc] : joints) validateJoint(name, jc, false);
    } catch (...) {
        cached_nodes_    = std::move(nodes);
        current_file_    = file;
        immutable_dirty_ = dirty;
        has_mass_        = had_mass;
        mass_            = mass;
        com_             = com;
        inertia_         = inertia;
        throw;
    }
    resetRigidParts();
}

void SatelliteDataset::load(const std::string& file_path) {
    switchFile(file_path, joints_);
    commitJoints(joints_);
    rebuildTriangles(committed_joints_);
    triangles_stale_ = false;
}

void SatelliteDataset::resetRigidParts() {
    file_pose_T_ = computeBodyTransforms({});
    part_active_.assign(cached_nodes_.size(), 0);
    structure_version_ = 0;   // the next commitJoints() builds the split
}

void SatelliteDataset::commitJoints(const std::map<std::string, JointConfig>& joints) {
    committed_joints_ = joints;
    geometry_version_ = next_geometry_version();
    triangles_stale_  = true;

    // A movable component is a separate rigid part exactly while its body
    // transform differs from the pose stored in the file; otherwise it stays
    // with the fixed body. So the split, and with it every result, depends
    // only on the current pose, and a model that is not articulated keeps the
    // single-structure GPU path of 1.1.0. A change of the split rebuilds the
    // GPU structures once (it happens when a component leaves or returns to
    // its file pose, not on ordinary pose changes).
    bool rebuild = (structure_version_ == 0);
    if (!cached_nodes_.empty()) {
        const std::vector<Mat4> Tb = computeBodyTransforms(committed_joints_);
        std::vector<char> inside(cached_nodes_.size(), 0);   // below a separate part
        for (size_t i = 0; i < cached_nodes_.size(); ++i) {
            const CachedNode& n = cached_nodes_[i];
            const int p = n.parent_index;
            if (p >= 0) inside[i] = inside[p] || part_active_[p];
            if (p < 0 || !n.joint.movable()) continue;
            // a movable component below a separate part is separate too
            const char moved = inside[i] ||
                std::memcmp(Tb[i].data(), file_pose_T_[i].data(), sizeof(Mat4)) != 0;
            if (moved != part_active_[i]) {
                part_active_[i] = moved;
                rebuild = true;
            }
        }
    }
    if (rebuild) {
        structure_version_ = next_geometry_version();
        buildSceneStructure();
    }
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
        load(current_file_);
        return;
    }
    commitJoints(joints_);           // geometry follows on demand
}

// =========================================================================
// Triangle access
// =========================================================================
const std::vector<Triangle>& SatelliteDataset::triangles() const {
    if (triangles_stale_) {
        rebuildTriangles(committed_joints_);
        triangles_stale_ = false;
    }
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
    for (const auto& t : triangles()) {
        if (std::find(names.begin(), names.end(), t.component_name) == names.end())
            names.push_back(t.component_name);
    }
    return names;
}

std::vector<SatelliteDataset::ComponentInfo> SatelliteDataset::componentInfos() const {
    std::vector<std::string> order;
    std::map<std::string, ComponentInfo> info_map;

    for (const auto& t : triangles()) {
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
    for (const auto& t : triangles())
        if (t.component_name == name) return true;
    return false;
}

std::vector<const Triangle*>
SatelliteDataset::componentTriangles(const std::string& name) const {
    std::vector<const Triangle*> result;
    for (const auto& t : triangles())
        if (t.component_name == name) result.push_back(&t);
    return result;
}

// =========================================================================
// Articulation - general joints
// =========================================================================
void SatelliteDataset::setJoint(const std::string& name, const JointConfig& jc) {
    if (!cached_nodes_.empty()) validateJoint(name, jc, true);
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
// Declared joints (HDF5 attributes joint_type, joint_axis, joint_min,
// joint_max, joint_default_value)
// =========================================================================
std::vector<std::string> SatelliteDataset::movableComponents() const {
    std::vector<std::string> names;
    for (const auto& n : cached_nodes_)
        if (n.parent_index >= 0 && n.joint.movable() && n.driver < 0) names.push_back(n.name);
    return names;
}

std::vector<std::string> SatelliteDataset::deploymentStateNames() const {
    std::vector<std::string> names;
    for (const auto& n : cached_nodes_)
        for (const auto& [state, v] : n.joint.deploy_states)
            if (std::find(names.begin(), names.end(), state) == names.end()) names.push_back(state);
    std::sort(names.begin(), names.end());
    return names;
}

void SatelliteDataset::setDeploymentState(const std::string& state) {
    bool any = false;
    for (const auto& n : cached_nodes_) {
        if (n.parent_index < 0 || !n.joint.movable() || n.driver >= 0) continue;
        auto it = n.joint.deploy_states.find(state);
        if (it == n.joint.deploy_states.end()) continue;
        double v = it->second;
        if (n.joint.has_limits) v = std::min(std::max(v, n.joint.min_value), n.joint.max_value);
        setJoint(n.name, declaredJoint(n.joint, v));
        any = true;
    }
    if (!any) {
        std::string known;
        for (const auto& k : deploymentStateNames()) known += (known.empty() ? "" : ", ") + k;
        throw std::invalid_argument("SatelliteDataset: no component of " + current_file_ +
                                    " declares the deployment state '" + state + "' (declared: " +
                                    (known.empty() ? "none" : known) + ")");
    }
    reload();
}

JointInfo SatelliteDataset::jointInfo(const std::string& name) const {
    for (const auto& n : cached_nodes_)
        if (n.name == name) return n.joint;
    throw std::invalid_argument("SatelliteDataset: no component '" + name + "' in " + current_file_);
}

void SatelliteDataset::setJointValue(const std::string& name, double value) {
    articulate(name, declaredJoint(jointInfo(name), value));   // validateJoint rejects fixed and driven components
}

void SatelliteDataset::validateJoint(const std::string& name, const JointConfig& jc,
                                     bool unknown_is_error) const {
    for (double x : { jc.value, jc.ax, jc.ay, jc.az, jc.px, jc.py, jc.pz })
        if (!std::isfinite(x))
            throw std::invalid_argument("SatelliteDataset: joint of '" + name + "' is not finite");
    bool found = false;
    for (const auto& n : cached_nodes_) {
        if (n.name != name) continue;
        if (found)
            throw std::invalid_argument("SatelliteDataset: several components are named '" + name +
                                        "' in " + current_file_ + ", so the name does not say which to articulate");
        found = true;
        const bool revolute = jc.type == JointConfig::Type::REVOLUTE;
        if (n.parent_index < 0 || !n.joint.movable()) {
            std::string movable;
            for (const auto& m : movableComponents()) movable += (movable.empty() ? "" : ", ") + m;
            throw std::invalid_argument(
                "SatelliteDataset: component '" + name + "' is declared " +
                (n.parent_index < 0 ? std::string("as the root") : "'" + n.joint.type + "'") +
                " in " + current_file_ + " and cannot be articulated. Movable components: " +
                (movable.empty() ? "none" : movable));
        }
        if (n.driver >= 0) {
            std::ostringstream msg;
            msg << "SatelliteDataset: component '" << name << "' is driven by '" << n.joint.driver
                << "' (q = " << n.joint.ratio << " * q_driver + " << n.joint.offset << ") in "
                << current_file_ << "; articulate '" << n.joint.driver << "' instead";
            throw std::invalid_argument(msg.str());
        }
        if (revolute != (n.joint.type == "revolute"))
            throw std::invalid_argument(
                "SatelliteDataset: component '" + name + "' is declared '" + n.joint.type +
                "' in " + current_file_ + ", but a " + (revolute ? "revolute" : "prismatic") +
                " joint was given");

        // Axis: used as given; a different direction is reported once.
        const double len = std::sqrt(jc.ax * jc.ax + jc.ay * jc.ay + jc.az * jc.az);
        const auto& a = n.joint.axis;
        const double alen = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        double cosang = 0.0;
        if (len > 1e-12 && alen > 1e-12)
            cosang = (jc.ax * a[0] + jc.ay * a[1] + jc.az * a[2]) / (len * alen);
        if (std::fabs(cosang) < 1.0 - 1e-9 && !axis_warned_.count(name)) {
            axis_warned_[name] = true;
            std::cerr << "SatelliteDataset: joint axis (" << jc.ax << ", " << jc.ay << ", " << jc.az
                      << ") given for '" << name << "' is not the declared axis (" << a[0] << ", "
                      << a[1] << ", " << a[2] << "); using the given axis\n";
        }

        // Limits, for the value along the declared axis direction.
        if (n.joint.has_limits) {
            const double v   = (cosang < 0.0) ? -jc.value : jc.value;
            const double tol = 1e-9 * std::max({1.0, std::fabs(n.joint.min_value),
                                                std::fabs(n.joint.max_value)});
            if (v < n.joint.min_value - tol || v > n.joint.max_value + tol) {
                std::ostringstream msg;
                msg << "SatelliteDataset: value " << v << (revolute ? " rad" : " m") << " of '"
                    << name << "' is outside the declared range [" << n.joint.min_value << ", "
                    << n.joint.max_value << "] in " << current_file_;
                throw std::invalid_argument(msg.str());
            }
        }
    }
    if (!found && unknown_is_error)
        throw std::invalid_argument("SatelliteDataset: no component '" + name + "' in " + current_file_);
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

    switchFile(file_path, jc_overrides);
    commitJoints(jc_overrides);
    rebuildTriangles(committed_joints_);
    triangles_stale_ = false;
}

const std::vector<Triangle>& SatelliteDataset::get_triangles() const {
    return triangles();
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
        const JointConfig& jc, const CachedNode& node)
{
    if (jc.type == JointConfig::Type::PRISMATIC) {
        double len = std::sqrt(jc.ax*jc.ax + jc.ay*jc.ay + jc.az*jc.az);
        if (len < 1e-12) return identity4();
        double inv = jc.value / len;
        return makeTrans(jc.ax * inv, jc.ay * inv, jc.az * inv);
    }

    // REVOLUTE - determine pivot (auto pivot: vertex mean, computed at load)
    double cx = jc.px, cy = jc.py, cz = jc.pz;
    if (jc.use_auto_pivot && !node.vertices_local.empty()) {
        cx = node.vertex_mean[0]; cy = node.vertex_mean[1]; cz = node.vertex_mean[2];
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
    immutable_dirty_ = true;   // new file: optics/names must be refreshed on rebuild
    has_mass_ = false;

    HighFive::File file(file_path, HighFive::File::ReadOnly);

    auto objects = file.listObjectNames();
    if (objects.empty())
        throw std::runtime_error("HDF5 file has no root group: " + file_path);

    // Optional mass properties written by SatForm3D (tensor is about the CoM).
    if (file.exist("inertia")) {
        try {
            auto ig = file.getGroup("inertia");
            std::vector<double> com;
            std::vector<std::vector<double>> I;
            ig.getAttribute("total_mass").read(mass_);
            ig.getDataSet("center_of_mass").read(com);
            ig.getDataSet("inertia_tensor").read(I);
            if (com.size() != 3 || I.size() != 3 || I[0].size() != 3 ||
                I[1].size() != 3 || I[2].size() != 3)
                throw std::runtime_error("unexpected dataset shape");
            for (int k = 0; k < 3; ++k) com_[k] = com[k];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) inertia_[r * 3 + c] = I[r][c];
            has_mass_ = true;
        } catch (const std::exception& e) {
            has_mass_ = false;
            std::cerr << "SatelliteDataset: /inertia group in " << file_path
                      << " ignored (" << e.what() << ")\n";
        }
    }

    // The root is the group that carries joint_type, like every component. A file
    // without one takes the first group that is not one of the additive tables
    // of the format (mass properties, SRP table, symmetry, shared meshes, origin).
    auto reserved = [](const std::string& n) {
        return n == "inertia" || n == "lumped_masses" || n == "provenance" ||
               n == "prototypes" || n == "srp" || n == "symmetry";
    };
    std::string root_name;
    for (const auto& name : objects) {
        if (reserved(name) || file.getObjectType(name) != HighFive::ObjectType::Group) continue;
        if (root_name.empty()) root_name = name;
        if (file.getGroup(name).hasAttribute("joint_type")) {
            root_name = name;
            break;
        }
    }
    if (root_name.empty())
        throw std::runtime_error("HDF5 file has no geometry root group: " + file_path);
    auto root_group = file.getGroup(root_name);

    auto read_mesh = [](const HighFive::Group& mg, CachedNode& node) {
        node.has_mesh = true;
        mg.getDataSet("vertices").read(node.vertices_local);
        mg.getDataSet("normals").read(node.normals_local);
        mg.getDataSet("triangles").read(node.tri_indices);
        node.areas       = readFlatVector(mg.getDataSet("areas"));
        node.reflectance = readFlatVector(mg.getDataSet("reflectance"));
        node.specularity = readFlatVector(mg.getDataSet("specularity"));
        node.emissivity  = readFlatVector(mg.getDataSet("emissivity"));
    };

    // Shared meshes (schema 2.1 instancing): /prototypes/<g> with attribute
    // "key" and a "mesh" subgroup, referenced by a node's "instance_key".
    std::map<std::string, CachedNode> prototypes;
    if (file.exist("prototypes")) {
        auto pg = file.getGroup("prototypes");
        for (const auto& name : pg.listObjectNames()) {
            auto g = pg.getGroup(name);
            if (!g.hasAttribute("key") || !g.exist("mesh")) continue;
            std::string key;
            g.getAttribute("key").read(key);
            CachedNode proto;
            read_mesh(g.getGroup("mesh"), proto);
            prototypes.emplace(key, std::move(proto));
        }
    }

    std::vector<std::string> paths;          // "/Spacecraft/Bus/..." per cached node
    std::vector<std::string> driver_paths;   // coupling driver, "" if none
    std::function<void(const HighFive::Group&, int, const std::string&, const std::string&)> visit;
    visit = [&](const HighFive::Group& group, int parent_idx, const std::string& gname,
                const std::string& path) {
        CachedNode node;
        node.name = gname;
        node.parent_index = parent_idx;
        node.T_local_offset = identity4();

        if (group.hasAttribute("T_local_offset")) {
            std::vector<double> t_flat;
            group.getAttribute("T_local_offset").read(t_flat);
            if (t_flat.size() != 16)
                throw std::runtime_error("Component '" + gname + "' in " + file_path +
                                         ": T_local_offset must have 16 values");
            for (int i = 0; i < 16; ++i) node.T_local_offset[i] = t_flat[i];
            // A rotation and a translation only: areas come from the file and normals
            // are only turned, so a scale would give each method a different model.
            for (int r = 0; r < 3; ++r)
                for (int c = r; c < 3; ++c) {
                    double d = (r == c) ? -1.0 : 0.0;
                    for (int k = 0; k < 3; ++k) d += t_flat[r * 4 + k] * t_flat[c * 4 + k];
                    if (!(std::fabs(d) < 1e-5))
                        throw std::runtime_error("Component '" + gname + "' in " + file_path +
                                                 ": T_local_offset is not a rotation and a translation");
                }
        }

        if (group.hasAttribute("component_type"))
            group.getAttribute("component_type").read(node.component_type);

        // Declared joint. Anything but "revolute" / "prismatic" is fixed.
        node.joint.name = gname;
        node.joint.type = "fixed";
        if (group.hasAttribute("joint_type")) {
            std::string jt;
            group.getAttribute("joint_type").read(jt);
            if (jt == "revolute" || jt == "prismatic") node.joint.type = jt;
        }
        std::string driver_path;
        if (node.joint.movable()) {
            if (group.hasAttribute("joint_axis")) {
                std::vector<double> ax;
                group.getAttribute("joint_axis").read(ax);
                if (ax.size() == 3) node.joint.axis = {ax[0], ax[1], ax[2]};
            }
            if (group.hasAttribute("joint_min") && group.hasAttribute("joint_max")) {
                group.getAttribute("joint_min").read(node.joint.min_value);
                group.getAttribute("joint_max").read(node.joint.max_value);
                node.joint.has_limits = node.joint.min_value <= node.joint.max_value;
            }
            if (group.hasAttribute("joint_default_value"))
                group.getAttribute("joint_default_value").read(node.joint.default_value);
            // pose stored in the file: joint_value (schema 2.2, written only when it
            // differs from the default), otherwise the default value
            node.joint.file_value = node.joint.default_value;
            if (group.hasAttribute("joint_value"))
                group.getAttribute("joint_value").read(node.joint.file_value);
            if (group.hasAttribute("coupling_driver_path")) {
                group.getAttribute("coupling_driver_path").read(driver_path);
                if (group.hasAttribute("coupling_ratio"))
                    group.getAttribute("coupling_ratio").read(node.joint.ratio);
                if (group.hasAttribute("coupling_offset"))
                    group.getAttribute("coupling_offset").read(node.joint.offset);
            }
            if (group.exist("deploy_states")) {
                auto dg = group.getGroup("deploy_states");
                std::vector<std::string> names;
                std::vector<double> values;
                if (dg.exist("names") && dg.exist("values")) {
                    dg.getDataSet("names").read(names);
                    values = readFlatVector(dg.getDataSet("values"));
                }
                for (size_t k = 0; k < names.size() && k < values.size(); ++k)
                    node.joint.deploy_states[names[k]] = values[k];
            }
        }

        std::string instance_key;
        if (group.hasAttribute("instance_key"))
            group.getAttribute("instance_key").read(instance_key);
        auto proto = instance_key.empty() ? prototypes.end() : prototypes.find(instance_key);
        if (proto != prototypes.end()) {
            // shared mesh with this instance's uniform material
            const CachedNode& pm = proto->second;
            node.has_mesh       = true;
            node.vertices_local = pm.vertices_local;
            node.normals_local  = pm.normals_local;
            node.tri_indices    = pm.tri_indices;
            node.areas          = pm.areas;
            node.reflectance    = pm.reflectance;
            node.specularity    = pm.specularity;
            node.emissivity     = pm.emissivity;
            if (group.hasAttribute("instance_material")) {
                std::vector<double> mat;
                group.getAttribute("instance_material").read(mat);
                if (mat.size() == 3) {
                    const size_t M = node.tri_indices.size();
                    node.reflectance.assign(M, mat[0]);
                    node.specularity.assign(M, mat[1]);
                    node.emissivity.assign(M, mat[2]);
                }
            }
        } else if (group.exist("mesh")) {
            read_mesh(group.getGroup("mesh"), node);
        } else if (!instance_key.empty()) {
            throw std::runtime_error("Component '" + gname + "' in " + file_path + ": instance_key '" +
                                     instance_key + "' is not in /prototypes");
        }
        if (!node.vertices_local.empty()) {
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (const auto& v : node.vertices_local) { cx += v[0]; cy += v[1]; cz += v[2]; }
            double n = static_cast<double>(node.vertices_local.size());
            cx /= n; cy /= n; cz /= n;
            node.vertex_mean = {cx, cy, cz};
        }

        if (node.has_mesh) {
            // reflectance = specularity = 0 is a legal black surface, so a short array
            // must not read as zero; the others would run off the end of the array
            const size_t M = node.tri_indices.size();
            const auto bad = [&](const std::string& what) {
                return std::runtime_error("Component '" + node.name + "' in " + file_path + ": " + what);
            };
            if (node.reflectance.size() < M || node.specularity.size() < M)
                throw bad("reflectance and specularity must have one value per triangle");
            if (node.normals_local.size() < M)
                throw bad("normals must have one row per triangle");
            const int V = static_cast<int>(node.vertices_local.size());
            for (const auto& tri : node.tri_indices)
                for (int k = 0; k < 3; ++k)
                    if (tri[k] < 0 || tri[k] >= V)
                        throw bad("triangle vertex index out of range");
        }
        cached_nodes_.push_back(std::move(node));
        paths.push_back(path);
        driver_paths.push_back(driver_path);
        const int my_idx = static_cast<int>(cached_nodes_.size()) - 1;

        // Every child group but these is a component; without joint_type it is fixed.
        for (const auto& child_name : group.listObjectNames()) {
            if (child_name == "mesh" || child_name == "ports" || child_name == "deploy_states" ||
                group.getObjectType(child_name) != HighFive::ObjectType::Group) continue;
            visit(group.getGroup(child_name), my_idx, child_name, path + "/" + child_name);
        }
    };

    visit(root_group, -1, root_name, "/" + root_name);

    const bool any_triangles = std::any_of(cached_nodes_.begin(), cached_nodes_.end(),
        [](const CachedNode& n) { return !n.tri_indices.empty(); });
    if (!any_triangles)
        throw std::runtime_error("No triangles in " + file_path);

    // Couplings: the driver is an ancestor in the generator's tree, so it is
    // cached before the driven node (computeBodyTransforms relies on that).
    for (size_t i = 0; i < cached_nodes_.size(); ++i) {
        if (driver_paths[i].empty()) continue;
        std::string dp = driver_paths[i];
        if (dp[0] != '/') dp = "/" + dp;
        int d = -1;
        for (size_t k = 0; k < i; ++k)
            if (paths[k] == dp) { d = static_cast<int>(k); break; }
        if (d < 0 || !cached_nodes_[d].joint.movable()) {
            std::cerr << "SatelliteDataset: coupling driver " << driver_paths[i] << " of '"
                      << cached_nodes_[i].name << "' is not a preceding movable component in "
                      << file_path << "; coupling ignored\n";
            continue;
        }
        cached_nodes_[i].driver = d;
        cached_nodes_[i].joint.driver = cached_nodes_[d].name;
    }
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
// The declared joint of a component at value v: type and axis from the file,
// pivot at the origin of the local frame.
JointConfig SatelliteDataset::declaredJoint(const JointInfo& J, double v) {
    JointConfig jc;
    jc.type  = (J.type == "prismatic") ? JointConfig::Type::PRISMATIC : JointConfig::Type::REVOLUTE;
    jc.ax    = J.axis[0];
    jc.ay    = J.axis[1];
    jc.az    = J.axis[2];
    jc.value = v;
    return jc.atOrigin();
}

// Value of a registered JointConfig along the declared axis (sign flipped for
// an antiparallel axis); the value itself for any other axis.
double SatelliteDataset::declaredValue(const JointConfig& jc, const JointInfo& J) {
    const double len  = std::sqrt(jc.ax * jc.ax + jc.ay * jc.ay + jc.az * jc.az);
    const double alen = std::sqrt(J.axis[0] * J.axis[0] + J.axis[1] * J.axis[1] + J.axis[2] * J.axis[2]);
    if (len < 1e-12 || alen < 1e-12) return jc.value;
    const double c = (jc.ax * J.axis[0] + jc.ay * J.axis[1] + jc.az * J.axis[2]) / (len * alen);
    return (c < -1.0 + 1e-9) ? -jc.value : jc.value;
}

std::vector<SatelliteDataset::Mat4> SatelliteDataset::computeBodyTransforms(
        const std::map<std::string, JointConfig>& joints) const
{
    // T_body for each cached node; index aligns with cached_nodes_.
    // q[i]: joint value of node i along its declared axis (drives couplings).
    std::vector<Mat4> body_transforms(cached_nodes_.size(), identity4());
    std::vector<double> q(cached_nodes_.size(), 0.0);
    for (size_t i = 0; i < cached_nodes_.size(); ++i) {
        const CachedNode& node = cached_nodes_[i];
        const JointInfo& J = node.joint;

        // T_local = T_local_offset · T_joint. A movable joint takes, in this
        // order: its driver (q = ratio * q_driver + offset, clipped), a joint
        // registered with setJoint(), or the value of the pose stored in the
        // file. A zero value is the identity and is not multiplied in, so files
        // whose joints sit at zero give the transforms of 1.1.0 bit for bit.
        Mat4 T_local = node.T_local_offset;
        auto it = joints.find(node.name);
        if (node.parent_index >= 0 && J.movable() && node.driver >= 0) {
            double v = J.ratio * q[node.driver] + J.offset;
            if (J.has_limits) v = std::min(std::max(v, J.min_value), J.max_value);
            q[i] = v;
            if (v != 0.0) T_local = mul4(T_local, buildJointTransform(declaredJoint(J, v), node));
        } else if (it != joints.end()) {
            Mat4 T_joint = buildJointTransform(it->second, node);
            T_local = mul4(T_local, T_joint);
            q[i] = declaredValue(it->second, J);
        } else if (node.parent_index >= 0 && J.movable()) {
            q[i] = J.file_value;
            if (J.file_value != 0.0)
                T_local = mul4(T_local, buildJointTransform(declaredJoint(J, J.file_value), node));
        }

        // T_body = T_body(parent) · T_local
        body_transforms[i] = (node.parent_index < 0)
            ? T_local
            : mul4(body_transforms[node.parent_index], T_local);
    }
    return body_transforms;
}

void SatelliteDataset::rebuildTriangles(
        const std::map<std::string, JointConfig>& joint_overrides) const
{
    if (cached_nodes_.empty()) {
        triangles_.clear();
        return;
    }

    // Pre-count total triangle count.
    size_t total_tris = 0;
    for (const auto& n : cached_nodes_) if (n.has_mesh) total_tris += n.tri_indices.size();

    // Detect first build vs subsequent articulation rebuild.
    // On first build after reading a file (immutable_dirty_) or when the
    // triangle count changes we must populate the immutable per-triangle
    // fields: ID, component_type, component_name, optical properties, area.
    // The count alone is not enough: another file may have the same number of
    // triangles and different optics. On articulation rebuilds the immutable
    // fields are already correct, so we only refresh geometry (vertices,
    // normal, centroid). This avoids ~3*N std::string allocations per
    // articulation - the dominant cost for a 750k-triangle mesh.
    const bool first_build = immutable_dirty_ || (triangles_.size() != total_tris);
    triangles_.resize(total_tris);

    const std::vector<Mat4> body_transforms = computeBodyTransforms(joint_overrides);

    size_t write_idx = 0;
    for (size_t i = 0; i < cached_nodes_.size(); ++i) {
        const CachedNode& node = cached_nodes_[i];
        const Mat4& T_body = body_transforms[i];

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
                tri.reflectance    = node.reflectance[t];
                tri.specularity    = node.specularity[t];
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

    immutable_dirty_ = false;
}

// =========================================================================
// Rigid parts for the two-level GPU acceleration structures (instance_scene.h)
//
// A part is the fixed body or a component declared movable in the file
// and currently out of its file pose (see commitJoints), together with its
// fixed descendants. Vertices are stored in the part frame:
// for the body that is the body frame (same arithmetic as rebuildTriangles,
// so bit-identical), for a movable part the frame of its movable node.
// =========================================================================
static std::array<double, 12> rows3x4(const std::array<double, 16>& M) {
    std::array<double, 12> r{};
    for (int k = 0; k < 12; ++k) r[k] = M[k];
    return r;
}

void SatelliteDataset::buildSceneStructure() {
    InstanceScene& s = *scene_;
    s = InstanceScene{};
    s.structure_version = structure_version_;

    const size_t nn = cached_nodes_.size();
    std::vector<int>  owner(nn, -1);         // movable node owning each node, -1 = body
    std::vector<Mat4> T_part(nn, identity4());
    for (size_t i = 0; i < nn; ++i) {
        const CachedNode& node = cached_nodes_[i];
        const int p = node.parent_index;
        if (p >= 0 && node.joint.movable() && part_active_[i]) {
            owner[i] = static_cast<int>(i);  // own frame
            continue;
        }
        owner[i] = (p < 0) ? -1 : owner[p];
        // body: every joint on the way sits at its file pose, so T_body is the
        // file-pose transform (bit-identical to rebuildTriangles); inside a moving
        // part only fixed components remain (see commitJoints)
        T_part[i] = (owner[i] < 0) ? file_pose_T_[i] : mul4(T_part[p], node.T_local_offset);
    }

    size_t total = 0;
    for (const auto& n : cached_nodes_) if (n.has_mesh) total += n.tri_indices.size();
    s.N = total;
    s.tri_instance.resize(total);
    s.part_vertices.resize(9 * total);
    s.part_normals.resize(3 * total);

    std::map<int, std::uint32_t> part_of_owner;
    std::uint32_t g = 0;
    for (size_t i = 0; i < nn; ++i) {
        const CachedNode& node = cached_nodes_[i];
        if (!node.has_mesh || node.tri_indices.empty()) continue;
        auto it = part_of_owner.find(owner[i]);
        if (it == part_of_owner.end()) {
            SceneInstance part;
            part.node = owner[i];
            part.T = {1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0};
            s.instances.push_back(std::move(part));
            it = part_of_owner.emplace(owner[i], (std::uint32_t)(s.instances.size() - 1)).first;
        }
        SceneInstance& part = s.instances[it->second];
        const std::uint32_t M = static_cast<std::uint32_t>(node.tri_indices.size());
        if (!part.first.empty() && part.first.back() + part.count.back() == g) part.count.back() += M;
        else { part.first.push_back(g); part.count.push_back(M); }
        part.size += M;

        const Mat4& T = T_part[i];
        for (std::uint32_t t = 0; t < M; ++t, ++g) {
            s.tri_instance[g] = it->second;
            for (int k = 0; k < 3; ++k) {
                const auto& v = node.vertices_local[node.tri_indices[t][k]];
                double* o = &s.part_vertices[9 * (size_t)g + 3 * k];
                transformPoint(T, v[0], v[1], v[2], o[0], o[1], o[2]);
                if (part.node >= 0) part.node_vertices.insert(part.node_vertices.end(), {v[0], v[1], v[2]});
            }
            const auto& nl = node.normals_local[t];
            double* on = &s.part_normals[3 * (size_t)g];
            transformNormal(T, nl[0], nl[1], nl[2], on[0], on[1], on[2]);
            if (part.node >= 0) {
                part.node_normals.insert(part.node_normals.end(), {nl[0], nl[1], nl[2]});
                part.tri_node.push_back(static_cast<std::uint32_t>(i));
            }
        }
    }
    s.node_T.assign(nn, rows3x4(identity4()));
    scene_pose_version_ = 0;   // transforms are filled by refreshScenePose()
}

void SatelliteDataset::refreshScenePose() const {
    InstanceScene& s = *scene_;
    if (scene_pose_version_ == geometry_version_ && s.structure_version == structure_version_) return;
    const std::vector<Mat4> Tb = computeBodyTransforms(committed_joints_);
    for (size_t i = 0; i < Tb.size() && i < s.node_T.size(); ++i) s.node_T[i] = rows3x4(Tb[i]);
    for (auto& part : s.instances)
        if (part.node >= 0) part.T = s.node_T[part.node];
    s.pose_version = geometry_version_;
    scene_pose_version_ = geometry_version_;
}

const InstanceScene& SatelliteDataset::instanceScene() const {
    refreshScenePose();
    return *scene_;
}

std::vector<SatelliteDataset::MeshMotion> SatelliteDataset::meshMotions() const {
    std::vector<MeshMotion> out;
    if (cached_nodes_.empty()) return out;
    const std::vector<Mat4> Tb = computeBodyTransforms(committed_joints_);
    int first = 0;
    for (size_t i = 0; i < cached_nodes_.size(); ++i) {
        const CachedNode& node = cached_nodes_[i];
        if (!node.has_mesh) continue;
        MeshMotion m;
        m.name           = node.name;
        m.first_triangle = first;
        m.triangle_count = static_cast<int>(node.tri_indices.size());
        first += m.triangle_count;
        const Mat4& F = file_pose_T_[i];
        if (std::memcmp(Tb[i].data(), F.data(), sizeof(Mat4)) == 0) {
            m.motion = { 1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0 };
        } else {
            // motion = T_body * F^-1, F affine (inverse of its 3x3 block by cofactors)
            const double a = F[0], b = F[1], c = F[2], d = F[4], e = F[5], f = F[6],
                         g = F[8], h = F[9], k = F[10];
            const double det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
            const double Ai[9] = { (e * k - f * h) / det, (c * h - b * k) / det, (b * f - c * e) / det,
                                   (f * g - d * k) / det, (a * k - c * g) / det, (c * d - a * f) / det,
                                   (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det };
            Mat4 Fi = identity4();
            for (int r = 0; r < 3; ++r) {
                for (int col = 0; col < 3; ++col) Fi[r * 4 + col] = Ai[r * 3 + col];
                Fi[r * 4 + 3] = -(Ai[r * 3] * F[3] + Ai[r * 3 + 1] * F[7] + Ai[r * 3 + 2] * F[11]);
            }
            const Mat4 M = mul4(Tb[i], Fi);
            for (int r = 0; r < 3; ++r)
                for (int col = 0; col < 4; ++col) m.motion[r * 4 + col] = M[r * 4 + col];
        }
        out.push_back(std::move(m));
    }
    return out;
}
