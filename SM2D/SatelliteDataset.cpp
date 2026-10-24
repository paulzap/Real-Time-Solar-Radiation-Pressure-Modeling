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
// The cache is invalidated only when the current file changes. Articulation
// (setJoint + reload) only commits the joint values: the triangle list is
// rebuilt when triangles() is next called, and the GPU methods move the rigid
// parts of the model without the list at all.
// =========================================================================
void SatelliteDataset::load(const std::string& file_path) {
    current_file_ = file_path;
    load_h5_into_cache(file_path);   // slow: file read + HDF5 parse, once
    for (const auto& [name, jc] : joints_) validateJoint(name, jc, false);
    structure_version_ = next_geometry_version();
    buildSceneStructure();
    commitJoints(joints_);
    rebuildTriangles(committed_joints_);
    triangles_stale_ = false;
}

void SatelliteDataset::commitJoints(const std::map<std::string, JointConfig>& joints) {
    committed_joints_ = joints;
    geometry_version_ = next_geometry_version();
    triangles_stale_  = true;
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
        if (n.parent_index >= 0 && n.joint.movable()) names.push_back(n.name);
    return names;
}

JointInfo SatelliteDataset::jointInfo(const std::string& name) const {
    for (const auto& n : cached_nodes_)
        if (n.name == name) return n.joint;
    throw std::invalid_argument("SatelliteDataset: no component '" + name + "' in " + current_file_);
}

void SatelliteDataset::setJointValue(const std::string& name, double value) {
    const JointInfo ji = jointInfo(name);
    JointConfig jc;
    jc.type  = (ji.type == "prismatic") ? JointConfig::Type::PRISMATIC : JointConfig::Type::REVOLUTE;
    jc.ax    = ji.axis[0];
    jc.ay    = ji.axis[1];
    jc.az    = ji.axis[2];
    jc.value = value;
    articulate(name, jc.atOrigin());   // validateJoint rejects fixed components
}

void SatelliteDataset::validateJoint(const std::string& name, const JointConfig& jc,
                                     bool unknown_is_error) const {
    bool found = false;
    for (const auto& n : cached_nodes_) {
        if (n.name != name) continue;
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

    current_file_ = file_path;
    load_h5_into_cache(file_path);
    for (const auto& [name, jc] : jc_overrides) validateJoint(name, jc, false);
    structure_version_ = next_geometry_version();
    buildSceneStructure();
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

    // The root is the first group that is not the mass-properties group.
    std::string root_name;
    for (const auto& name : objects)
        if (name != "inertia" && name != "lumped_masses" && name != "provenance") {
            root_name = name;
            break;
        }
    if (root_name.empty())
        throw std::runtime_error("HDF5 file has no geometry root group: " + file_path);
    auto root_group = file.getGroup(root_name);

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

        // Declared joint. Anything but "revolute" / "prismatic" is fixed.
        node.joint.name = gname;
        node.joint.type = "fixed";
        if (group.hasAttribute("joint_type")) {
            std::string jt;
            group.getAttribute("joint_type").read(jt);
            if (jt == "revolute" || jt == "prismatic") node.joint.type = jt;
        }
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
        if (!node.vertices_local.empty()) {
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (const auto& v : node.vertices_local) { cx += v[0]; cy += v[1]; cz += v[2]; }
            double n = static_cast<double>(node.vertices_local.size());
            cx /= n; cy /= n; cz /= n;
            node.vertex_mean = {cx, cy, cz};
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

    visit(root_group, -1, root_name);
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
std::vector<SatelliteDataset::Mat4> SatelliteDataset::computeBodyTransforms(
        const std::map<std::string, JointConfig>& joints) const
{
    // T_body for each cached node; index aligns with cached_nodes_.
    std::vector<Mat4> body_transforms(cached_nodes_.size(), identity4());
    for (size_t i = 0; i < cached_nodes_.size(); ++i) {
        const CachedNode& node = cached_nodes_[i];

        // T_local = T_local_offset · T_joint (if a joint is registered for this node).
        Mat4 T_local = node.T_local_offset;
        auto it = joints.find(node.name);
        if (it != joints.end()) {
            Mat4 T_joint = buildJointTransform(it->second, node);
            T_local = mul4(T_local, T_joint);
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

    immutable_dirty_ = false;
}

// =========================================================================
// Rigid parts for the two-level GPU acceleration structures (instance_scene.h)
//
// A part is the fixed body or a component declared movable in the file
// together with its fixed descendants. Vertices are stored in the part frame:
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
        if (p >= 0 && node.joint.movable()) {
            owner[i] = static_cast<int>(i);  // own frame
            continue;
        }
        owner[i] = (p < 0) ? -1 : owner[p];
        // body: no joints on the way, so this is T_body exactly as in rebuildTriangles
        T_part[i] = (p < 0) ? node.T_local_offset : mul4(T_part[p], node.T_local_offset);
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
