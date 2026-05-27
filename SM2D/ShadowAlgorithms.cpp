#include "ShadowAlgorithms.h"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <vector>
#include <array>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <filesystem>

namespace py = pybind11;
namespace fs = std::filesystem;


// Solar radiation pressure scale factor - shared global, configurable from menu.
double g_srp_phi0 = 1.0;        // dimensionless (set to 4.56e-6 N/m² for physical SI output)

// ---------------------------------------------------------------------------
// Solar Radiation Pressure (SRP) force/moment computation
// Uses per-triangle optical properties from HDF5 (reflectance = rho, specularity = s).
// Falls back to alpha=0.5, mu=0.5 for legacy CSV data (where reflectance=specularity=0).
// ---------------------------------------------------------------------------
SRPResult compute_srp_forces(
    const std::vector<Triangle>& triangles,
    std::vector<int> labels,
    double sun_x, double sun_y, double sun_z)
{
    const double phi0  = g_srp_phi0;

    std::array<double, 3> total_force  = {0.0, 0.0, 0.0};
    std::array<double, 3> total_moment = {0.0, 0.0, 0.0};

    size_t N = triangles.size();
    for (size_t i = 0; i < N; ++i) {
        if (labels[i] == 0) continue;  // shadowed polygon contributes nothing

        const Triangle& t = triangles[i];
        double nx = t.normal_x, ny = t.normal_y, nz = t.normal_z;
        double cos_theta = nx * sun_x + ny * sun_y + nz * sun_z;
        if (cos_theta <= 0.0) continue;  // back-facing (safety check)

        // Per-triangle optical properties (fall back to 0.5/0.5 for legacy CSV)
        double alpha = (t.reflectance > 0.0 || t.specularity > 0.0) ? t.reflectance : 0.5;
        double mu    = (t.reflectance > 0.0 || t.specularity > 0.0) ? t.specularity : 0.5;

        // Triangle area - use precomputed if available, else compute from vertices
        double A = t.area;
        if (A <= 0.0) {
            double e1x = t.v2_x - t.v1_x, e1y = t.v2_y - t.v1_y, e1z = t.v2_z - t.v1_z;
            double e2x = t.v3_x - t.v1_x, e2y = t.v3_y - t.v1_y, e2z = t.v3_z - t.v1_z;
            double crx = e1y*e2z - e1z*e2y;
            double cry = e1z*e2x - e1x*e2z;
            double crz = e1x*e2y - e1y*e2x;
            A = 0.5 * std::sqrt(crx*crx + cry*cry + crz*crz);
        }

        // Triangle centroid - use precomputed if available
        double cx = (t.centroid_x != 0.0 || t.centroid_y != 0.0 || t.centroid_z != 0.0)
            ? t.centroid_x : (t.v1_x + t.v2_x + t.v3_x) / 3.0;
        double cy = (t.centroid_x != 0.0 || t.centroid_y != 0.0 || t.centroid_z != 0.0)
            ? t.centroid_y : (t.v1_y + t.v2_y + t.v3_y) / 3.0;
        double cz = (t.centroid_x != 0.0 || t.centroid_y != 0.0 || t.centroid_z != 0.0)
            ? t.centroid_z : (t.v1_z + t.v2_z + t.v3_z) / 3.0;

        // Force components (see formulas):
        //   Fabs  = Phi0 * A * (1-alpha) * cos_theta * (-s)
        //   Fspec = Phi0 * A * 2*alpha*mu * cos^2_theta * (-n)
        //   Fdiff = Phi0 * A * alpha*(1-mu) * cos_theta * (-s - 2/3 * n)
        //
        // Collecting s and n terms:
        //   s_coeff = -Phi0 * A * cos_theta * ((1-alpha) + alpha*(1-mu))
        //   n_coeff = -Phi0 * A * (2*alpha*mu*cos_theta + 2/3*alpha*(1-mu)) * cos_theta
        double base = phi0 * A * cos_theta;                                                           //PHY
        double s_coeff = -base * ((1.0 - alpha) + alpha * (1.0 - mu));                               //PHY
        double n_coeff = -base * (2.0 * alpha * mu * cos_theta + (2.0/3.0) * alpha * (1.0 - mu));   //PHY

        double fx = s_coeff * sun_x + n_coeff * nx;   //PHY
        double fy = s_coeff * sun_y + n_coeff * ny;   //PHY
        double fz = s_coeff * sun_z + n_coeff * nz;   //PHY

        total_force[0] += fx;   //PHY
        total_force[1] += fy;   //PHY
        total_force[2] += fz;   //PHY

        // Moment contribution: r_i x F_i
        total_moment[0] += cy * fz - cz * fy;   //PHY
        total_moment[1] += cz * fx - cx * fz;   //PHY
        total_moment[2] += cx * fy - cy * fx;   //PHY
    }

    return {std::move(labels), total_force, total_moment};
}


void visualize_triangles(const std::string& csv_file, int every, bool show_normals,
    std::array<double, 3> force, std::array<double, 3> moment) {
    try {
        py::module_ visualize = py::module_::import("visualize3d");
        visualize.attr("visualize_triangles")(csv_file, every, show_normals, force, moment);
    }
    catch (const py::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << "\n";
        throw;
    }
}

void save_results(const std::vector<Triangle>& triangles, const std::vector<int>& labels, const std::vector<int>& reference_labels, const std::string& filename, bool pixel_grid_mode) {
    std::filesystem::create_directory("results_3d");
    std::string full_path = "results_3d/" + filename;

    std::ofstream file(full_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + full_path + " for writing");
    }

    file << "Triangle ID,Component Type,V1_X,V1_Y,V1_Z,V2_X,V2_Y,V2_Z,V3_X,V3_Y,V3_Z,Normal_X,Normal_Y,Normal_Z,Area,Reflectance,Specularity,Emissivity,Label,Correct\n";

    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& t = triangles[i];
        // pixel_grid_mode: Correct=0 only when the grid completely missed a polygon
        // that the reference marks as lit (false negative). A polygon hit by at least
        // one grid ray (label=1) is always shown yellow, even if the reference says 0.
        int correct;
        if (pixel_grid_mode)
            correct = (labels[i] == 0 && reference_labels[i] == 1) ? 0 : 1;
        else
            correct = (labels[i] == reference_labels[i]) ? 1 : 0;
        file << i << "," << t.component_type << ","
            << t.v1_x << "," << t.v1_y << "," << t.v1_z << ","
            << t.v2_x << "," << t.v2_y << "," << t.v2_z << ","
            << t.v3_x << "," << t.v3_y << "," << t.v3_z << ","
            << t.normal_x << "," << t.normal_y << "," << t.normal_z << ","
            << t.area << "," << t.reflectance << "," << t.specularity << "," << t.emissivity << ","
            << labels[i] << "," << correct << "\n";
    }

    file.close();
}

void save_results_with_bounces(
    const std::vector<Triangle>& triangles,
    const std::vector<int>& labels,
    const std::vector<int>& reference_labels,
    const std::vector<int>& bounce_levels,
    const std::vector<std::array<double,3>>& incident_dirs,
    const std::vector<std::array<double,3>>& origin_pts,
    const std::string& filename)
{
    std::filesystem::create_directory("results_3d");
    std::string full_path = "results_3d/" + filename;

    std::ofstream file(full_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + full_path + " for writing");
    }

    // Header: standard columns + optical + bounce level + incident direction + ray origin
    file << "Triangle ID,Component Type,"
         << "V1_X,V1_Y,V1_Z,V2_X,V2_Y,V2_Z,V3_X,V3_Y,V3_Z,"
         << "Normal_X,Normal_Y,Normal_Z,"
         << "Area,Reflectance,Specularity,Emissivity,"
         << "Label,Correct,ReflectionBounce,"
         << "Incident_X,Incident_Y,Incident_Z,"
         << "Origin_X,Origin_Y,Origin_Z\n";

    const bool have_dirs   = (incident_dirs.size() == triangles.size());
    const bool have_origin = (origin_pts.size()    == triangles.size());
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& t = triangles[i];
        int correct = (labels[i] == reference_labels[i]) ? 1 : 0;

        double ix = 0.0, iy = 0.0, iz = 0.0;
        if (have_dirs) {
            ix = incident_dirs[i][0];
            iy = incident_dirs[i][1];
            iz = incident_dirs[i][2];
        }

        double ox = NaN, oy = NaN, oz = NaN;
        if (have_origin) {
            ox = origin_pts[i][0];
            oy = origin_pts[i][1];
            oz = origin_pts[i][2];
        }

        file << i << "," << t.component_type << ","
            << t.v1_x << "," << t.v1_y << "," << t.v1_z << ","
            << t.v2_x << "," << t.v2_y << "," << t.v2_z << ","
            << t.v3_x << "," << t.v3_y << "," << t.v3_z << ","
            << t.normal_x << "," << t.normal_y << "," << t.normal_z << ","
            << t.area << "," << t.reflectance << "," << t.specularity << "," << t.emissivity << ","
            << labels[i] << "," << correct << ","
            << bounce_levels[i] << ","
            << ix << "," << iy << "," << iz << ","
            << ox << "," << oy << "," << oz << "\n";
    }

    file.close();
}
