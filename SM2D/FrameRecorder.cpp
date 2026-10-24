// FrameRecorder - writes computed frames for an animation (see SRPLibrary.h;
// file format in docs/USAGE.md, player SM2D/animate_srp.py).

#include "SRPLibrary.h"
#include "SatelliteDataset.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>

namespace {

constexpr int kFormatVersion = 1;

std::string json_string(const std::string& s)
{
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        }
        else out += static_cast<char>(c);
    }
    return out + "\"";
}

template <class T>
void put(std::ofstream& f, const T* data, size_t count)
{
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
}

} // namespace

struct FrameRecorder::Impl {
    std::filesystem::path dir;
    std::ofstream frames;
    std::string model;
    std::vector<SatelliteDataset::MeshMotion> meshes;   // names and ranges of the first frame
    int  triangles = 0;
    int  frame_count = 0;
    bool finished = false;
    std::vector<std::string> notes;

    void writeGeometry(const SatelliteDataset& ds)
    {
        // Triangles at the pose stored in the file: the current vertices moved
        // back by the inverse motion of their mesh.
        const std::vector<Triangle>& tris = ds.triangles();
        std::vector<float> v(9 * tris.size());
        for (const auto& m : meshes) {
            const auto& M = m.motion;
            const double a = M[0], b = M[1], c = M[2], d = M[4], e = M[5], f = M[6],
                         g = M[8], h = M[9], k = M[10];
            const double det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
            const double Ai[9] = { (e * k - f * h) / det, (c * h - b * k) / det, (b * f - c * e) / det,
                                   (f * g - d * k) / det, (a * k - c * g) / det, (c * d - a * f) / det,
                                   (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det };
            for (int t = m.first_triangle; t < m.first_triangle + m.triangle_count; ++t) {
                const Triangle& T = tris[t];
                const double p[9] = { T.v1_x, T.v1_y, T.v1_z, T.v2_x, T.v2_y, T.v2_z,
                                      T.v3_x, T.v3_y, T.v3_z };
                for (int j = 0; j < 3; ++j) {
                    const double x = p[3 * j] - M[3], y = p[3 * j + 1] - M[7], z = p[3 * j + 2] - M[11];
                    for (int r = 0; r < 3; ++r)
                        v[9 * t + 3 * j + r] = static_cast<float>(Ai[3 * r] * x + Ai[3 * r + 1] * y + Ai[3 * r + 2] * z);
                }
            }
        }
        std::ofstream g(dir / "geometry.bin", std::ios::binary | std::ios::trunc);
        if (!g) throw std::runtime_error("FrameRecorder: cannot write " + (dir / "geometry.bin").string());
        put(g, v.data(), v.size());
    }

    void writeScene() const
    {
        std::ofstream s(dir / "scene.json", std::ios::trunc);
        if (!s) throw std::runtime_error("FrameRecorder: cannot write " + (dir / "scene.json").string());
        const size_t record = 19 * sizeof(double) + meshes.size() * 12 * sizeof(float) + triangles;
        s << "{\n"
          << "  \"format\": \"SolaRT frames\",\n"
          << "  \"format_version\": " << kFormatVersion << ",\n"
          << "  \"solart_version\": \"" << SOLART_VERSION_STRING << "\",\n"
          << "  \"model\": " << json_string(model) << ",\n"
          << "  \"triangles\": " << triangles << ",\n"
          << "  \"frames\": " << frame_count << ",\n"
          << "  \"record_bytes\": " << record << ",\n"
          << "  \"record\": \"time, sun[3], force[3], moment[3], body_to_view[9] as float64; "
             "motion[meshes][12] as float32; label[triangles] as uint8\",\n"
          << "  \"labels\": \"1 = illuminated, 0 = shadowed\",\n"
          << "  \"meshes\": [\n";
        for (size_t i = 0; i < meshes.size(); ++i)
            s << "    {\"name\": " << json_string(meshes[i].name) << ", \"first\": "
              << meshes[i].first_triangle << ", \"count\": " << meshes[i].triangle_count << "}"
              << (i + 1 < meshes.size() ? ",\n" : "\n");
        s << "  ],\n  \"notes\": [";
        for (size_t i = 0; i < notes.size(); ++i)
            s << (i ? ", " : "") << json_string(notes[i]);
        s << "]\n}\n";
    }
};

FrameRecorder::FrameRecorder(const std::string& out_dir) : pImpl(std::make_unique<Impl>())
{
    pImpl->dir = std::filesystem::path(out_dir);
    std::filesystem::create_directories(pImpl->dir);
    pImpl->frames.open(pImpl->dir / "frames.bin", std::ios::binary | std::ios::trunc);
    if (!pImpl->frames)
        throw std::runtime_error("FrameRecorder: cannot write " + (pImpl->dir / "frames.bin").string());
}

FrameRecorder::~FrameRecorder()
{
    try { finish(); } catch (...) {}
}

int FrameRecorder::frameCount() const { return pImpl->frame_count; }

void FrameRecorder::addFrame(const SRPEngine& engine, const SRPResult& res, double time,
                             const std::string& note, const std::array<double, 9>& body_to_view)
{
    Impl& R = *pImpl;
    if (R.finished) throw std::runtime_error("FrameRecorder: addFrame() after finish()");
    const SatelliteDataset& ds = engine.dataset();
    std::vector<SatelliteDataset::MeshMotion> motions = ds.meshMotions();

    if (R.frame_count == 0) {
        R.model     = ds.currentFile();
        R.triangles = ds.triangleCount();
        R.meshes    = motions;
        R.writeGeometry(ds);
    } else if (ds.currentFile() != R.model || ds.triangleCount() != R.triangles ||
               motions.size() != R.meshes.size()) {
        throw std::runtime_error("FrameRecorder: the model changed during the recording");
    }
    if (res.labels.empty())
        throw std::runtime_error("FrameRecorder: the result has no labels (compute() on the GPU "
                                 "does not return them); record the result of computeViz()");
    if (static_cast<int>(res.labels.size()) != R.triangles)
        throw std::runtime_error("FrameRecorder: the result does not belong to the loaded model");

    const std::array<double, 3> sun = engine.getSunDirection();
    double head[19] = { time, sun[0], sun[1], sun[2],
                        res.total_force[0], res.total_force[1], res.total_force[2],
                        res.total_moment[0], res.total_moment[1], res.total_moment[2] };
    for (int i = 0; i < 9; ++i) head[10 + i] = body_to_view[i];
    put(R.frames, head, 19);

    std::vector<float> m(12 * motions.size());
    for (size_t i = 0; i < motions.size(); ++i)
        for (int j = 0; j < 12; ++j) m[12 * i + j] = static_cast<float>(motions[i].motion[j]);
    put(R.frames, m.data(), m.size());

    std::vector<unsigned char> lab(R.triangles, 0);
    for (size_t i = 0; i < res.labels.size(); ++i)
        lab[i] = static_cast<unsigned char>(res.labels[i] < 0 ? 0 : (res.labels[i] > 255 ? 255 : res.labels[i]));
    put(R.frames, lab.data(), lab.size());
    if (!R.frames) throw std::runtime_error("FrameRecorder: write error in frames.bin");

    R.notes.push_back(note);
    ++R.frame_count;
}

void FrameRecorder::finish()
{
    Impl& R = *pImpl;
    if (R.finished) return;
    R.finished = true;
    R.frames.close();
    R.writeScene();
}
