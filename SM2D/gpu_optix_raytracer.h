#pragma once
#include <vector>
#include "SatelliteDataset.h"
#include "ShadowAlgorithms.h"

// Method 7: RTX Ray Casting -- center-method shadow ray per triangle using OptiX RT Cores.
// Physics identical to calculate_labels_ray_casting (CPU) but ~10-100x faster.
SRPResult calculate_labels_ray_casting_rtx(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    int max_reflections,
    const std::string& primary_emitter_name);

// Method 8: RTX Pixel Grid -- parallel grid rays using OptiX RT Cores.
// Physics identical to calculate_labels_pixel_grid_gpu but with hardware BVH traversal.
SRPResult calculate_labels_pixel_grid_rtx(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections,
    bool verbose,
    const std::string& primary_emitter_name);

// Returns true if the OptiX runtime library (nvoptix.dll/liboptix.so) loaded
// successfully on this system. Safe to call from any translation unit.
bool is_optix_runtime_available();
