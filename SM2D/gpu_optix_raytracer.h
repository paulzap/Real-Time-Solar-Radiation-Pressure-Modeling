// SolaRT - solar radiation pressure on GPU RT cores.
// Copyright (c) 2026 Zapevalin P.R. SPDX-License-Identifier: MIT
// If you use this code in published work, please cite:
// https://doi.org/10.1016/j.ast.2026.113714

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
