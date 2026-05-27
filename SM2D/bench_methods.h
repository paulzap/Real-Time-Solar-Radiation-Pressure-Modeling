#pragma once
// bench_methods.h
// ============================================================
// Benchmark-only lean function declarations.
// No bounce tracking / visualization overhead.
// Used exclusively by SM3D_benchmark.cpp (menu item 2).
// In all other contexts the standard methods are used.
// ============================================================
#include <vector>
#include <string>
#include "SatelliteDataset.h"
#include "ShadowAlgorithms.h"

// GPU Shadow+Reflections (bench): no bounce arrays; max_reflections==0
// costs the same as the plain GPU shadow method.
SRPResult calculate_labels_reflections_gpu_bench(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    int max_reflections,
    bool verbose = false);

// GPU Pixel Grid (bench): no bounce tracking vectors, no set_bounce_globals.
SRPResult calculate_labels_pixel_grid_gpu_bench(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections,
    bool verbose = false);

// RTX Shadow (bench): no bounce arrays zeroed/downloaded, lean PTX.
SRPResult calculate_labels_ray_casting_rtx_bench(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    int max_reflections);

// RTX Pixel Grid (bench): no bounce arrays, lean PTX, double accumulators.
SRPResult calculate_labels_pixel_grid_rtx_bench(
    const std::vector<Triangle>& triangles,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections,
    bool verbose = false);
