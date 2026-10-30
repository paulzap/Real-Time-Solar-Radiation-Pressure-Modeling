// SolaRT - solar radiation pressure on GPU RT cores.
// Copyright (c) 2026 Zapevalin P.R. SPDX-License-Identifier: MIT
// If you use this code in published work, please cite:
// https://doi.org/10.1016/j.ast.2026.113714

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

#ifdef SM3D_HAS_CUDA
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

// Two-level variants for files with movable components (gpu_instance_scene.h):
// one BVH per rigid part, a pose change only moves the parts.
SRPResult calculate_labels_reflections_gpu_bench_parts(
    const SatelliteDataset& ds,
    const std::vector<double>& sun_vector,
    int max_reflections,
    bool verbose = false);

SRPResult calculate_labels_pixel_grid_gpu_bench_parts(
    const SatelliteDataset& ds,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections,
    bool verbose = false);
#endif // SM3D_HAS_CUDA

#ifdef SM3D_HAS_OPTIX
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

// Two-level variants for files with movable components: IAS over one GAS per
// rigid part, a pose change rewrites the instance transforms only.
SRPResult calculate_labels_ray_casting_rtx_bench_parts(
    const SatelliteDataset& ds,
    const std::vector<double>& sun_vector,
    int max_reflections);

SRPResult calculate_labels_pixel_grid_rtx_bench_parts(
    const SatelliteDataset& ds,
    const std::vector<double>& sun_vector,
    double grid_step,
    int max_reflections,
    bool verbose = false);
#endif // SM3D_HAS_OPTIX
