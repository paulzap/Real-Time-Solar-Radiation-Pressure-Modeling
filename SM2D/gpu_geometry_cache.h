#pragma once
// ---------------------------------------------------------------------------
// gpu_geometry_cache.h - internal header.
//
// Every GPU/RTX backend keeps a copy of the triangle list (and its BVH) in
// device memory between calls. Each backend rebuilds that copy by itself only
// when the triangle COUNT changes. A rotated panel or another file with the
// same number of triangles keeps the count, so the copy has to be dropped
// explicitly: the functions below make the next call of the corresponding
// backend rebuild its copy from the current triangle list.
//
// SRPEngine calls all of them whenever SatelliteDataset::geometryVersion()
// differs from the version last sent to the GPU (see SRPLibrary.cpp).
// ---------------------------------------------------------------------------

#ifdef SM3D_HAS_CUDA
void invalidate_refl_gpu_cache();        // gpu_reflection_raytracer.cu       - CentroidGPU, computeViz()
void invalidate_refl_gpu_bench_cache();  // gpu_reflection_raytracer_bench.cu - CentroidGPU, compute()
void invalidate_pg_gpu_cache();          // gpu_pixel_grid_raytracer.cu       - PixelGridGPU, computeViz()
void invalidate_pg_gpu_bench_cache();    // gpu_pixel_grid_raytracer_bench.cu - PixelGridGPU, compute()
#endif

#ifdef SM3D_HAS_OPTIX
void invalidate_rtx_geometry();          // gpu_optix_shared.cu - every RTX method
#endif
