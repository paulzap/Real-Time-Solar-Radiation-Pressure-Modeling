cmake_minimum_required(VERSION 3.23)
project(SRPLibrary LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Build options — all default ON, disabled automatically if tools not found
# ---------------------------------------------------------------------------
option(SM3D_ENABLE_CUDA   "Enable CUDA acceleration"           ON)
option(SM3D_ENABLE_OPTIX  "Enable OptiX RT-Core methods"       ON)
option(SM3D_ENABLE_PYTHON "Enable Python visualisation"        ON)
option(SM3D_BUILD_DEMO    "Build demo executable (main.cpp)"   ON)

# ---------------------------------------------------------------------------
# Required dependencies (always needed)
# ---------------------------------------------------------------------------
find_package(HDF5 REQUIRED)
find_package(HighFive REQUIRED CONFIG)

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
if(SM3D_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(SM3D_HAS_CUDA TRUE)
        message(STATUS "SM3D: CUDA ${CUDAToolkit_VERSION} found — CentroidGPU/PixelGridGPU enabled")
    else()
        message(STATUS "SM3D: CUDA not found — GPU methods disabled, CPU-only build")
    endif()
else()
    message(STATUS "SM3D: CUDA disabled by SM3D_ENABLE_CUDA=OFF")
endif()

# ---------------------------------------------------------------------------
# OptiX (requires CUDA)
# ---------------------------------------------------------------------------
if(SM3D_HAS_CUDA AND SM3D_ENABLE_OPTIX)
    find_path(OPTIX_INCLUDE_DIR optix.h
        PATHS
            # env var override (cross-platform): set OPTIX_INSTALL_DIR=/path/to/optix
            "$ENV{OPTIX_INSTALL_DIR}/include"
            # Windows default install paths
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/include"
            # Linux default install paths
            "/opt/optix/include"
            "/usr/local/optix/include"
            "$ENV{HOME}/optix/include"
    )
    if(OPTIX_INCLUDE_DIR)
        set(SM3D_HAS_OPTIX TRUE)
        message(STATUS "SM3D: OptiX found at ${OPTIX_INCLUDE_DIR} — RTX methods enabled")
    else()
        message(STATUS "SM3D: OptiX not found — RTX methods disabled. Set OPTIX_INSTALL_DIR env var or install OptiX SDK.")
    endif()
elseif(SM3D_ENABLE_OPTIX AND NOT SM3D_HAS_CUDA)
    message(STATUS "SM3D: OptiX requires CUDA — RTX methods disabled")
endif()

# ---------------------------------------------------------------------------
# Python / pybind11 (for visualiseLastResult only)
# ---------------------------------------------------------------------------
if(SM3D_ENABLE_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        set(SM3D_HAS_PYTHON TRUE)
        message(STATUS "SM3D: pybind11 found — Python visualisation enabled")
    else()
        message(STATUS "SM3D: pybind11 not found — visualiseLastResult() will print a warning at runtime")
    endif()
endif()

# ---------------------------------------------------------------------------
# Source groups
# ---------------------------------------------------------------------------
set(CPU_SRC
    SM2D/SRPLibrary.cpp
    SM2D/SatelliteDataset.cpp
    SM2D/ShadowAlgorithms.cpp
    SM2D/ReflectionRayCasting.cpp
    SM2D/PixelGridRayCasting.cpp
    SM2D/SM3D_hw_info.cpp
)

set(CUDA_SRC
    SM2D/gpu_reflection_raytracer.cu
    SM2D/gpu_reflection_raytracer_bench.cu
    SM2D/gpu_pixel_grid_raytracer.cu
    SM2D/gpu_pixel_grid_raytracer_bench.cu
)

set(OPTIX_SRC
    SM2D/gpu_optix_shared.cu
    SM2D/gpu_optix_center_raytracer.cu
    SM2D/gpu_optix_center_raytracer_bench.cu
    SM2D/gpu_optix_pixel_grid_raytracer.cu
    SM2D/gpu_optix_pixel_grid_raytracer_bench.cu
)

# OptiX PTX shaders — compiled to .ptx via custom commands, NOT linked into the .lib
set(PTX_SRC
    SM2D/optix_shaders_center.cu
    SM2D/optix_shaders_center_bench.cu
    SM2D/optix_shaders_pixel_grid.cu
    SM2D/optix_shaders_pixel_grid_bench.cu
)

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
add_library(srp STATIC ${CPU_SRC})
target_include_directories(srp PUBLIC SM2D)

# HDF5 + HighFive
target_link_libraries(srp PUBLIC HighFive::HighFive)
if(TARGET HDF5::HDF5)
    target_link_libraries(srp PUBLIC HDF5::HDF5)
else()
    # Fallback for older HDF5 CMake modules
    target_include_directories(srp PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(srp PUBLIC ${HDF5_LIBRARIES})
endif()

# CUDA
if(SM3D_HAS_CUDA)
    target_sources(srp PRIVATE ${CUDA_SRC})
    target_compile_definitions(srp PUBLIC SM3D_HAS_CUDA)
    target_link_libraries(srp PRIVATE CUDA::cudart CUDA::nvml)
    set_target_properties(srp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(srp PROPERTIES CUDA_ARCHITECTURES "75;86;89")
    endif()
endif()

# OptiX
if(SM3D_HAS_OPTIX)
    target_sources(srp PRIVATE ${OPTIX_SRC})
    target_compile_definitions(srp PUBLIC SM3D_HAS_OPTIX)
    target_include_directories(srp PRIVATE ${OPTIX_INCLUDE_DIR})

    # Compile PTX shaders — output goes to build directory alongside the demo exe
    foreach(SRC IN LISTS PTX_SRC)
        get_filename_component(NAME ${SRC} NAME_WE)
        set(PTX_OUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}.ptx")
        if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
            set(PTX_OUT "${CMAKE_BINARY_DIR}/${NAME}.ptx")
        endif()
        add_custom_command(
            OUTPUT  "${PTX_OUT}"
            COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
                    -ptx -arch=compute_75
                    "-I${OPTIX_INCLUDE_DIR}"
                    "${CMAKE_SOURCE_DIR}/${SRC}"
                    -o "${PTX_OUT}"
            DEPENDS "${CMAKE_SOURCE_DIR}/${SRC}"
            COMMENT "Compiling OptiX PTX: ${NAME}.ptx"
        )
        list(APPEND PTX_FILES "${PTX_OUT}")
    endforeach()
    add_custom_target(srp_ptx ALL DEPENDS ${PTX_FILES})
    add_dependencies(srp srp_ptx)
endif()

# Python
if(SM3D_HAS_PYTHON)
    target_compile_definitions(srp PUBLIC SM3D_HAS_PYTHON)
    target_link_libraries(srp PRIVATE pybind11::embed)
endif()

# ---------------------------------------------------------------------------
# Demo executable
# ---------------------------------------------------------------------------
if(SM3D_BUILD_DEMO)
    add_executable(srp_demo SM2D/main.cpp)
    target_link_libraries(srp_demo PRIVATE srp)
    # PTX files must be next to the exe — copy them to exe output dir
    if(SM3D_HAS_OPTIX)
        add_dependencies(srp_demo srp_ptx)
    endif()
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== SM3D SRP Library build configuration ===")
message(STATUS "  CUDA   : ${SM3D_HAS_CUDA}")
message(STATUS "  OptiX  : ${SM3D_HAS_OPTIX}")
message(STATUS "  Python : ${SM3D_HAS_PYTHON}")
message(STATUS "  Demo   : ${SM3D_BUILD_DEMO}")
message(STATUS "  Available methods:")
message(STATUS "    CentroidCPU  PixelGridCPU  (always)")
if(SM3D_HAS_CUDA)
    message(STATUS "    CentroidGPU  PixelGridGPU  (CUDA)")
endif()
if(SM3D_HAS_OPTIX)
    message(STATUS "    CentroidRTX  PixelGridRTX  (OptiX)")
endif()
message(STATUS "============================================")
message(STATUS "")
