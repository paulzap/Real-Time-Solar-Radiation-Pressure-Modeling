cmake_minimum_required(VERSION 3.23)
project(SRPLibrary LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(SM3D_ENABLE_CUDA   "Enable CUDA acceleration"         ON)
option(SM3D_ENABLE_OPTIX  "Enable OptiX RT-Core methods"     ON)
option(SM3D_ENABLE_PYTHON "Enable Python visualisation"      ON)

# Demo defaults ON only when this is the top-level project.
# When used via add_subdirectory / FetchContent it defaults OFF so it
# does not pollute the parent build.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(SM3D_BUILD_DEMO "Build demo executable (main.cpp)" ON)
else()
    option(SM3D_BUILD_DEMO "Build demo executable (main.cpp)" OFF)
endif()

# ---------------------------------------------------------------------------
# HDF5 — try CONFIG mode first (vcpkg exports hdf5::hdf5 with all
# transitive deps: zlib, szip/libaec). Fall back to module mode for
# system-installed HDF5 (apt/brew).
# ---------------------------------------------------------------------------
find_package(HDF5 CONFIG QUIET)
if(HDF5_FOUND AND (TARGET hdf5::hdf5 OR TARGET HDF5::HDF5))
    message(STATUS "SM3D: HDF5 found via CONFIG mode")
    if(TARGET hdf5::hdf5)
        set(_SM3D_HDF5_TARGET hdf5::hdf5)
    else()
        set(_SM3D_HDF5_TARGET HDF5::HDF5)
    endif()
else()
    find_package(HDF5 REQUIRED COMPONENTS C)
    message(STATUS "SM3D: HDF5 found via module mode")
    set(_SM3D_HDF5_TARGET HDF5::HDF5)
endif()

# ---------------------------------------------------------------------------
# HighFive — header-only C++ wrapper around HDF5.
# Try vcpkg/system first; if not found, download automatically.
# ---------------------------------------------------------------------------
find_package(HighFive CONFIG QUIET)
# Check the target directly — HighFive_FOUND=TRUE is not enough; some system
# packages don't export HighFive::HighFive, and some Ubuntu setups cause
# HighFive's internal find_package(HDF5) to fail silently, leaving no target.
if(NOT TARGET HighFive::HighFive)
    message(STATUS "SM3D: HighFive not found locally, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(HighFive
        GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
        GIT_TAG        v2.9.0
        GIT_SHALLOW    TRUE)
    set(HIGHFIVE_USE_BOOST    OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_BUILD_DOCS   OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_UNIT_TESTS   OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_EXAMPLES     OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(HighFive)

    # Fallback: if HighFive's own CMakeLists.txt still didn't create the target
    # (happens on Ubuntu when its internal HDF5 detection uses non-standard paths),
    # create a minimal interface target directly from the downloaded headers.
    # HighFive is header-only — we only need its include/ directory + HDF5.
    if(NOT TARGET HighFive::HighFive)
        FetchContent_GetProperties(HighFive SOURCE_DIR _hf_src)
        message(STATUS "SM3D: Creating HighFive target manually from ${_hf_src}")
        add_library(_sm3d_HighFive INTERFACE)
        add_library(HighFive::HighFive ALIAS _sm3d_HighFive)
        target_include_directories(_sm3d_HighFive INTERFACE "${_hf_src}/include")
        target_link_libraries(_sm3d_HighFive INTERFACE ${_SM3D_HDF5_TARGET})
    endif()
endif()

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
        message(STATUS "SM3D: CUDA not found — CPU-only build")
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
            "$ENV{OPTIX_INSTALL_DIR}/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/include"
            "/opt/optix/include"
            "/usr/local/optix/include"
            "$ENV{HOME}/optix/include"
    )
    if(OPTIX_INCLUDE_DIR)
        set(SM3D_HAS_OPTIX TRUE)
        message(STATUS "SM3D: OptiX found — RTX methods enabled")
    else()
        message(STATUS "SM3D: OptiX not found — RTX methods disabled (set OPTIX_INSTALL_DIR)")
    endif()
elseif(SM3D_ENABLE_OPTIX AND NOT SM3D_HAS_CUDA)
    message(STATUS "SM3D: OptiX requires CUDA — RTX methods disabled")
endif()

# ---------------------------------------------------------------------------
# Python / pybind11
# ---------------------------------------------------------------------------
if(SM3D_ENABLE_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        set(SM3D_HAS_PYTHON TRUE)
        message(STATUS "SM3D: pybind11 found — Python visualisation enabled")
    else()
        message(STATUS "SM3D: pybind11 not found — visualizeLastResult() will warn at runtime")
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

# Public include dir — works whether this is top-level or a subdirectory
target_include_directories(srp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/SM2D>
    $<INSTALL_INTERFACE:include>
)

# HDF5 + HighFive
target_link_libraries(srp PUBLIC HighFive::HighFive ${_SM3D_HDF5_TARGET})

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

    foreach(SRC IN LISTS PTX_SRC)
        get_filename_component(NAME ${SRC} NAME_WE)
        set(PTX_OUT "${CMAKE_BINARY_DIR}/${NAME}.ptx")
        add_custom_command(
            OUTPUT  "${PTX_OUT}"
            COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
                    -ptx -arch=compute_75
                    "-I${OPTIX_INCLUDE_DIR}"
                    # Use CMAKE_CURRENT_SOURCE_DIR — correct when used as subdirectory
                    "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}"
                    -o "${PTX_OUT}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}"
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
    if(SM3D_HAS_OPTIX)
        add_dependencies(srp_demo srp_ptx)
    endif()
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== SM3D SRP Library ===")
message(STATUS "  CUDA   : ${SM3D_HAS_CUDA}")
message(STATUS "  OptiX  : ${SM3D_HAS_OPTIX}")
message(STATUS "  Python : ${SM3D_HAS_PYTHON}")
message(STATUS "  Demo   : ${SM3D_BUILD_DEMO}")
message(STATUS "  Methods always available: CentroidCPU  PixelGridCPU")
if(SM3D_HAS_CUDA)
    message(STATUS "  Methods with CUDA:        CentroidGPU  PixelGridGPU")
endif()
if(SM3D_HAS_OPTIX)
    message(STATUS "  Methods with OptiX:       CentroidRTX  PixelGridRTX")
endif()
message(STATUS "========================")
message(STATUS "")
