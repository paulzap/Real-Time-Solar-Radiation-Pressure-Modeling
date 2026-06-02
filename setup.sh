#!/usr/bin/env bash
# setup.sh — build the SRP library on Linux / macOS
#
# Usage:
#   ./setup.sh                  # auto-detect everything, build dist/
#   ./setup.sh --vcpkg ~/vcpkg  # override vcpkg path
#   ./setup.sh --optix /opt/optix   # override OptiX path
#   ./setup.sh --force          # rebuild even if dist/ already exists
#   ./setup.sh --help
#
# What this script does:
#   1. Checks cmake, C++ compiler
#   2. Finds or installs vcpkg (for HDF5, HighFive, pybind11)
#   3. Detects CUDA (optional — enables CentroidGPU, PixelGridGPU)
#   4. Detects OptiX (optional — enables CentroidRTX, PixelGridRTX)
#   5. Detects Python + installs visualization packages (optional)
#   6. Runs cmake + build
#   7. Copies results to dist/
#
# macOS note: CUDA and OptiX are not supported on macOS (NVIDIA dropped
# macOS GPU support). Only CentroidCPU and PixelGridCPU will be built.

set -uo pipefail

# ── colours ─────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  GRN='\033[0;32m'; YLW='\033[1;33m'; RED='\033[0;31m'; CYN='\033[0;36m'; NC='\033[0m'
else
  GRN=''; YLW=''; RED=''; NC=''; CYN=''
fi

ok()   { echo -e "  ${GRN}[OK]${NC}  $*"; }
warn() { echo -e "  ${YLW}[!]${NC}   $*"; }
err()  { echo -e "  ${RED}[ERR]${NC} $*"; }
info() { echo -e "        ${CYN}$*${NC}"; }
step() { echo -e "\n==> $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── argument parsing ─────────────────────────────────────────────────────────
OPT_VCPKG=""
OPT_OPTIX=""
OPT_PYTHON=""
OPT_FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vcpkg)   OPT_VCPKG="$2";  shift 2 ;;
    --optix)   OPT_OPTIX="$2";  shift 2 ;;
    --python)  OPT_PYTHON="$2"; shift 2 ;;
    --force)   OPT_FORCE=1;     shift   ;;
    --help|-h)
      sed -n '2,20p' "$0" | sed 's/^# \{0,2\}//'
      exit 0 ;;
    *) echo "Unknown option: $1 (run with --help)"; exit 1 ;;
  esac
done

# ── detect OS ────────────────────────────────────────────────────────────────
OS="linux"
PKG_MGR="none"
if [[ "$(uname -s)" == "Darwin" ]]; then
  OS="macos"
fi

if command -v apt-get &>/dev/null; then
  PKG_MGR="apt"
elif [[ "$OS" == "macos" ]] && command -v brew &>/dev/null; then
  PKG_MGR="brew"
fi

echo ""
echo "================================================================"
echo "  SM3D SRP Library  --  build setup ($OS)"
echo "================================================================"

# ── 1. cmake ────────────────────────────────────────────────────────────────
step "Checking cmake"

if ! command -v cmake &>/dev/null; then
  err "cmake not found."
  if [[ "$PKG_MGR" == "apt" ]]; then
    read -rp "  Install cmake via apt now? [Y/n] " ans
    if [[ ! "$ans" =~ ^[Nn] ]]; then
      sudo apt-get install -y cmake
    fi
  elif [[ "$PKG_MGR" == "brew" ]]; then
    read -rp "  Install cmake via brew now? [Y/n] " ans
    if [[ ! "$ans" =~ ^[Nn] ]]; then
      brew install cmake
    fi
  fi
  if ! command -v cmake &>/dev/null; then
    err "cmake still not found. Install manually from https://cmake.org/download/"
    exit 1
  fi
fi

CMAKE_VER=$(cmake --version | head -1 | grep -oP '\d+\.\d+\.\d+')
ok "cmake $CMAKE_VER"

# ── 2. C++ compiler ──────────────────────────────────────────────────────────
step "Checking C++ compiler"

CXX_FOUND=""
for exe in g++ clang++; do
  if command -v $exe &>/dev/null; then
    CXX_FOUND=$(command -v $exe)
    ok "$exe  ->  $CXX_FOUND"
    break
  fi
done

if [[ -z "$CXX_FOUND" ]]; then
  err "No C++ compiler found (need g++ ≥ 12 or clang++ ≥ 15)."
  if [[ "$PKG_MGR" == "apt" ]]; then
    read -rp "  Install g++ via apt now? [Y/n] " ans
    if [[ ! "$ans" =~ ^[Nn] ]]; then
      sudo apt-get install -y g++
      CXX_FOUND=$(command -v g++)
      ok "g++ installed."
    fi
  elif [[ "$PKG_MGR" == "brew" ]]; then
    info "Run: brew install gcc"
  fi
  if [[ -z "$CXX_FOUND" ]]; then
    err "C++ compiler required — cannot continue."
    exit 1
  fi
fi

# ── 3. vcpkg (provides HDF5, HighFive, pybind11) ────────────────────────────
step "Looking for vcpkg"

VCPKG_ROOT=""

if [[ -n "$OPT_VCPKG" ]]; then
  VCPKG_ROOT="$OPT_VCPKG"
else
  for d in "${VCPKG_ROOT:-}" "$HOME/vcpkg" "$HOME/.local/vcpkg" "/opt/vcpkg" "/usr/local/vcpkg"; do
    if [[ -x "$d/vcpkg" ]]; then
      VCPKG_ROOT="$d"; break
    fi
  done
fi

if [[ -z "$VCPKG_ROOT" || ! -x "$VCPKG_ROOT/vcpkg" ]]; then
  warn "vcpkg not found."
  read -rp "  Install vcpkg to $HOME/vcpkg now? [Y/n] " ans
  if [[ ! "$ans" =~ ^[Nn] ]]; then
    git clone https://github.com/microsoft/vcpkg "$HOME/vcpkg" --depth 1
    "$HOME/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
    VCPKG_ROOT="$HOME/vcpkg"
    ok "vcpkg installed at $VCPKG_ROOT"
  else
    err "vcpkg is required for HDF5 + HighFive. Cannot continue."
    info "Manual install:"
    info "  git clone https://github.com/microsoft/vcpkg ~/vcpkg"
    info "  ~/vcpkg/bootstrap-vcpkg.sh -disableMetrics"
    info "  ~/vcpkg/vcpkg install hdf5 highfive pybind11"
    exit 1
  fi
else
  ok "vcpkg  ->  $VCPKG_ROOT"
fi

VCPKG_TRIPLET="x64-linux"
if [[ "$OS" == "macos" ]]; then
  VCPKG_TRIPLET="$(uname -m)-osx"  # arm64-osx or x86_64-osx
fi

VCPKG_INC="$VCPKG_ROOT/installed/$VCPKG_TRIPLET/include"
VCPKG_TOOLCHAIN="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

# Check / install required packages
NEED_HDF5=0; NEED_HF=0; NEED_PB11=0
[[ ! -f "$VCPKG_INC/hdf5.h" ]]                    && NEED_HDF5=1
[[ ! -f "$VCPKG_INC/highfive/highfive.hpp" ]]      && NEED_HF=1
[[ ! -f "$VCPKG_INC/pybind11/embed.h" ]]           && NEED_PB11=1

if [[ $NEED_HDF5 -eq 0 ]]; then ok "hdf5     ->  found"; else warn "hdf5     -- not installed"; fi
if [[ $NEED_HF   -eq 0 ]]; then ok "highfive ->  found"; else warn "highfive -- not installed"; fi
if [[ $NEED_PB11 -eq 0 ]]; then ok "pybind11 ->  found"; else warn "pybind11 -- not installed"; fi

if [[ $NEED_HDF5 -eq 1 || $NEED_HF -eq 1 || $NEED_PB11 -eq 1 ]]; then
  PKGS=""
  [[ $NEED_HDF5 -eq 1 ]] && PKGS="$PKGS hdf5"
  [[ $NEED_HF   -eq 1 ]] && PKGS="$PKGS highfive"
  [[ $NEED_PB11 -eq 1 ]] && PKGS="$PKGS pybind11"
  read -rp "  Install missing vcpkg packages ($PKGS )? [Y/n] " ans
  if [[ ! "$ans" =~ ^[Nn] ]]; then
    for pkg in $PKGS; do
      "$VCPKG_ROOT/vcpkg" install "$pkg"
    done
  else
    if [[ $NEED_HDF5 -eq 1 || $NEED_HF -eq 1 ]]; then
      err "HDF5 and HighFive are required. Cannot continue."
      exit 1
    fi
  fi
fi

# ── 4. CUDA (optional, Linux only) ──────────────────────────────────────────
CUDA_FOUND=0
CUDA_VERSION=""

if [[ "$OS" != "macos" ]]; then
  step "Looking for CUDA Toolkit (optional)"

  if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version 2>&1 | grep -oP 'release \K[0-9.]+' | head -1)
    CUDA_FOUND=1
    ok "CUDA $CUDA_VERSION  ->  $(command -v nvcc)"
  else
    warn "CUDA Toolkit not found — CentroidGPU / PixelGridGPU will be disabled."
    info "Install from: https://developer.nvidia.com/cuda-downloads"
  fi
else
  info "macOS: CUDA not supported by NVIDIA — skipping."
fi

# ── 5. OptiX SDK (optional, Linux only) ─────────────────────────────────────
OPTIX_FOUND=0
OPTIX_DIR=""

if [[ "$OS" != "macos" ]]; then
  step "Looking for OptiX SDK (optional)"

  if [[ -n "$OPT_OPTIX" ]]; then
    OPTIX_DIR="$OPT_OPTIX"
  elif [[ -n "${OPTIX_INSTALL_DIR:-}" ]]; then
    OPTIX_DIR="$OPTIX_INSTALL_DIR"
  else
    for d in /opt/optix /usr/local/optix "$HOME/optix" "$HOME/NVIDIA-OptiX-SDK"*; do
      if [[ -f "$d/include/optix.h" ]]; then
        OPTIX_DIR="$d"; break
      fi
    done
  fi

  if [[ -n "$OPTIX_DIR" && -f "$OPTIX_DIR/include/optix.h" ]]; then
    OPTIX_FOUND=1
    ok "OptiX  ->  $OPTIX_DIR"
  else
    OPTIX_DIR=""
    warn "OptiX SDK not found — CentroidRTX / PixelGridRTX will be disabled."
    info "Download: https://developer.nvidia.com/designworks/optix/downloads/legacy"
    info "Or: ./setup.sh --optix /path/to/OptiX"
  fi
else
  info "macOS: OptiX not supported — skipping."
fi

# Require CUDA for OptiX
if [[ $OPTIX_FOUND -eq 1 && $CUDA_FOUND -eq 0 ]]; then
  warn "OptiX requires CUDA — RTX methods disabled."
  OPTIX_FOUND=0; OPTIX_DIR=""
fi

# ── 6. Python (optional) ─────────────────────────────────────────────────────
PYTHON_FOUND=0
PYTHON_EXE=""
PYTHON_ROOT=""

step "Looking for Python (optional, needed for visualizeLastResult)"

if [[ -n "$OPT_PYTHON" ]]; then
  PYTHON_EXE="$OPT_PYTHON"
else
  for exe in python3 python; do
    if command -v $exe &>/dev/null; then
      VER=$($exe -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "")
      if [[ "$VER" == "3" ]]; then
        PYTHON_EXE=$(command -v $exe)
        break
      fi
    fi
  done
fi

if [[ -n "$PYTHON_EXE" ]]; then
  PYTHON_ROOT=$($PYTHON_EXE -c "import sys; print(sys.prefix)" 2>/dev/null || echo "")
  PY_VER=$($PYTHON_EXE --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1)
  if [[ -n "$PYTHON_ROOT" ]]; then
    PYTHON_FOUND=1
    ok "Python $PY_VER  ->  $PYTHON_EXE"
    ok "Prefix  ->  $PYTHON_ROOT"
    # Set PYTHONHOME if not already correct
    if [[ "${PYTHONHOME:-}" != "$PYTHON_ROOT" ]]; then
      export PYTHONHOME="$PYTHON_ROOT"
      # Persist in shell profile
      PROFILE_FILE="$HOME/.bashrc"
      [[ "$OS" == "macos" ]] && PROFILE_FILE="$HOME/.zshrc"
      if ! grep -q "PYTHONHOME=" "$PROFILE_FILE" 2>/dev/null; then
        echo "export PYTHONHOME=\"$PYTHON_ROOT\"" >> "$PROFILE_FILE"
        warn "Added PYTHONHOME to $PROFILE_FILE — restart terminal for it to take effect."
      fi
    else
      ok "PYTHONHOME already set."
    fi
  fi
else
  warn "Python 3 not found — visualizeLastResult() will be unavailable."
  info "Install: sudo apt install python3  (or brew install python)"
fi

# ── 7. Python packages ───────────────────────────────────────────────────────
if [[ $PYTHON_FOUND -eq 1 ]]; then
  step "Checking Python packages (numpy, pandas, plotly, matplotlib)"
  MISSING_PYPKGS=()
  for pkg in numpy pandas plotly matplotlib; do
    if $PYTHON_EXE -c "import $pkg" &>/dev/null 2>&1; then
      ok "$pkg"
    else
      warn "$pkg  -- not installed"
      MISSING_PYPKGS+=("$pkg")
    fi
  done

  if [[ ${#MISSING_PYPKGS[@]} -gt 0 ]]; then
    read -rp "  Install missing packages via pip now? [Y/n] " ans
    if [[ ! "$ans" =~ ^[Nn] ]]; then
      $PYTHON_EXE -m pip install "${MISSING_PYPKGS[@]}"
    fi
  fi
fi

# ── 8. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Build configuration"
echo "================================================================"

print_status() {
  local label="$1" value="$2"
  if [[ -n "$value" ]]; then
    printf "  ${GRN}%-16s${NC} %s\n" "$label" "$value"
  else
    printf "  ${YLW}%-16s${NC} not found (optional)\n" "$label"
  fi
}
print_status "CUDA"    "${CUDA_VERSION:-}"
print_status "OptiX"   "${OPTIX_DIR:-}"
print_status "Python"  "${PYTHON_ROOT:-}"
echo ""

METHODS="CentroidCPU, PixelGridCPU"
[[ $CUDA_FOUND   -eq 1 ]] && METHODS="$METHODS, CentroidGPU, PixelGridGPU"
[[ $OPTIX_FOUND  -eq 1 ]] && METHODS="$METHODS, CentroidRTX, PixelGridRTX"
echo -e "  Methods that will be compiled: ${GRN}$METHODS${NC}"
echo ""

# ── 9. cmake configure ───────────────────────────────────────────────────────
step "Configuring with CMake"

BUILD_DIR="$SCRIPT_DIR/build_dist"
DIST_DIR="$SCRIPT_DIR/dist"

if [[ $OPT_FORCE -eq 1 ]]; then
  rm -rf "$BUILD_DIR" "$DIST_DIR"
fi

CMAKE_ARGS=(
  "-B" "$BUILD_DIR"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCMAKE_TOOLCHAIN_FILE=$VCPKG_TOOLCHAIN"
  "-DVCPKG_TARGET_TRIPLET=$VCPKG_TRIPLET"
)

if [[ $CUDA_FOUND -eq 0 ]]; then
  CMAKE_ARGS+=("-DSM3D_ENABLE_CUDA=OFF" "-DSM3D_ENABLE_OPTIX=OFF")
elif [[ $OPTIX_FOUND -eq 0 ]]; then
  CMAKE_ARGS+=("-DSM3D_ENABLE_OPTIX=OFF")
else
  CMAKE_ARGS+=("-DOPTIX_INSTALL_DIR=$OPTIX_DIR")
fi

if [[ $PYTHON_FOUND -eq 0 ]]; then
  CMAKE_ARGS+=("-DSM3D_ENABLE_PYTHON=OFF")
fi

info "cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"
if [[ $? -ne 0 ]]; then
  err "cmake configure failed."
  info "Check TROUBLESHOOTING.md for common issues."
  exit 1
fi

# ── 10. build ────────────────────────────────────────────────────────────────
step "Building (this may take a few minutes)..."

NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
cmake --build "$BUILD_DIR" --parallel "$NPROC"
if [[ $? -ne 0 ]]; then
  err "Build failed."
  info "See output above for errors."
  exit 1
fi

ok "Build succeeded."

# ── 11. copy to dist/ ────────────────────────────────────────────────────────
step "Copying outputs to dist/"

mkdir -p "$DIST_DIR/include" "$DIST_DIR/lib" "$DIST_DIR/bin"

# Public header
cp "$SCRIPT_DIR/SM2D/SRPLibrary.h" "$DIST_DIR/include/"
ok "dist/include/SRPLibrary.h"

# Static library
LIB_SRC=""
for candidate in "$BUILD_DIR/libsrp.a" "$BUILD_DIR/Release/libsrp.a"; do
  if [[ -f "$candidate" ]]; then LIB_SRC="$candidate"; break; fi
done

if [[ -n "$LIB_SRC" ]]; then
  cp "$LIB_SRC" "$DIST_DIR/lib/libsrp.a"
  ok "dist/lib/libsrp.a"
else
  err "libsrp.a not found in build_dist — build may have failed."
fi

# OptiX PTX shaders
PTX_COUNT=0
while IFS= read -r -d '' ptx; do
  cp "$ptx" "$DIST_DIR/bin/"
  ok "dist/bin/$(basename "$ptx")"
  PTX_COUNT=$((PTX_COUNT + 1))
done < <(find "$BUILD_DIR" -name "*.ptx" -print0 2>/dev/null)

# ── 12. usage instructions ───────────────────────────────────────────────────
echo ""
echo -e "${GRN}================================================================${NC}"
echo -e "${GRN}  dist/ is ready!${NC}"
echo -e "${GRN}================================================================${NC}"
echo -e "  Available methods: ${GRN}$METHODS${NC}"
echo ""
echo "  To use in a CMake project:"
info "    target_include_directories(my_app PRIVATE path/to/dist/include)"
info "    target_link_libraries(my_app PRIVATE path/to/dist/lib/libsrp.a)"
echo ""
echo "  Or add the source directly:"
info "    add_subdirectory(path/to/SM3D_GPU_CC10)"
info "    target_link_libraries(my_app PRIVATE srp)"
echo ""
if [[ $PTX_COUNT -gt 0 ]]; then
  warn "Copy dist/bin/*.ptx next to your executable for RTX methods."
fi
warn "HDF5 is a shared dependency — link with hdf5 or ensure libhdf5.so is in your library path."
info "  See docs/USAGE.md for full integration guide."
echo ""
