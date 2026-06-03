#Requires -Version 5.1
<#
.SYNOPSIS
    SM3D SRP Library — environment setup

.DESCRIPTION
    VS mode (default):
        Scans for CUDA, OptiX, vcpkg, Python.
        Generates SM2D\LocalPaths.props for Visual Studio.
        All dependencies are required.

    Library mode  (-Mode Library):
        Same scan, then runs CMake and builds srp.lib into dist\.
        CUDA / OptiX / Python are OPTIONAL — missing tools reduce
        available methods but do not stop the build.
        Only HDF5 + HighFive (via vcpkg) are required.

.EXAMPLE
    .\setup.ps1                   # VS mode: check + write LocalPaths.props
    .\setup.ps1 -Mode Library     # Library mode: check + build dist\srp.lib
    .\setup.ps1 -Mode Library -Force  # rebuild even if dist\ already exists

.PARAMETER Mode
    "VS" (default) or "Library"

.PARAMETER VcpkgRoot
    Override vcpkg path.

.PARAMETER OptixRoot
    Override OptiX SDK path.

.PARAMETER PythonRoot
    Override Python path.

.PARAMETER Force
    VS: overwrite LocalPaths.props without prompting.
    Library: delete and recreate build_dist\ and dist\.
#>
param(
    [ValidateSet("VS","Library")]
    [string]$Mode       = "VS",
    [string]$VcpkgRoot  = "",
    [string]$OptixRoot  = "",
    [string]$PythonRoot = "",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

function Write-OK   { param($msg) Write-Host "  [OK]  $msg" -ForegroundColor Green  }
function Write-Warn { param($msg) Write-Host "  [!]   $msg" -ForegroundColor Yellow }
function Write-Err  { param($msg) Write-Host "  [ERR] $msg" -ForegroundColor Red    }
function Write-Info { param($msg) Write-Host "        $msg" -ForegroundColor Cyan   }
function Write-Step { param($msg) Write-Host "`n==> $msg"   -ForegroundColor White  }

$ScriptDir  = $PSScriptRoot
$PropsFile  = Join-Path $ScriptDir "SM2D\LocalPaths.props"
$ModeLabel  = if ($Mode -eq "Library") { "Library build" } else { "VS setup" }

Write-Host ""
Write-Host "================================================================" -ForegroundColor White
Write-Host "  SM3D SRP Library  --  $ModeLabel" -ForegroundColor White
Write-Host "================================================================" -ForegroundColor White

# -------------------------------------------------------------------------
# 1. CUDA Toolkit
# -------------------------------------------------------------------------
Write-Step "Looking for CUDA Toolkit"

$cudaVersion = ""
$cudaPath    = ""

if ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) {
    $cudaPath = $env:CUDA_PATH
} else {
    Get-ChildItem Env: | Where-Object { $_.Name -match '^CUDA_PATH_V\d+_\d+$' } |
        Sort-Object Name -Descending | ForEach-Object {
            if (-not $cudaPath -and (Test-Path "$($_.Value)\bin\nvcc.exe")) {
                $cudaPath = $_.Value
            }
        }
}

if (-not $cudaPath) {
    $nvBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $nvBase) {
        Get-ChildItem $nvBase -Directory | Sort-Object Name -Descending | ForEach-Object {
            if (-not $cudaPath -and (Test-Path "$($_.FullName)\bin\nvcc.exe")) {
                $cudaPath = $_.FullName
            }
        }
    }
}

if ($cudaPath) {
    try {
        $nvccOut = & "$cudaPath\bin\nvcc.exe" --version 2>&1 | Out-String
        if ($nvccOut -match 'release (\d+\.\d+)') { $cudaVersion = $Matches[1] }
    } catch {}
    if (-not $cudaVersion -and $cudaPath -match 'v(\d+\.\d+)') { $cudaVersion = $Matches[1] }
    Write-OK "CUDA $cudaVersion  ->  $cudaPath"
} else {
    if ($Mode -eq "VS") { Write-Err  "CUDA Toolkit not found." }
    else                { Write-Warn "CUDA Toolkit not found — GPU/RTX methods will be disabled." }
    Write-Info "Install from: https://developer.nvidia.com/cuda-downloads"
}

# -------------------------------------------------------------------------
# 2. GPU compute capability
# -------------------------------------------------------------------------
Write-Step "Detecting GPU architecture"

$gpuArch      = "compute_86,sm_86"
$optixPtxArch = "compute_75"

$smiPath = ""
foreach ($p in @("C:\Windows\System32\nvidia-smi.exe",
                 "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")) {
    if (Test-Path $p) { $smiPath = $p; break }
}
if (-not $smiPath) {
    $smiCmd = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($smiCmd) { $smiPath = $smiCmd.Source }
}

if ($smiPath) {
    try {
        $capStr = (& $smiPath --query-gpu=compute_cap --format=csv,noheader 2>&1 |
                   Select-Object -First 1).Trim()
        if ($capStr -match '^(\d+)\.(\d+)$') {
            $maj = $Matches[1]; $min = $Matches[2]
            $gpuArch = "compute_$($maj)$($min),sm_$($maj)$($min)"
            if ([int]"$($maj)$($min)" -ge 75) { $optixPtxArch = "compute_$($maj)$($min)" }
        }
        $gpuName = (& $smiPath --query-gpu=name --format=csv,noheader 2>&1 |
                    Select-Object -First 1).Trim()
        Write-OK "$gpuName  ->  $gpuArch"
    } catch {
        Write-Warn "nvidia-smi found but could not read compute capability. Default: $gpuArch"
    }
} else {
    Write-Warn "nvidia-smi not found. Default: $gpuArch"
}

# -------------------------------------------------------------------------
# 3. OptiX SDK
# -------------------------------------------------------------------------
Write-Step "Looking for OptiX SDK"

$foundOptixRoot = $OptixRoot

if (-not $foundOptixRoot) {
    $optixBase = "C:\ProgramData\NVIDIA Corporation"
    if (Test-Path $optixBase) {
        Get-ChildItem $optixBase -Directory -Filter "OptiX SDK*" |
            Sort-Object Name -Descending | ForEach-Object {
                if (-not $foundOptixRoot -and (Test-Path "$($_.FullName)\include\optix.h")) {
                    $foundOptixRoot = $_.FullName
                }
            }
    }
}

if ($foundOptixRoot -and (Test-Path "$foundOptixRoot\include\optix.h")) {
    Write-OK "$(Split-Path $foundOptixRoot -Leaf)  ->  $foundOptixRoot"
} else {
    $foundOptixRoot = ""
    if ($Mode -eq "VS") { Write-Err  "OptiX SDK not found." }
    else                { Write-Warn "OptiX SDK not found — RTX methods will be disabled." }
    Write-Info "Download: https://developer.nvidia.com/designworks/optix/downloads/legacy"
    Write-Info "Or: .\setup.ps1 -OptixRoot ""C:\path\to\OptiX SDK X.Y.Z"""
}

# -------------------------------------------------------------------------
# 4. vcpkg + required packages
# -------------------------------------------------------------------------
Write-Step "Looking for vcpkg + required packages (HDF5, HighFive, pybind11)"

$foundVcpkgRoot = $VcpkgRoot

if (-not $foundVcpkgRoot) {
    foreach ($c in @($env:VCPKG_ROOT, "$env:USERPROFILE\vcpkg",
                     "$env:USERPROFILE\source\repos\vcpkg",
                     "C:\vcpkg", "C:\tools\vcpkg",
                     "$env:LOCALAPPDATA\vcpkg", "C:\src\vcpkg")) {
        if ($c -and (Test-Path "$c\vcpkg.exe")) { $foundVcpkgRoot = $c; break }
    }
}

if ($foundVcpkgRoot -and (Test-Path "$foundVcpkgRoot\vcpkg.exe")) {
    Write-OK "vcpkg  ->  $foundVcpkgRoot"

    $vcpkgInc  = "$foundVcpkgRoot\installed\x64-windows\include"
    $vcpkgPkgs = @(
        @{ name = "hdf5";     header = "$vcpkgInc\hdf5.h";               pkg = "hdf5:x64-windows"     },
        @{ name = "highfive"; header = "$vcpkgInc\highfive\highfive.hpp"; pkg = "highfive:x64-windows" },
        @{ name = "pybind11"; header = "$vcpkgInc\pybind11\embed.h";      pkg = "pybind11:x64-windows" }
    )

    $missingPkgs = @()
    foreach ($p in $vcpkgPkgs) {
        if (Test-Path $p.header) { Write-OK "$($p.name)  ->  found" }
        else                     { Write-Warn "$($p.name)  -- not installed"; $missingPkgs += $p }
    }

    if ($missingPkgs.Count -gt 0) {
        $ans = Read-Host "`n  Install missing vcpkg packages now? [Y/n]"
        if ($ans -notmatch '^[Nn]') {
            foreach ($p in $missingPkgs) {
                Write-Info "Installing $($p.pkg) ..."
                & "$foundVcpkgRoot\vcpkg.exe" install $p.pkg
                if ($LASTEXITCODE -eq 0) { Write-OK "$($p.name) installed." }
                else { Write-Err "Failed to install $($p.pkg). Run manually: vcpkg install $($p.pkg)" }
            }
            & "$foundVcpkgRoot\vcpkg.exe" integrate install
        } else {
            $pkgList = ($missingPkgs | ForEach-Object { $_.pkg }) -join "  "
            Write-Info "Run manually: cd `"$foundVcpkgRoot`" && .\vcpkg install $pkgList"
        }
    }
} else {
    $foundVcpkgRoot = ""
    Write-Err "vcpkg not found."
    Write-Info "Install vcpkg:"
    Write-Info "  git clone https://github.com/microsoft/vcpkg `"$env:USERPROFILE\vcpkg`""
    Write-Info "  cd `"$env:USERPROFILE\vcpkg`" && .\bootstrap-vcpkg.bat"
    Write-Info "  .\vcpkg install hdf5:x64-windows highfive:x64-windows pybind11:x64-windows"
    Write-Info "  .\vcpkg integrate install"
}

# -------------------------------------------------------------------------
# 5. Python
# -------------------------------------------------------------------------
Write-Step "Looking for Python"

$foundPythonRoot = $PythonRoot
$pythonExe       = ""
$pythonLib       = ""

if (-not $foundPythonRoot) {
    $pyCmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $pythonExe = $pyCmd.Source
        try {
            $prefix = (& $pythonExe -c "import sys; print(sys.prefix)" 2>&1).Trim()
            if (Test-Path "$prefix\include\Python.h") { $foundPythonRoot = $prefix }
        } catch {}
    }
}

if (-not $foundPythonRoot) {
    $pyBase = "$env:LOCALAPPDATA\Programs\Python"
    if (Test-Path $pyBase) {
        Get-ChildItem $pyBase -Directory | Sort-Object Name -Descending | ForEach-Object {
            if (-not $foundPythonRoot -and (Test-Path "$($_.FullName)\include\Python.h")) {
                $foundPythonRoot = $_.FullName
                $pythonExe       = "$foundPythonRoot\python.exe"
            }
        }
    }
}

if (-not $foundPythonRoot) {
    foreach ($p in @("C:\Python313","C:\Python312","C:\Python311","C:\Python310")) {
        if (Test-Path "$p\include\Python.h") { $foundPythonRoot = $p; $pythonExe = "$p\python.exe"; break }
    }
}

if (-not $foundPythonRoot) {
    foreach ($regBase in @("HKCU:\Software\Python\PythonCore","HKLM:\Software\Python\PythonCore")) {
        if (Test-Path $regBase) {
            Get-ChildItem $regBase | Sort-Object Name -Descending | ForEach-Object {
                if (-not $foundPythonRoot) {
                    try {
                        $ip = (Get-ItemProperty "$($_.PSPath)\InstallPath" -ErrorAction Stop).'(default)'
                        if ($ip -and (Test-Path "$ip\include\Python.h")) {
                            $foundPythonRoot = $ip.TrimEnd('\'); $pythonExe = "$foundPythonRoot\python.exe"
                        }
                    } catch {}
                }
            }
        }
    }
}

if ($foundPythonRoot -and (Test-Path "$foundPythonRoot\include\Python.h")) {
    if (-not $pythonExe) { $pythonExe = "$foundPythonRoot\python.exe" }
    try {
        $pyVer = (& $pythonExe -c "import sys; print(str(sys.version_info.major)+str(sys.version_info.minor))" 2>&1).Trim()
        $pythonLib = "python$pyVer.lib"
    } catch {}
    if (-not $pythonLib) {
        $lf = Get-ChildItem "$foundPythonRoot\libs" -Filter "python*.lib" -ErrorAction SilentlyContinue |
              Where-Object { $_.Name -notmatch '_d\.lib$' } | Sort-Object Name -Descending | Select-Object -First 1
        if ($lf) { $pythonLib = $lf.Name }
    }
    Write-OK "Python  ->  $foundPythonRoot"
    Write-OK "Lib     ->  $pythonLib"
} else {
    $foundPythonRoot = ""; $pythonExe = ""; $pythonLib = ""
    if ($Mode -eq "VS") { Write-Err  "Python not found (need include\Python.h)." }
    else                { Write-Warn "Python not found — visualizeLastResult() will be unavailable." }
    Write-Info "Install Python 3.10+ from: https://www.python.org/downloads/"
}

# -------------------------------------------------------------------------
# 6. Python packages
# -------------------------------------------------------------------------
Write-Step "Checking Python packages (numpy, pandas, plotly, matplotlib)"

if ($pythonExe -and (Test-Path $pythonExe)) {
    $missingPyPkgs = @()
    foreach ($pkg in @("numpy","pandas","plotly","matplotlib")) {
        $check = (& $pythonExe -c "import importlib.util; print('ok' if importlib.util.find_spec('$pkg') else 'missing')" 2>&1).Trim()
        if ($check -eq 'ok') { Write-OK $pkg } else { Write-Warn "$pkg  -- not installed"; $missingPyPkgs += $pkg }
    }

    if ($missingPyPkgs.Count -gt 0) {
        $ans = Read-Host "`n  Install missing packages via pip now? [Y/n]"
        if ($ans -notmatch '^[Nn]') {
            foreach ($pkg in $missingPyPkgs) {
                & $pythonExe -m pip install $pkg --quiet
                if ($LASTEXITCODE -eq 0) { Write-OK "$pkg installed" }
                else { Write-Err "Failed to install $pkg. Run: $pythonExe -m pip install $pkg" }
            }
        }
    }
} else {
    Write-Warn "Python not found — package check skipped."
}

# -------------------------------------------------------------------------
# 7. Summary
# -------------------------------------------------------------------------
Write-Host ""
Write-Host "================================================================" -ForegroundColor White
Write-Host "  Summary" -ForegroundColor White
Write-Host "================================================================" -ForegroundColor White

# In Library mode: CUDA / OptiX / Python are optional (reduce available methods).
# In VS mode: all are required.
$allOk = $true

function Show-Status {
    param([string]$label, [string]$value, [bool]$required = $true)
    # $required: if false AND Mode=="Library", show as optional (yellow), not error
    $isRequired = $required -or ($script:Mode -eq "VS")
    if ($value) {
        Write-Host ("  {0,-16} {1}" -f $label, $value) -ForegroundColor Green
    } elseif ($isRequired) {
        Write-Host ("  {0,-16} NOT FOUND  <--" -f $label) -ForegroundColor Red
        $script:allOk = $false
    } else {
        Write-Host ("  {0,-16} not found  (optional — some methods disabled)" -f $label) -ForegroundColor Yellow
    }
}

Show-Status "CUDA"      $cudaVersion     -required $false   # optional in Library mode
Show-Status "GPU arch"  $gpuArch         -required $false
Show-Status "OptiX"     $foundOptixRoot  -required $false   # optional in Library mode
Show-Status "vcpkg"     $foundVcpkgRoot  -required $true    # always required
Show-Status "Python"    $foundPythonRoot -required $false   # optional in Library mode
Show-Status "PythonLib" $pythonLib       -required $false

# Set PYTHONHOME (both modes — needed for embedded Python at runtime)
if ($foundPythonRoot) {
    $currentPH = [System.Environment]::GetEnvironmentVariable("PYTHONHOME", "User")
    if ($currentPH -ne $foundPythonRoot) {
        [System.Environment]::SetEnvironmentVariable("PYTHONHOME", $foundPythonRoot, "User")
        Write-OK "PYTHONHOME set: $foundPythonRoot"
        Write-Warn "Restart terminal / VS for PYTHONHOME to take effect."
    } else {
        Write-OK "PYTHONHOME already correct."
    }
}

if (-not $allOk) {
    Write-Host ""
    Write-Err "Required dependencies are missing — see above."
    Write-Info "Fix the issues and run setup.ps1 again."
    Write-Info "Override paths: .\setup.ps1 -VcpkgRoot C:\path -OptixRoot C:\path -PythonRoot C:\path"
    exit 1
}

# =========================================================================
# VS MODE — generate LocalPaths.props + patch SM2D.vcxproj
# =========================================================================
if ($Mode -eq "VS") {

    if ((Test-Path $PropsFile) -and -not $Force) {
        $ans = Read-Host "`n  $PropsFile already exists. Overwrite? [y/N]"
        if ($ans -notmatch '^[Yy]') { Write-Info "Skipped. Use -Force to overwrite."; exit 0 }
    }

    $propsContent = @"
<?xml version="1.0" encoding="utf-8"?>
<!-- Generated by setup.ps1 on $(Get-Date -Format 'yyyy-MM-dd HH:mm') -->
<!-- DO NOT COMMIT this file (it is in .gitignore). -->
<!-- If you upgrade CUDA, also update the "CUDA X.Y.props" / ".targets" lines in SM2D.vcxproj manually. -->
<Project ToolsVersion="4.0"
         xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <PropertyGroup Label="UserLocalPaths">
    <CudaVersion>$cudaVersion</CudaVersion>
    <GpuArch>$gpuArch</GpuArch>
    <OptixPtxArch>$optixPtxArch</OptixPtxArch>
    <OptixRoot>$foundOptixRoot</OptixRoot>
    <VcpkgRoot>$foundVcpkgRoot</VcpkgRoot>
    <PythonRoot>$foundPythonRoot</PythonRoot>
    <PythonLib>$pythonLib</PythonLib>
  </PropertyGroup>
</Project>
"@

    $propsContent | Set-Content -Path $PropsFile -Encoding UTF8
    Write-OK "Created: $PropsFile"

    # Patch CUDA version in SM2D.vcxproj
    Write-Step "Patching CUDA version in SM2D.vcxproj"
    $vcxprojPath = Join-Path $ScriptDir "SM2D\SM2D.vcxproj"
    if (Test-Path $vcxprojPath) {
        $vcxContent = Get-Content $vcxprojPath -Raw
        $currentVer = ""
        if ($vcxContent -match '<Import[^>]+CUDA (\d+\.\d+)\.props') { $currentVer = $Matches[1] }

        if ($currentVer -eq $cudaVersion) {
            Write-OK "SM2D.vcxproj already has CUDA $cudaVersion."
        } elseif ($currentVer) {
            $vcxContent = $vcxContent -replace "CUDA $([regex]::Escape($currentVer))\.props",   "CUDA $cudaVersion.props"
            $vcxContent = $vcxContent -replace "CUDA $([regex]::Escape($currentVer))\.targets", "CUDA $cudaVersion.targets"
            [System.IO.File]::WriteAllText($vcxprojPath, $vcxContent, [System.Text.UTF8Encoding]::new($false))
            Write-OK "Updated SM2D.vcxproj: CUDA $currentVer -> CUDA $cudaVersion"
            Write-Warn "Close and reopen the .sln in Visual Studio."
        } else {
            Write-Warn "Could not detect CUDA version in SM2D.vcxproj — update manually."
        }
    } else {
        Write-Warn "SM2D.vcxproj not found."
    }

    Write-Host ""
    Write-Host "  Next step: open SM2D\SM2D.sln in Visual Studio 2022," -ForegroundColor White
    Write-Host "  then Build -> Rebuild Solution  (Release | x64)." -ForegroundColor White
    Write-Host ""
    exit 0
}

# =========================================================================
# LIBRARY MODE — cmake + build + copy to dist\
# =========================================================================

Write-Step "Building srp.lib with CMake"

$buildDir = Join-Path $ScriptDir "build_dist"
$distDir  = Join-Path $ScriptDir "dist"

if ($Force) {
    if (Test-Path $buildDir) { Remove-Item $buildDir -Recurse -Force }
    if (Test-Path $distDir)  { Remove-Item $distDir  -Recurse -Force }
}

# ---- cmake configure ----
$cmakeArgs = @("-B", $buildDir)

if ($foundVcpkgRoot) {
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$foundVcpkgRoot\scripts\buildsystems\vcpkg.cmake"
    $cmakeArgs += "-DVCPKG_TARGET_TRIPLET=x64-windows"
}

# Disable missing optional features
if (-not $cudaPath) {
    $cmakeArgs += "-DSM3D_ENABLE_CUDA=OFF"
    $cmakeArgs += "-DSM3D_ENABLE_OPTIX=OFF"
} elseif (-not $foundOptixRoot) {
    $cmakeArgs += "-DSM3D_ENABLE_OPTIX=OFF"
} else {
    $cmakeArgs += "-DOPTIX_INSTALL_DIR=$foundOptixRoot"
}
if (-not $foundPythonRoot) {
    $cmakeArgs += "-DSM3D_ENABLE_PYTHON=OFF"
}

Write-Info "cmake $($cmakeArgs -join ' ')"
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { Write-Err "cmake configure failed."; exit 1 }

# ---- cmake build ----
Write-Step "Compiling (Release)..."
& cmake --build $buildDir --config Release
if ($LASTEXITCODE -ne 0) { Write-Err "cmake build failed."; exit 1 }

# ---- copy to dist\ ----
Write-Step "Copying outputs to dist\"

New-Item -ItemType Directory -Force (Join-Path $distDir "include") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $distDir "lib")     | Out-Null
New-Item -ItemType Directory -Force (Join-Path $distDir "bin")     | Out-Null

# Public header
Copy-Item (Join-Path $ScriptDir "SM2D\SRPLibrary.h") (Join-Path $distDir "include") -Force
Write-OK "dist\include\SRPLibrary.h"

# Static library (multi-config VS generator puts it in Release\; Ninja puts it directly)
$libCandidates = @(
    (Join-Path $buildDir "Release\srp.lib"),
    (Join-Path $buildDir "srp.lib")
)
$libSrc = $libCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($libSrc) {
    Copy-Item $libSrc (Join-Path $distDir "lib\srp.lib") -Force
    Write-OK "dist\lib\srp.lib"
} else {
    Write-Err "srp.lib not found in build_dist — build may have failed."
}

# OptiX PTX shaders
$ptxFiles = Get-ChildItem $buildDir -Filter "*.ptx" -Recurse -ErrorAction SilentlyContinue
foreach ($ptx in $ptxFiles) {
    Copy-Item $ptx.FullName (Join-Path $distDir "bin") -Force
    Write-OK "dist\bin\$($ptx.Name)"
}

# HDF5 runtime DLL (needed next to user's exe)
if ($foundVcpkgRoot) {
    $hdf5dll = Join-Path $foundVcpkgRoot "installed\x64-windows\bin\hdf5.dll"
    if (Test-Path $hdf5dll) {
        Copy-Item $hdf5dll (Join-Path $distDir "bin") -Force
        Write-OK "dist\bin\hdf5.dll"
    }
}

# Python visualization script
New-Item -ItemType Directory -Force (Join-Path $distDir "python") | Out-Null
Copy-Item (Join-Path $ScriptDir "SM2D\visualize3d.py") (Join-Path $distDir "python") -Force
Write-OK "dist\python\visualize3d.py"

# ---- generate SRPLibraryConfig.cmake (with baked-in vcpkg hint) ----
$vcpkgInstalled = ""
if ($foundVcpkgRoot) {
    $candidate = Join-Path $foundVcpkgRoot "installed\x64-windows"
    if (Test-Path $candidate) { $vcpkgInstalled = $candidate }
}

$cmakeConfig = @"
# SRPLibraryConfig.cmake -- auto-generated by setup.ps1
# Usage: find_package(SRPLibrary REQUIRED CONFIG HINTS "path/to/dist")
#        target_link_libraries(my_app PRIVATE SRPLibrary::srp)
cmake_minimum_required(VERSION 3.15)
get_filename_component(_SRP_DIR "`${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)

# Help CMake find vcpkg-installed packages (path baked in at build time)
foreach(_hint
    "$vcpkgInstalled"
    "`$ENV{VCPKG_ROOT}/installed/x64-windows"
    "`$ENV{HOME}/vcpkg/installed/x64-windows"
    "C:/vcpkg/installed/x64-windows")
  if(IS_DIRECTORY "`${_hint}")
    list(APPEND CMAKE_PREFIX_PATH "`${_hint}")
  endif()
endforeach()

# HDF5 - try CONFIG mode first (vcpkg hdf5-config pulls in zlib + szip automatically)
find_package(HDF5 CONFIG QUIET)
if(HDF5_FOUND AND TARGET hdf5::hdf5)
    set(_SRP_HDF5_TARGET "hdf5::hdf5")
elseif(HDF5_FOUND AND TARGET HDF5::HDF5)
    set(_SRP_HDF5_TARGET "HDF5::HDF5")
else()
    find_package(HDF5 REQUIRED COMPONENTS C)
    set(_SRP_HDF5_TARGET "HDF5::HDF5")
    find_package(ZLIB QUIET)
    find_package(libaec QUIET CONFIG)
endif()

# Python (required if the library was built with visualization support)
if(EXISTS "`${_SRP_DIR}/python/visualize3d.py")
    find_package(Python3 COMPONENTS Development QUIET)
endif()

if(NOT TARGET SRPLibrary::srp)
    add_library(SRPLibrary::srp STATIC IMPORTED GLOBAL)
    find_library(_SRP_LIB NAMES srp HINTS "`${_SRP_DIR}/lib" NO_DEFAULT_PATH)
    set_target_properties(SRPLibrary::srp PROPERTIES
        IMPORTED_LOCATION             "`${_SRP_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "`${_SRP_DIR}/include"
    )
    set_property(TARGET SRPLibrary::srp APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES "`${_SRP_HDF5_TARGET}")
    if(ZLIB_FOUND AND NOT TARGET hdf5::hdf5)
        set_property(TARGET SRPLibrary::srp APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES ZLIB::ZLIB)
    endif()
    if(libaec_FOUND AND NOT TARGET hdf5::hdf5)
        set_property(TARGET SRPLibrary::srp APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES libaec::sz)
    endif()
    if(Python3_FOUND AND EXISTS "`${_SRP_DIR}/python/visualize3d.py")
        set_property(TARGET SRPLibrary::srp APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES Python3::Python)
    endif()
endif()
"@
$cmakeConfig | Out-File -FilePath (Join-Path $distDir "SRPLibraryConfig.cmake") -Encoding utf8
Write-OK "dist\SRPLibraryConfig.cmake"

# ---- print usage instructions ----
$methods = "CentroidCPU, PixelGridCPU"
if ($cudaPath)        { $methods += ", CentroidGPU, PixelGridGPU" }
if ($foundOptixRoot)  { $methods += ", CentroidRTX, PixelGridRTX" }

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  dist\ is ready!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Available methods: $methods" -ForegroundColor White
Write-Host ""
Write-Host "  To use in a CMake project:" -ForegroundColor White
Write-Info "    find_package(SRPLibrary REQUIRED CONFIG HINTS `"path/to/dist`")"
Write-Info "    target_link_libraries(my_app PRIVATE SRPLibrary::srp)"
Write-Host ""
Write-Host "  For visualization (visualizeLastResult):" -ForegroundColor White
Write-Info "    Copy dist\python\visualize3d.py to your executable's working directory."
Write-Host ""
if ($ptxFiles.Count -gt 0) {
    Write-Warn "Copy dist\bin\*.ptx next to your .exe for RTX methods."
}
Write-Warn "Copy dist\bin\hdf5.dll next to your .exe."
Write-Host ""
