#Requires -Version 5.1
<#
.SYNOPSIS
    SM3D GPU SRP - auto-configure SM2D\LocalPaths.props

.DESCRIPTION
    Scans standard installation paths for CUDA, OptiX SDK, vcpkg+HDF5, Python.
    If everything is found - generates SM2D\LocalPaths.props.
    If something is missing - prints instructions on what to install.
    Missing Python packages (numpy, pandas, plotly, matplotlib) are installed automatically.

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -VcpkgRoot C:\tools\vcpkg -OptixRoot "C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0"

.PARAMETER VcpkgRoot
    Override vcpkg path (if not found automatically).

.PARAMETER OptixRoot
    Override OptiX SDK path.

.PARAMETER PythonRoot
    Override Python installation path.

.PARAMETER Force
    Overwrite LocalPaths.props even if it already exists.
#>
param(
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

$ScriptDir = $PSScriptRoot
$PropsFile = Join-Path $ScriptDir "SM2D\LocalPaths.props"

Write-Host ""
Write-Host "================================================================" -ForegroundColor White
Write-Host "  SM3D GPU SRP  --  environment setup" -ForegroundColor White
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
    if (-not $cudaVersion -and $cudaPath -match 'v(\d+\.\d+)') {
        $cudaVersion = $Matches[1]
    }
    Write-OK "CUDA $cudaVersion  ->  $cudaPath"
} else {
    Write-Err "CUDA Toolkit not found."
    Write-Info "Install from: https://developer.nvidia.com/cuda-downloads"
}

# -------------------------------------------------------------------------
# 2. GPU compute capability (via nvidia-smi)
# -------------------------------------------------------------------------
Write-Step "Detecting GPU architecture"

$gpuArch      = "compute_86,sm_86"
$optixPtxArch = "compute_75"

$smiPath = ""
$smiCandidates = @(
    "C:\Windows\System32\nvidia-smi.exe",
    "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
)
foreach ($p in $smiCandidates) {
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
            $cc = [int]"$($maj)$($min)"
            if ($cc -ge 75) { $optixPtxArch = "compute_$($maj)$($min)" }
        }
        $gpuName = (& $smiPath --query-gpu=name --format=csv,noheader 2>&1 |
                    Select-Object -First 1).Trim()
        Write-OK "$gpuName  ->  $gpuArch  (PTX: $optixPtxArch)"
    } catch {
        Write-Warn "nvidia-smi found but could not read compute capability. Using default: $gpuArch"
    }
} else {
    Write-Warn "nvidia-smi not found. Using default: $gpuArch"
    Write-Info "Make sure the NVIDIA driver is installed."
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
    Write-Err "OptiX SDK not found (expected optix.h inside SDK\include\)."
    Write-Info "Download OptiX SDK 7.x / 8.x / 9.x:"
    Write-Info "  https://developer.nvidia.com/designworks/optix/downloads/legacy"
    Write-Info "Installer places SDK at:"
    Write-Info "  C:\ProgramData\NVIDIA Corporation\OptiX SDK X.Y.Z\"
    Write-Info "Or specify the path manually:"
    Write-Info "  .\setup.ps1 -OptixRoot ""C:\path\to\OptiX SDK X.Y.Z"""
}

# -------------------------------------------------------------------------
# 4. vcpkg + required packages (HDF5, HighFive, pybind11)
# -------------------------------------------------------------------------
Write-Step "Looking for vcpkg + required packages"

$foundVcpkgRoot = $VcpkgRoot

if (-not $foundVcpkgRoot) {
    $vcpkgCandidates = @(
        $env:VCPKG_ROOT,
        "$env:USERPROFILE\vcpkg",
        "$env:USERPROFILE\source\repos\vcpkg",
        "C:\vcpkg",
        "C:\tools\vcpkg",
        "$env:LOCALAPPDATA\vcpkg",
        "C:\src\vcpkg"
    )
    foreach ($c in $vcpkgCandidates) {
        if ($c -and (Test-Path "$c\vcpkg.exe")) { $foundVcpkgRoot = $c; break }
    }
}

if ($foundVcpkgRoot -and (Test-Path "$foundVcpkgRoot\vcpkg.exe")) {
    Write-OK "vcpkg  ->  $foundVcpkgRoot"

    # Check each required package
    $vcpkgInc = "$foundVcpkgRoot\installed\x64-windows\include"
    $vcpkgPackages = @(
        @{ name = "hdf5";     header = "$vcpkgInc\hdf5.h";                 pkg = "hdf5:x64-windows"     },
        @{ name = "highfive"; header = "$vcpkgInc\highfive\highfive.hpp";   pkg = "highfive:x64-windows" },
        @{ name = "pybind11"; header = "$vcpkgInc\pybind11\embed.h";        pkg = "pybind11:x64-windows" }
    )

    $missingPkgs = @()
    foreach ($p in $vcpkgPackages) {
        if (Test-Path $p.header) {
            Write-OK "$($p.name)  ->  found"
        } else {
            Write-Warn "$($p.name)  -- NOT installed"
            $missingPkgs += $p
        }
    }

    if ($missingPkgs.Count -gt 0) {
        $pkgList = ($missingPkgs | ForEach-Object { $_.pkg }) -join "  "
        Write-Host ""
        $ans = Read-Host "  Install missing vcpkg packages now? [Y/n]"
        if ($ans -notmatch '^[Nn]') {
            foreach ($p in $missingPkgs) {
                Write-Info "Installing $($p.pkg) ..."
                & "$foundVcpkgRoot\vcpkg.exe" install $p.pkg
                if ($LASTEXITCODE -eq 0) {
                    Write-OK "$($p.name) installed."
                } else {
                    Write-Err "Failed to install $($p.pkg). Try manually:"
                    Write-Info "  cd `"$foundVcpkgRoot`"  &&  .\vcpkg install $($p.pkg)"
                }
            }
            Write-Info "Running: vcpkg integrate install ..."
            & "$foundVcpkgRoot\vcpkg.exe" integrate install
        } else {
            Write-Info "Skipped. Install manually:"
            Write-Info "  cd `"$foundVcpkgRoot`""
            Write-Info "  .\vcpkg install $pkgList"
            Write-Info "  .\vcpkg integrate install"
        }
    }
} else {
    $foundVcpkgRoot = ""
    Write-Err "vcpkg not found."
    Write-Info "Install vcpkg:"
    Write-Info "  git clone https://github.com/microsoft/vcpkg `"$env:USERPROFILE\vcpkg`""
    Write-Info "  cd `"$env:USERPROFILE\vcpkg`""
    Write-Info "  .\bootstrap-vcpkg.bat"
    Write-Info "  .\vcpkg install hdf5:x64-windows highfive:x64-windows pybind11:x64-windows"
    Write-Info "  .\vcpkg integrate install"
    Write-Info "Or specify the path manually:"
    Write-Info "  .\setup.ps1 -VcpkgRoot ""C:\path\to\vcpkg"""
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
            if (Test-Path "$prefix\include\Python.h") {
                $foundPythonRoot = $prefix
            }
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
        if (Test-Path "$p\include\Python.h") {
            $foundPythonRoot = $p
            $pythonExe = "$p\python.exe"
            break
        }
    }
}

if (-not $foundPythonRoot) {
    foreach ($regBase in @("HKCU:\Software\Python\PythonCore","HKLM:\Software\Python\PythonCore")) {
        if (Test-Path $regBase) {
            Get-ChildItem $regBase | Sort-Object Name -Descending | ForEach-Object {
                if (-not $foundPythonRoot) {
                    try {
                        $installPath = (Get-ItemProperty "$($_.PSPath)\InstallPath" -ErrorAction Stop).'(default)'
                        if ($installPath -and (Test-Path "$installPath\include\Python.h")) {
                            $foundPythonRoot = $installPath.TrimEnd('\')
                            $pythonExe = "$foundPythonRoot\python.exe"
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
        $libFile = Get-ChildItem "$foundPythonRoot\libs" -Filter "python*.lib" -ErrorAction SilentlyContinue |
                   Where-Object { $_.Name -notmatch '_d\.lib$' } |
                   Sort-Object Name -Descending | Select-Object -First 1
        if ($libFile) { $pythonLib = $libFile.Name }
    }
    Write-OK "Python  ->  $foundPythonRoot"
    Write-OK "Lib     ->  $pythonLib"
} else {
    $foundPythonRoot = ""; $pythonExe = ""; $pythonLib = ""
    Write-Err "Python not found (need include\Python.h)."
    Write-Info "Install Python 3.10+ from: https://www.python.org/downloads/"
    Write-Info "Check 'Add Python to PATH' during installation."
    Write-Info "Or specify manually:  .\setup.ps1 -PythonRoot ""C:\path\to\python"""
}

# -------------------------------------------------------------------------
# 6. Python packages for visualization
# -------------------------------------------------------------------------
Write-Step "Checking Python packages (numpy, pandas, plotly, matplotlib)"

$requiredPkgs = @("numpy", "pandas", "plotly", "matplotlib")
$missingPkgs  = @()

if ($pythonExe -and (Test-Path $pythonExe)) {
    foreach ($pkg in $requiredPkgs) {
        $check = (& $pythonExe -c "import importlib.util; print('ok' if importlib.util.find_spec('$pkg') else 'missing')" 2>&1).Trim()
        if ($check -eq 'ok') {
            Write-OK $pkg
        } else {
            Write-Warn "$pkg  -- not installed"
            $missingPkgs += $pkg
        }
    }

    if ($missingPkgs.Count -gt 0) {
        Write-Host ""
        $ans = Read-Host "  Install missing packages via pip now? [Y/n]"
        if ($ans -notmatch '^[Nn]') {
            foreach ($pkg in $missingPkgs) {
                Write-Info "Installing $pkg ..."
                & $pythonExe -m pip install $pkg --quiet
                if ($LASTEXITCODE -eq 0) {
                    Write-OK "$pkg installed"
                } else {
                    Write-Err "Failed to install $pkg"
                    Write-Info "Try manually:  $pythonExe -m pip install $pkg"
                }
            }
        } else {
            Write-Info "Skipped. Install manually:"
            Write-Info "  $pythonExe -m pip install $($missingPkgs -join ' ')"
        }
    }
} else {
    Write-Warn "Python not found -- package check skipped."
}

# -------------------------------------------------------------------------
# 7. Summary
# -------------------------------------------------------------------------
Write-Host ""
Write-Host "================================================================" -ForegroundColor White
Write-Host "  Summary" -ForegroundColor White
Write-Host "================================================================" -ForegroundColor White

$allOk = $true

function Show-Status {
    param([string]$label, [string]$value, [bool]$required = $true)
    if ($value) {
        Write-Host ("  {0,-14} {1}" -f $label, $value) -ForegroundColor Green
    } elseif ($required) {
        Write-Host ("  {0,-14} NOT FOUND  <--" -f $label) -ForegroundColor Red
        $script:allOk = $false
    } else {
        Write-Host ("  {0,-14} not found (optional)" -f $label) -ForegroundColor Yellow
    }
}

Show-Status "CUDA"      $cudaVersion
Show-Status "GPU arch"  $gpuArch
Show-Status "OptiX"     $foundOptixRoot
Show-Status "vcpkg"     $foundVcpkgRoot
Show-Status "Python"    $foundPythonRoot
Show-Status "PythonLib" $pythonLib

# -------------------------------------------------------------------------
# 7b. Set PYTHONHOME in user environment (needed by embedded pybind11)
# -------------------------------------------------------------------------
if ($foundPythonRoot) {
    $currentPH = [System.Environment]::GetEnvironmentVariable("PYTHONHOME", "User")
    if ($currentPH -ne $foundPythonRoot) {
        [System.Environment]::SetEnvironmentVariable("PYTHONHOME", $foundPythonRoot, "User")
        Write-OK "PYTHONHOME set to: $foundPythonRoot  (user environment)"
        Write-Warn "Restart Visual Studio / terminal for PYTHONHOME to take effect."
    } else {
        Write-OK "PYTHONHOME already set correctly."
    }
}

# -------------------------------------------------------------------------
# 8. Generate LocalPaths.props
# -------------------------------------------------------------------------
if (-not $allOk) {
    Write-Host ""
    Write-Err "Some required dependencies were not found."
    Write-Info "Fix the issues above and run setup.ps1 again."
    Write-Info "You can override paths with parameters:"
    Write-Info "  .\setup.ps1 -VcpkgRoot ""C:\path"" -OptixRoot ""C:\path"" -PythonRoot ""C:\path"""
    exit 1
}

if ((Test-Path $PropsFile) -and -not $Force) {
    Write-Host ""
    Write-Warn "$PropsFile already exists."
    $ans = Read-Host "  Overwrite? [y/N]"
    if ($ans -notmatch '^[Yy]') {
        Write-Info "Skipped. Use -Force to overwrite without prompting."
        exit 0
    }
}

# NOTE: closing "@ must be at column 0 (no leading whitespace)
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

# -------------------------------------------------------------------------
# 9. Patch CUDA version in SM2D.vcxproj
# -------------------------------------------------------------------------
# VS CUDA plugin requires a literal version string in the Import path --
# it cannot resolve MSBuild macros like $(CudaVersion) there.
# Detect what version is currently in the file and update it if needed.
Write-Step "Patching CUDA version in SM2D.vcxproj"

$vcxprojPath = Join-Path $ScriptDir "SM2D\SM2D.vcxproj"
if (Test-Path $vcxprojPath) {
    $vcxContent = Get-Content $vcxprojPath -Raw

    # Find the version in the actual <Import> line (not in comments)
    $currentVer = ""
    if ($vcxContent -match '<Import[^>]+CUDA (\d+\.\d+)\.props') { $currentVer = $Matches[1] }

    if ($currentVer -eq $cudaVersion) {
        Write-OK "SM2D.vcxproj already has CUDA $cudaVersion -- no change needed."
    } elseif ($currentVer) {
        $vcxContent = $vcxContent -replace "CUDA $([regex]::Escape($currentVer))\.props",   "CUDA $cudaVersion.props"
        $vcxContent = $vcxContent -replace "CUDA $([regex]::Escape($currentVer))\.targets", "CUDA $cudaVersion.targets"
        [System.IO.File]::WriteAllText($vcxprojPath, $vcxContent, [System.Text.UTF8Encoding]::new($false))
        Write-OK "Updated SM2D.vcxproj: CUDA $currentVer -> CUDA $cudaVersion"
        Write-Warn "Close and reopen the solution in Visual Studio to pick up the change."
    } else {
        Write-Warn "Could not detect CUDA version in SM2D.vcxproj -- please update it manually."
        Write-Info "Find the two lines with 'CUDA X.Y.props' / '.targets' and set them to CUDA $cudaVersion."
    }
} else {
    Write-Warn "SM2D.vcxproj not found at expected path: $vcxprojPath"
}

Write-Host ""
Write-Host "  Next step: close and reopen SM2D\SM2D.sln in Visual Studio 2022," -ForegroundColor White
Write-Host "  then run  Build -> Rebuild Solution (Release x64)." -ForegroundColor White
Write-Host ""
