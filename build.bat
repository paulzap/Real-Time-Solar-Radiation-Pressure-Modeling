@echo off
REM ===========================================================================
REM  SM3D SRP Library - one-click CPU/GPU build (double-click this file).
REM
REM  It runs setup.ps1 -Auto, which will:
REM    - auto-install vcpkg into %USERPROFILE%\vcpkg (if not already there)
REM    - auto-install the required packages (HDF5, HighFive, pybind11)
REM    - locate CMake from your Visual Studio installation
REM    - build the library into the  dist\  folder
REM
REM  Requirements: Visual Studio (Desktop C++ workload) + internet access.
REM  The FIRST run can take 15-30 minutes because vcpkg builds HDF5 from source.
REM ===========================================================================
setlocal
REM Run from this script's own folder (matters if launched elevated: an admin
REM cmd starts in C:\Windows\System32, which is not the project directory).
cd /d "%~dp0"

echo ============================================================
echo   SM3D SRP Library - one-click build
echo   First run may take 15-30 minutes (building HDF5 via vcpkg).
echo ============================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -Auto
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo ------------------------------------------------------------
    echo   DONE. The library is ready in the  dist\  folder:
    echo     dist\include\SRPLibrary.h
    echo     dist\lib\srp.lib
    echo     dist\SRPLibraryConfig.cmake
    echo   Use it from CMake:
    echo     find_package^(SRPLibrary REQUIRED CONFIG HINTS "path/to/dist"^)
    echo     target_link_libraries^(my_app PRIVATE SRPLibrary::srp^)
    echo ------------------------------------------------------------
) else (
    echo Build did not complete ^(exit code %RC%^). See the messages above.
)

echo.
pause
endlocal
