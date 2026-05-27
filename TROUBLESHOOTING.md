# Troubleshooting

[English](TROUBLESHOOTING.md) · [Русский](TROUBLESHOOTING.ru.md)

---

### `error MSB4019: imported project "CUDA X.Y.props" not found`

Your installed CUDA version does not match the version hardcoded in
`SM2D.vcxproj`. The VS CUDA plugin requires a literal version string in the
import path and does not support MSBuild macros there.

**Fix:** Run `.\setup.ps1` — it patches `SM2D.vcxproj` automatically.
Or edit `SM2D.vcxproj` manually: find `CUDA 12.4.props` / `CUDA 12.4.targets`
and replace `12.4` with your installed version (e.g. `13.1`).

---

### `error MSB4019` — but `setup.ps1` reported `[OK] already has CUDA 13.1`

`setup.ps1` matched its regex against a comment line that mentioned `CUDA 13.1`
rather than the actual `<Import>` tag.

**Fix:** Open `SM2D.vcxproj` in a text editor and find both `<Import>` tags
containing `BuildCustomizations`. Verify they have the correct version:
```xml
<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.1.props" />
...
<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.1.targets" />
```
If wrong — run `.\setup.ps1 -Force`.

---

### `fatal error C1083: highfive/highfive.hpp: No such file or directory`
### `fatal error C1083: pybind11/embed.h: No such file or directory`

The vcpkg packages `highfive` and/or `pybind11` are not installed.
All three packages are required.

**Fix:**
```powershell
vcpkg install highfive:x64-windows pybind11:x64-windows
```
Or re-run `.\setup.ps1` — it checks all three required packages and installs
missing ones with your confirmation.

---

### `CUDA error: too many resources requested for launch (error 701)`

You are building or running in the **Debug** configuration. Debug CUDA kernels
disable register optimisations and add full debug instrumentation, causing
register count per thread to exceed SM limits. Note: some simple kernels (e.g.
OptiX raygen) may still work in Debug, giving a false sense that the
configuration is correct — the error only surfaces on complex kernels.

**Fix:** Switch to **Release | x64** in Visual Studio and **Rebuild Solution**.

---

### `Visualisation skipped: Failed to import encodings module`
### `ModuleNotFoundError: No module named 'encodings'`

The embedded Python interpreter (pybind11) cannot find the standard library.
`PYTHONHOME` is not set or points to the wrong Python installation.

**Fix:**
1. Run `.\setup.ps1` — it locates Python and sets `PYTHONHOME` in user
   environment variables.
2. **Restart Visual Studio** afterwards — VS inherits environment variables
   at launch time, not at runtime.
3. Or set manually in PowerShell:
   ```powershell
   [System.Environment]::SetEnvironmentVariable("PYTHONHOME",
       "C:\Users\<you>\AppData\Local\Programs\Python\Python312", "User")
   ```
4. Alternative: set in VS project settings:
   `Project → Properties → Debugging → Environment → PYTHONHOME=C:\...\Python312`

---

### `CUDA error: PTX JIT compilation failed` / blank OptiX output

The `.ptx` shader files are missing from the executable directory.
Pre-build events compile them; they may be skipped on a partial build.

**Fix:** **Build → Rebuild Solution** (not just Build). Confirm the four
`.ptx` files appear alongside `SM3D.exe`:
```
optix_shaders_center.ptx
optix_shaders_center_bench.ptx
optix_shaders_pixel_grid.ptx
optix_shaders_pixel_grid_bench.ptx
```

---

### `No HDF5 files found in <path>`

The path passed to `SRPEngine(folder)` does not contain `.h5` / `.hdf5` files,
or the working directory is wrong.

**Fix:** Run from `x64\Release\` and pass a relative path:
```powershell
cd x64\Release
.\SM3D.exe ..\..\data3d_hdf5_0.5
```

---

### CPU and GPU results differ noticeably

**Expected difference:** `Centroid*` and `PixelGrid*` sample visibility
differently — a ~1–5% difference on complex geometry is normal.

**Possible bug:** if `CentroidCPU` and `CentroidGPU` diverge by more than
0.1%, the shadow-check logic in one of the kernels may be incorrect.

---

### Programme launches in Debug from VS (F5) instead of Release

VS stores the active configuration in the `.suo` file. Even after switching
to Release in the toolbar, F5 may launch the previously built Debug binary.

**Fix:** After switching to Release, do **Build → Rebuild Solution**, then F5.
Or run directly from `x64\Release\SM3D.exe`.
