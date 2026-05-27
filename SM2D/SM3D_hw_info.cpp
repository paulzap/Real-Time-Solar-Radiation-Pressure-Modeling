// SM3D_hw_info.cpp
// ============================================================
// Hardware information display: CUDA devices, compute
// capability, memory, OptiX/RTX support.
//
// NOTE: optix_function_table_definition.h is NOT included here;
// the function table is defined in gpu_optix_shared.cu.
// ============================================================

#include "gpu_optix_raytracer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static std::string fmt_ver(int v)
{
    std::ostringstream ss;
    ss << v / 1000 << "." << (v % 1000) / 10;
    return ss.str();
}

static const char* arch_name(int sm)
{
    if (sm >= 90) return "Hopper / Ada Lovelace";
    if (sm >= 89) return "Ada Lovelace (RTX 40xx)";
    if (sm >= 86) return "Ampere (RTX 30xx)";
    if (sm >= 80) return "Ampere (A100 / A30)";
    if (sm >= 75) return "Turing (RTX 20xx / T4)";
    if (sm >= 70) return "Volta (V100)";
    if (sm >= 61) return "Pascal (GTX 10xx / P40)";
    if (sm >= 60) return "Pascal (P100)";
    if (sm >= 52) return "Maxwell (GTX 9xx)";
    if (sm >= 50) return "Maxwell (GTX 7xx+)";
    return "Kepler";
}

static int cuda_cores_per_sm(int sm)
{
    if (sm >= 80) return 128;   // Ampere, Ada, Hopper
    if (sm >= 70) return 64;    // Volta, Turing
    if (sm >= 61) return 128;   // Pascal GTX
    if (sm >= 60) return 64;    // Pascal P100
    if (sm >= 50) return 128;   // Maxwell
    return 192;                 // Kepler
}

// ---------------------------------------------------------------------------
// print_hardware_info
// ---------------------------------------------------------------------------
void print_hardware_info()
{
    constexpr int W = 32;   // label column width
    auto row = [&](const char* label, const std::string& value) {
        std::cout << "  " << std::left << std::setw(W) << label << value << "\n";
    };

    std::cout << "\n========================================================\n";
    std::cout <<   "              Hardware & Driver Information           \n";
    std::cout <<   "========================================================\n\n";

    // ── CUDA versions ───────────────────────────────────────────────────────
    int runtime_ver = 0, driver_ver = 0;
    cudaRuntimeGetVersion(&runtime_ver);
    cudaDriverGetVersion(&driver_ver);

    std::cout << "  CUDA Runtime : " << fmt_ver(runtime_ver) << "\n";
    std::cout << "  CUDA Driver  : " << fmt_ver(driver_ver)  << "\n";

    // ── OptiX runtime init ───────────────────────────────────────────────────
    // Delegated to gpu_optix_shared.cu (a .cu unit) which can call optixInit().
    bool optix_ok = is_optix_runtime_available();
    std::cout << "  OptiX init   : " << (optix_ok ? "OK" : "FAILED") << "\n";

    // ── GPU devices ─────────────────────────────────────────────────────────
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::cout << "\n  No CUDA devices found.\n";
        return;
    }

    std::cout << "\n  Found " << dev_count << " CUDA device(s)\n";

    int active_dev = 0;
    cudaGetDevice(&active_dev);

    for (int d = 0; d < dev_count; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);

        int sm = p.major * 10 + p.minor;

        std::cout << "\n   Device " << d << (d == active_dev ? " [ACTIVE]" : "")
                  << " --------------------------------------------------\n";
        std::cout << "  \n";

        // Name & SM
        row("  -  GPU model",         std::string(p.name));
        {
            std::ostringstream ss;
            ss << p.major << "." << p.minor << "  (" << arch_name(sm) << ")";
            row("  -  Compute Capability", ss.str());
        }

        // Feature support
        {
            auto yn = [](bool b, const char* note="") -> std::string {
                return b ? std::string("YES") : (std::string("NO") + (note[0] ? std::string("  - ") + note : ""));
            };
            row("  -  RTX (hardware RT)",     yn(sm >= 75, "requires SM 7.5+  (Turing)"));
            row("  -  OptiX capable",         yn(sm >= 50, "requires SM 5.0+"));
            row("  -  Native fp64 atomicAdd", yn(sm >= 60, "CAS fallback used for SM < 6.0"));
            row("  -  Tensor Cores",          yn(sm >= 70, "requires SM 7.0+  (Volta)"));
        }

        // Memory
        {
            double total_gb = p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << total_gb << " GB";
            row("  -  GPU memory (total)", ss.str());
        }

        // Compute resources
        {
            int cps  = cuda_cores_per_sm(sm);
            int total_cores = p.multiProcessorCount * cps;
            std::ostringstream ss;
            ss << p.multiProcessorCount
               << " SMs × " << cps << " = " << total_cores << " CUDA cores (approx)";
            row("  -  Streaming Multiprocs", ss.str());
        }

        // Clocks
        {
            int clockKHz = 0;
            cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, d);
            std::ostringstream ss;
            ss << clockKHz / 1000 << " MHz  (boost clock)";
            row("  -  GPU clock", ss.str());
        }

        // Memory bandwidth
        {
            int memClockKHz = 0, busWidth = 0;
            cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, d);
            cudaDeviceGetAttribute(&busWidth,    cudaDevAttrGlobalMemoryBusWidth, d);
            double bw = 2.0 * (memClockKHz / 1.0e6) * (busWidth / 8.0);
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << bw << " GB/s"
               << "  (bus width: " << busWidth << " bit)";
            row("  -  Memory bandwidth", ss.str());
        }

        // Cache
        {
            std::ostringstream ss;
            ss << p.l2CacheSize / 1024 << " KB";
            row("  -  L2 cache", ss.str());
        }

        // Threads
        {
            std::ostringstream ss;
            ss << p.warpSize << " threads/warp,  max "
               << p.maxThreadsPerBlock << " threads/block";
            row("  -  Warp / block limits", ss.str());
        }

        // PCI
        {
            std::ostringstream ss;
            ss << p.pciBusID << ":" << p.pciDeviceID;
            row("  -  PCI Bus:Device", ss.str());
        }

        std::cout << "  -\n";
        std::cout << "  -----------------------------------------------------\n";
    }

    // ── Summary ─────────────────────────────────────────────────────────────
    std::cout << "\n  Build targets compiled into this binary:\n";
    std::cout << "    SM 5.2 (Maxwell)   - broad compatibility, CAS double atomicAdd\n";
    std::cout << "    SM 8.6 (Ampere)    - native double atomicAdd, full RTX/OptiX\n";
    std::cout << "  The SM of the active device is used at runtime; older paths are\n";
    std::cout << "  never executed on this hardware.\n\n";
}

// ---------------------------------------------------------------------------
// Availability helpers - used by main_calculation()
// ---------------------------------------------------------------------------

bool is_cuda_device_available()
{
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
}

// Returns empty string when RTX is fully usable; otherwise a human-readable
// reason explaining why it is not available.
std::string rtx_unavailable_reason()
{
    if (!is_optix_runtime_available())
        return "OptiX runtime not loaded (nvoptix.dll missing or incompatible driver)";

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return "no CUDA-capable GPU found";

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    int sm = p.major * 10 + p.minor;
    if (sm < 75) {
        std::ostringstream ss;
        ss << "GPU \"" << p.name << "\" is SM " << p.major << "." << p.minor
           << " - RT Cores require SM 7.5+ (Turing / RTX 20xx or newer)";
        return ss.str();
    }
    return "";  // available
}

bool is_rtx_device_available()
{
    return rtx_unavailable_reason().empty();
}
