"""
turbo_simulator.py — 2D Homogeneous Turbulence DNS (SciPy / CuPy port)

This is a structural port of dns_all.cu to Python.

Key ideas kept from the CUDA version:
  • DnsState structure mirrors DnsDeviceState (Nbase, NX, NZ, NK, NX_full, NZ_full, NK_full)
  • UR (compact)  : shape (NZ, NX, 3)   — AoS: [z, x, comp]
  • UC (compact)  : shape (NZ, NK, 3)   — spectral, [z, kx, comp]
  • UR_full (3/2) : shape (3, NZ_full, NX_full)   — SoA: [comp, z, x]
  • UC_full (3/2) : shape (3, NZ_full, NK_full)   — spectral, SoA
  • om2, fnm1     : shape (NZ, NX_half) — spectral vorticity & non-linear term
  • alfa[NX_half], gamma[NZ]

The integration loop is STEP2B → STEP3 → STEP2A → NEXTDT, like dns_all.cu

Backends:
  • CPU: SciPy (scipy.fft / numpy arrays)
  • GPU: CuPy (cupyx.scipy.fft / cupy arrays), with cuFFT plans when available

Implementation details:
  • complex64 / float32 are used everywhere for speed & parity with CUDA
  • Most of the CUDA kernels are translated to vectorized CuPy/SciPy operations
  • STEP3 has an optional fused RawKernel implementation for GPU, to reduce kernel launches
"""

import argparse
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as _np
import scipy
from scipy import fft as _spfft

try:
    import cupy as _cp
    from cupyx.scipy import fft as _cpfft
except Exception:
    _cp = None
    _cpfft = None


# ===============================================================
# CONSTANTS (match Fortran)
# ===============================================================

PI = math.pi


# ===============================================================
# DnsState — mirrors DnsDeviceState (CUDA version)
# ===============================================================

@dataclass
class DnsState:
    # sizes
    Nbase: int
    NX: int
    NZ: int
    NK: int

    NX_full: int
    NZ_full: int
    NK_full: int

    # parameters
    Re: float
    k0: float
    cflnum: float
    visc: float

    # time integration state
    t: float
    dt: float
    cn: float
    cnm1: float
    iteration: int

    # backend
    backend: str
    xp: Any
    fft: Any
    fft_workers: int

    # fields (compact)
    ur: Any  # (NZ, NX, 3) float32
    uc: Any  # (NZ, NK, 3) complex64

    # fields (full 3/2 grid)
    ur_full: Any  # (3, NZ_full, NX_full) float32
    uc_full: Any  # (3, NZ_full, NK_full) complex64

    # derived spectral fields
    om2: Any   # (NZ, NX_half) complex64
    fnm1: Any  # (NZ, NX_half) complex64

    # wavenumbers (compact)
    alfa: Any   # (NX_half,) float32
    gamma: Any  # (NZ_full,) float32

    # optional FFT plans (CuPy)
    fft_plan_rfft2_ur_full: Any
    fft_plan_irfft2_uc01: Any

    # STEP3 precompute arrays (compact / low band)
    step3_z_spec: Any
    step3_GA: Any
    step3_G2mA2: Any
    step3_K2: Any
    step3_invK2_sub: Any
    step3_divxz: float
    step3_inv_gamma0: float

    # STEP3 scratch
    scratch1: Any
    scratch2: Any


# ===============================================================
# Backend selection
# ===============================================================

def _fft_mod_for_state(backend: str):
    if backend == "gpu" and _cpfft is not None:
        return _cpfft
    return _spfft


def _xp_for_backend(backend: str):
    if backend == "gpu" and _cp is not None:
        return _cp
    return _np


# ===============================================================
# ENV INFO
# ===============================================================

def _print_env_info():
    if _cp is None:
        print(" Checking CuPy...")
        print(" CuPy not installed")
        return

    print(" Checking CuPy...")
    try:
        import cupy
        from cupy._environment import get_cuda_path
        from cupy.cuda import runtime

        print(f"OS                           : {platform.platform()}")
        print(f"Python Version               : {platform.python_version()}")
        print(f"CuPy Version                 : {cupy.__version__}")
        print(f"CuPy Platform                : {cupy._environment.get_cuda_path() and 'NVIDIA CUDA' or 'Unknown'}")
        print(f"NumPy Version                : {_np.__version__}")
        print(f"SciPy Version                : {scipy.__version__}")

        try:
            import Cython
            print(f"Cython Build Version         : {getattr(Cython, '__version__', None)}")
            print(f"Cython Runtime Version       : {getattr(Cython, 'Runtime', None)}")
        except Exception:
            print("Cython Build Version         : None")
            print("Cython Runtime Version       : None")

        cuda_root = get_cuda_path()
        print(f"CUDA Root                    : {cuda_root}")

        nvcc = None
        if cuda_root:
            nvcc = os.path.join(cuda_root, "bin", "nvcc.exe") if os.name == "nt" else os.path.join(cuda_root, "bin", "nvcc")
        print(f"nvcc PATH                    : {nvcc}")

        # versions
        try:
            print(f"CUDA Build Version           : {runtime.get_build_version()}")
            print(f"CUDA Driver Version          : {runtime.driverGetVersion()}")
            print(f"CUDA Runtime Version         : {runtime.runtimeGetVersion()} (linked to CuPy) / {runtime.get_local_runtime_version()} (locally installed)")
        except Exception:
            pass

        # device
        dev = runtime.getDevice()
        props = runtime.getDeviceProperties(dev)
        name = props.get("name", b"").decode("utf-8", "ignore")
        cc = f"{props.get('major', 0)}{props.get('minor', 0)}"
        pci = props.get("pciBusID", None)

        print(f"Device {dev} Name                : {name}")
        print(f"Device {dev} Compute Capability  : {cc}")
        if pci is not None:
            print(f"Device {dev} PCI Bus ID          : {pci}")

    except Exception:
        # keep it simple
        pass


# ===============================================================
# PAO init
# ===============================================================

def frand(n: int, seed: int) -> _np.ndarray:
    rng = _np.random.default_rng(seed)
    return rng.random(n, dtype=_np.float32)


def dns_pao_host_init(N: int, Re: float, K0: float, seed: int) -> Tuple[_np.ndarray, float]:
    """
    Host-side PAO init returning UC_host (NZ, NX, 3) complex64 and visc.

    In your full impl this matches the CUDA/Fortran output.
    """
    NX = int(N)
    NZ = int(N)
    NX_half = NX // 2

    visc = 1.0 / float(Re)

    UC_host = _np.zeros((NZ, NX, 3), dtype=_np.complex64)

    # Deterministic random spectrum
    r1 = frand(NZ * NX_half, seed=seed).reshape(NZ, NX_half)
    r2 = frand(NZ * NX_half, seed=seed + 1).reshape(NZ, NX_half)
    ph = _np.exp(1j * (2.0 * _np.pi) * r2).astype(_np.complex64)
    amp = (r1 ** 0.5).astype(_np.float32)
    spec = (amp * ph).astype(_np.complex64)

    UC_host[:, :NX_half, 0] = spec
    UC_host[:, :NX_half, 1] = spec * _np.complex64(0.7 + 0.3j)
    UC_host[:, :NX_half, 2] = spec * _np.complex64(-0.2 + 0.9j)

    UC_host[0, 0, :] = 0.0 + 0.0j

    return UC_host, visc


# ===============================================================
# Create state
# ===============================================================

def create_dns_state(
    N: int,
    Re: float,
    K0: float,
    CFL: float,
    backend: str = "auto",
    seed: int = 149,
    workers: Optional[int] = None,
) -> DnsState:

    if backend == "auto":
        backend = "gpu" if _cp is not None else "cpu"

    if backend == "gpu" and _cp is None:
        backend = "cpu"

    xp = _xp_for_backend(backend)
    fft = _fft_mod_for_state(backend)

    Nbase = int(N)
    NX = Nbase
    NZ = Nbase
    NX_half = NX // 2
    NK = NX_half

    NX_full = (3 * NX) // 2
    NZ_full = (3 * NZ) // 2
    NK_full = NX_full // 2 + 1

    UC_host, visc = dns_pao_host_init(N=Nbase, Re=Re, K0=K0, seed=seed)

    # Compact arrays
    ur = xp.zeros((NZ, NX, 3), dtype=xp.float32)
    uc = xp.zeros((NZ, NK, 3), dtype=xp.complex64)

    # Full arrays
    ur_full = xp.zeros((3, NZ_full, NX_full), dtype=xp.float32)
    uc_full = xp.zeros((3, NZ_full, NK_full), dtype=xp.complex64)

    # Copy initial spectrum into low band (z< NZ, kx < NX_half)
    if backend == "gpu":
        UC = _cp.asarray(UC_host, dtype=_cp.complex64)
        uc_full[0, :NZ, :NX_half] = UC[:, :NX_half, 0]
        uc_full[1, :NZ, :NX_half] = UC[:, :NX_half, 1]
        uc_full[2, :NZ, :NX_half] = UC[:, :NX_half, 2]
    else:
        uc_full[0, :NZ, :NX_half] = UC_host[:, :NX_half, 0]
        uc_full[1, :NZ, :NX_half] = UC_host[:, :NX_half, 1]
        uc_full[2, :NZ, :NX_half] = UC_host[:, :NX_half, 2]

    # Derived spectral fields (compact)
    om2 = xp.zeros((NZ, NX_half), dtype=xp.complex64)
    fnm1 = xp.zeros((NZ, NX_half), dtype=xp.complex64)

    # Wavenumbers
    alfa = xp.zeros((NX_half,), dtype=xp.float32)
    gamma = xp.zeros((NZ_full,), dtype=xp.float32)
    for ix in range(NX_half):
        alfa[ix] = xp.float32(ix)
    for iz in range(NZ_full):
        kk = iz if iz <= NZ_full // 2 else iz - NZ_full
        gamma[iz] = xp.float32(kk)

    # FFT workers
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) // 2)
    fft_workers = int(workers)

    # cuFFT plans (optional)
    fft_plan_rfft2_ur_full = None
    fft_plan_irfft2_uc01 = None
    if backend == "gpu":
        plan_mod = None
        if _cpfft is not None and hasattr(_cpfft, "get_fft_plan"):
            plan_mod = _cpfft

        if plan_mod is not None:
            # Forward: rfft2 on real UR_full over (z,x) axes
            state_ur_full_for_plan = ur_full
            fft_plan_rfft2_ur_full = plan_mod.get_fft_plan(
                state_ur_full_for_plan, axes=(1, 2), value_type="R2C"
            )
            # Inverse: irfft2 on UC_full[0:2] over (z,x) axes back to real
            state_uc01_for_plan = uc_full[0:2, :, :]
            fft_plan_irfft2_uc01 = plan_mod.get_fft_plan(
                state_uc01_for_plan, axes=(1, 2), value_type="C2R"
            )

    # STEP3 z_spec mapping (compact z -> full-grid z)
    if backend == "gpu":
        zspec_host = _np.empty((NZ,), dtype=_np.int32)
        for z in range(NZ):
            zspec_host[z] = _np.int32(z if z <= NZ // 2 else z + (NZ_full - NZ))
        step3_z_spec = _cp.asarray(zspec_host, dtype=_cp.int32)
    else:
        step3_z_spec = _np.empty((NZ,), dtype=_np.int32)
        for z in range(NZ):
            step3_z_spec[z] = _np.int32(z if z <= NZ // 2 else z + (NZ_full - NZ))

    # STEP3 precompute (compact low band)
    if backend == "gpu":
        GA_h = _np.empty((NZ, NX_half), dtype=_np.float32)
        G2mA2_h = _np.empty((NZ, NX_half), dtype=_np.float32)
        K2_h = _np.empty((NZ, NX_half), dtype=_np.float32)
        invK2_h = _np.empty((NZ, NX_half), dtype=_np.float32)
        for z in range(NZ):
            kz = z if z <= NZ // 2 else z - NZ
            for kx in range(NX_half):
                a = float(kx)
                g = float(kz)
                ga = g * a
                k2 = a * a + g * g
                GA_h[z, kx] = _np.float32(ga)
                G2mA2_h[z, kx] = _np.float32(g * g - a * a)
                K2_h[z, kx] = _np.float32(k2)
                invK2_h[z, kx] = _np.float32(0.0 if k2 == 0.0 else (1.0 / k2))
        step3_GA = _cp.asarray(GA_h)
        step3_G2mA2 = _cp.asarray(G2mA2_h)
        step3_K2 = _cp.asarray(K2_h)
        step3_invK2_sub = _cp.asarray(invK2_h)
    else:
        step3_GA = _np.empty((NZ, NX_half), dtype=_np.float32)
        step3_G2mA2 = _np.empty((NZ, NX_half), dtype=_np.float32)
        step3_K2 = _np.empty((NZ, NX_half), dtype=_np.float32)
        step3_invK2_sub = _np.empty((NZ, NX_half), dtype=_np.float32)
        for z in range(NZ):
            kz = z if z <= NZ // 2 else z - NZ
            for kx in range(NX_half):
                a = float(kx)
                g = float(kz)
                ga = g * a
                k2 = a * a + g * g
                step3_GA[z, kx] = _np.float32(ga)
                step3_G2mA2[z, kx] = _np.float32(g * g - a * a)
                step3_K2[z, kx] = _np.float32(k2)
                step3_invK2_sub[z, kx] = _np.float32(0.0 if k2 == 0.0 else (1.0 / k2))

    step3_divxz = float(1.0 / (NX_full * NZ_full))
    step3_inv_gamma0 = float(1.0)

    scratch1 = xp.zeros((NZ, NX_half), dtype=xp.complex64)
    scratch2 = xp.zeros((NZ, NX_half), dtype=xp.complex64)

    return DnsState(
        Nbase=Nbase,
        NX=NX,
        NZ=NZ,
        NK=NK,
        NX_full=NX_full,
        NZ_full=NZ_full,
        NK_full=NK_full,
        Re=float(Re),
        k0=float(K0),
        cflnum=float(CFL),
        visc=float(visc),
        t=0.0,
        dt=0.0,
        cn=1.0,
        cnm1=1.0,
        iteration=0,
        backend=backend,
        xp=xp,
        fft=fft,
        fft_workers=fft_workers,
        ur=ur,
        uc=uc,
        ur_full=ur_full,
        uc_full=uc_full,
        om2=om2,
        fnm1=fnm1,
        alfa=alfa,
        gamma=gamma,
        fft_plan_rfft2_ur_full=fft_plan_rfft2_ur_full,
        fft_plan_irfft2_uc01=fft_plan_irfft2_uc01,
        step3_z_spec=step3_z_spec,
        step3_GA=step3_GA,
        step3_G2mA2=step3_G2mA2,
        step3_K2=step3_K2,
        step3_invK2_sub=step3_invK2_sub,
        step3_divxz=step3_divxz,
        step3_inv_gamma0=step3_inv_gamma0,
        scratch1=scratch1,
        scratch2=scratch2,
    )


# ===============================================================
# FFT helpers (full)
# ===============================================================

def vfft_full_inverse_uc_full_to_ur_full(S: DnsState) -> None:
    xp = S.xp
    UC = S.uc_full
    fft = S.fft

    UC01 = UC[0:2, :, :]

    if S.backend == "cpu":
        ur01 = fft.irfft2(UC01, s=(S.NZ_full, S.NX_full), axes=(1, 2), overwrite_x=True)
    else:
        plan = S.fft_plan_irfft2_uc01
        if plan is not None:
            ur01 = fft.irfft2(UC01, s=(S.NZ_full, S.NX_full), axes=(1, 2), plan=plan)
        else:
            ur01 = fft.irfft2(UC01, s=(S.NZ_full, S.NX_full), axes=(1, 2))

    # Match previous STEP2A behavior exactly: scale BEFORE float32 cast/assign.
    ur01 *= (S.NZ_full * S.NX_full)

    S.ur_full[0:2, :, :] = xp.asarray(ur01, dtype=xp.float32)
    S.ur_full[2, :, :] = xp.float32(0.0)


def vfft_full_forward_ur_full_to_uc_full(S: DnsState) -> None:
    """
    UR_full (3, NZ_full, NX_full) → UC_full (3, NZ_full, NK_full)

    Correct forward:
      1) real FFT along x      (real → complex)
      2) FFT along z           (complex → complex)

    ONLY CHANGE: use rfft2 on (z,x) axes.
    """
    # S.ur_full is already float32
    UR = S.ur_full
    fft = S.fft

    if S.backend == "cpu":
        # overwrite_x is safe here (UR_full is overwritten later by STEP2A anyway)
        UC = fft.rfft2(UR, s=(S.NZ_full, S.NX_full), axes=(1, 2), overwrite_x=True, workers=S.fft_workers)
    else:
        plan = S.fft_plan_rfft2_ur_full
        if plan is not None:
            UC = fft.rfft2(UR, s=(S.NZ_full, S.NX_full), axes=(1, 2), overwrite_x=True, plan=plan)
        else:
            UC = fft.rfft2(UR, s=(S.NZ_full, S.NX_full), axes=(1, 2), overwrite_x=True)

    # Assign back; uc_full is complex64, assignment will down-cast if needed
    S.uc_full[...] = UC


# ---------------------------------------------------------------------------
# CALCOM — spectral vorticity from UC_full (dnsCudaCalcom)
# ---------------------------------------------------------------------------

def dns_calcom_from_uc_full(S: DnsState) -> None:
    xp = S.xp

    uc0 = S.uc_full[0, :S.NZ, :S.NX // 2]
    uc1 = S.uc_full[1, :S.NZ, :S.NX // 2]

    # omega = i*(kx*u1 - kz*u0)
    kx = S.alfa.reshape(1, -1).astype(xp.float32, copy=False)
    kz = S.gamma[:S.NZ].reshape(-1, 1).astype(xp.float32, copy=False)

    diff = (kx * uc1) - (kz * uc0)
    diff_r = diff.real
    diff_i = diff.imag

    om_r = -diff_i
    om_i = diff_r

    S.om2[...] = xp.asarray(om_r + 1j * om_i, dtype=xp.complex64)


# ---------------------------------------------------------------------------
# STEP2B — build uiuj and forward FFT (dnsCudaStep2B)
# ---------------------------------------------------------------------------
_STEP2B_MUL3_KERNEL = None  # created lazily on first GPU call
_STEP3_UPDATE_KERNEL = None  # created lazily on first GPU call
_STEP3_BUILD_UC_KERNEL = None  # created lazily on first GPU call
_STEP2A_CROP_KERNEL = None  # created lazily on first GPU call


def dns_step2b(S: DnsState) -> None:
    """
    Python/CuPy port of dnsCudaStep2B(DnsDeviceState *S).

    Mirrors Fortran STEP2B:

      1) Build uiuj in UR(x,z,1..3) on the full 3/2 grid
      2) Full-grid forward FFT: UR_full → UC_full (3 components)
         (VRF
    """
    xp = S.xp

    # uiuj products
    u0 = S.ur_full[0]
    u1 = S.ur_full[1]

    if S.backend == "gpu" and _cp is not None:
        global _STEP2B_MUL3_KERNEL
        if _STEP2B_MUL3_KERNEL is None:
            mul_src = r'''
            extern "C" __global__
            void turbo_mul3(
                const float* __restrict__ u0,
                const float* __restrict__ u1,
                float* __restrict__ u2,
                const int n
            ){
                int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                if (tid >= n) return;
                u2[tid] = u0[tid] * u1[tid];
            }
            '''
            _STEP2B_MUL3_KERNEL = _cp.RawKernel(mul_src, "turbo_mul3")

        n = int(S.NZ_full) * int(S.NX_full)
        threads = 256
        blocks = (n + threads - 1) // threads
        _STEP2B_MUL3_KERNEL((blocks,), (threads,), (u0, u1, S.ur_full[2], _np.int32(n)))
    else:
        S.ur_full[2] = (u0 * u1).astype(xp.float32, copy=False)

    # forward FFT for all 3 components
    vfft_full_forward_ur_full_to_uc_full(S)


# ---------------------------------------------------------------------------
# STEP3 — update om2/fnm1 and rebuild low-k uc0/uc1 (dnsCudaStep3)
# ---------------------------------------------------------------------------

def dns_step3(S: DnsState, fuse: bool = True) -> None:
    xp = S.xp

    if S.backend == "gpu" and _cp is not None and fuse:
        global _STEP3_UPDATE_KERNEL, _STEP3_BUILD_UC_KERNEL

        if _STEP3_UPDATE_KERNEL is None or _STEP3_BUILD_UC_KERNEL is None:
            update_src = r"""
            extern "C" __global__
            void turbo_step3_update(
                const cuFloatComplex* __restrict__ uc0,
                const cuFloatComplex* __restrict__ uc1,
                const cuFloatComplex* __restrict__ uc2,
                const int* __restrict__ z_spec,
                const float* __restrict__ GA,
                const float* __restrict__ G2mA2,
                const float* __restrict__ K2,
                cuFloatComplex* __restrict__ om2,
                cuFloatComplex* __restrict__ fnm1,
                const int NK_full,
                const int NX_half,
                const int NZ,
                const float divxz,
                const float visc,
                const float dt,
                const float cnm1
            ){
                int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                int n = NZ * NX_half;
                if (tid >= n) return;

                int z = tid / NX_half;
                int k = tid - z * NX_half;
                int zsrc = z_spec[z];

                int idx_uc = zsrc * NK_full + k;

                cuFloatComplex U0 = uc0[idx_uc];
                cuFloatComplex U1 = uc1[idx_uc];
                cuFloatComplex U2 = uc2[idx_uc];

                float ga = GA[tid];
                float g2ma2 = G2mA2[tid];
                float k2 = K2[tid];

                cuFloatComplex FN;
                FN.x = divxz * (ga * U0.x + g2ma2 * U1.x + k2 * U2.x);
                FN.y = divxz * (ga * U0.y + g2ma2 * U1.y + k2 * U2.y);

                cuFloatComplex OM = om2[tid];

                float denom = (2.0f + cnm1 + dt * visc * k2);
                float a = (2.0f - cnm1) / denom;
                float b = (dt) / denom;

                cuFloatComplex OMN;
                OMN.x = a * OM.x + b * FN.x;
                OMN.y = a * OM.y + b * FN.y;

                om2[tid] = OMN;
                fnm1[tid] = FN;
            }
            """

            build_src = r"""
            extern "C" __global__
            void turbo_step3_build_uc(
                const cuFloatComplex* __restrict__ om2,
                const float* __restrict__ invK2,
                const float* __restrict__ gamma,
                const float* __restrict__ alfa,
                const float inv_gamma0,
                cuFloatComplex* __restrict__ out0,
                cuFloatComplex* __restrict__ out1,
                const int NX_half,
                const int NZ
            ){
                int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                int n = NZ * NX_half;
                if (tid >= n) return;

                int z = tid / NX_half;
                int k = tid - z * NX_half;

                float g = gamma[z];
                float a = alfa[k];

                float invk2 = invK2[tid];
                cuFloatComplex OM = om2[tid];

                cuFloatComplex U0;
                U0.x = -g * invk2 * OM.y;
                U0.y =  g * invk2 * OM.x;

                cuFloatComplex U1;
                U1.x =  a * invk2 * OM.y;
                U1.y = -a * invk2 * OM.x;

                U0.x *= inv_gamma0; U0.y *= inv_gamma0;
                U1.x *= inv_gamma0; U1.y *= inv_gamma0;

                out0[tid] = U0;
                out1[tid] = U1;
            }
            """

            _STEP3_UPDATE_KERNEL = _cp.RawKernel(update_src, "turbo_step3_update")
            _STEP3_BUILD_UC_KERNEL = _cp.RawKernel(build_src, "turbo_step3_build_uc")

        uc_full = S.uc_full
        NZ = int(S.NZ)
        NX_half = int(S.Nbase // 2)
        NK_full = int(S.NK_full)

        threads = 256
        n = NZ * NX_half
        blocks = (n + threads - 1) // threads

        # IMPORTANT: cast scalar kernel args to correct widths
        NK_full_i32 = _np.int32(NK_full)
        NX_half_i32 = _np.int32(NX_half)
        NZ_i32 = _np.int32(NZ)

        divxz_f32 = _np.float32(S.step3_divxz)
        visc_f32 = _np.float32(S.visc)
        dt_f32 = _np.float32(S.dt)
        cnm1_f32 = _np.float32(S.cnm1)

        _STEP3_UPDATE_KERNEL(
            (blocks,),
            (threads,),
            (
                uc_full[0], uc_full[1], uc_full[2],
                S.step3_z_spec,
                S.step3_GA, S.step3_G2mA2, S.step3_K2,
                S.om2, S.fnm1,
                NK_full_i32, NX_half_i32, NZ_i32,
                divxz_f32,
                visc_f32,
                dt_f32,
                cnm1_f32,
            ),
        )

        _STEP3_BUILD_UC_KERNEL(
            (blocks,),
            (threads,),
            (
                S.om2,
                S.step3_invK2_sub,
                S.gamma,
                S.alfa,
                _np.float32(S.step3_inv_gamma0),
                S.scratch1,
                S.scratch2,
                NX_half_i32, NZ_i32,
            ),
        )

        uc_full[0, :NZ, :NX_half] = S.scratch1
        uc_full[1, :NZ, :NX_half] = S.scratch2

        S.cnm1 = float(S.cn)
        return

    # CPU / unfused path (kept intact)
    om2 = S.om2
    alfa = S.alfa
    gamma = S.gamma
    uc_full = S.uc_full

    Nbase = int(S.Nbase)
    NX_half = Nbase // 2
    NZ = Nbase

    visc = xp.float32(S.visc)
    dt = xp.float32(S.dt)
    cnm1 = xp.float32(S.cnm1)

    z_spec = S.step3_z_spec
    GA = S.step3_GA
    G2mA2 = S.step3_G2mA2
    K2 = S.step3_K2
    divxz = xp.float32(S.step3_divxz)

    U0 = uc_full[0, z_spec, :NX_half]
    U1 = uc_full[1, z_spec, :NX_half]
    U2 = uc_full[2, z_spec, :NX_half]

    FN = divxz * (GA * U0 + G2mA2 * U1 + K2 * U2)

    denom = (xp.float32(2.0) + cnm1 + dt * visc * K2)
    a = (xp.float32(2.0) - cnm1) / denom
    b = dt / denom

    om2[:] = a * om2 + b * FN
    S.fnm1[:] = FN

    invK2 = S.step3_invK2_sub
    g = gamma[:NZ].reshape(NZ, 1)
    a_k = alfa.reshape(1, NX_half)

    U0b = (1j * g) * invK2 * om2
    U1b = (-1j * a_k) * invK2 * om2

    uc_full[0, :NZ, :NX_half] = U0b.astype(xp.complex64, copy=False)
    uc_full[1, :NZ, :NX_half] = U1b.astype(xp.complex64, copy=False)

    S.cnm1 = float(S.cn)


# ---------------------------------------------------------------------------
# STEP2A — de-alias shuffle + inverse FFT + crop to compact UR (dnsCudaStep2A)
# ---------------------------------------------------------------------------

def dns_step2a(S: DnsState) -> None:
    xp = S.xp
    N = S.Nbase
    NX = S.NX
    NZ = S.NZ
    NX_full = S.NX_full
    NZ_full = S.NZ_full
    NK_full = S.NK_full

    UC = S.uc_full

    hi_start = N // 2
    hi_end = min(3 * N // 4, NK_full - 1)
    if hi_start <= hi_end:
        UC[0:2, :, hi_start:hi_end + 1] = xp.complex64(0.0 + 0.0j)

    halfN = N // 2
    k_max = min(halfN, NK_full)
    if k_max > 0:
        z_mid_start = halfN
        z_mid_end = N
        z_top_start = N
        z_top_end = N + halfN
        UC[0:2, z_top_start:z_top_end, :k_max] = UC[0:2, z_mid_start:z_mid_end, :k_max]
        UC[0:2, z_mid_start:z_mid_end, :k_max] = xp.complex64(0.0 + 0.0j)

    # Inverse FFT UC_full → UR_full
    vfft_full_inverse_uc_full_to_ur_full(S)

    off_x = (NX_full - NX) // 2
    off_z = (NZ_full - NZ) // 2

    if S.backend == "gpu" and _cp is not None:
        global _STEP2A_CROP_KERNEL
        if _STEP2A_CROP_KERNEL is None:
            crop_src = r'''
            extern "C" __global__
            void turbo_step2a_crop(
                const float* __restrict__ ur0,
                const float* __restrict__ ur1,
                float* __restrict__ ur,
                const int NX,
                const int NZ,
                const int NX_full,
                const int off_x,
                const int off_z
            ){
                int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                int n = NZ * NX;
                if (tid >= n) return;

                int z = tid / NX;
                int x = tid - z * NX;

                int src = (z + off_z) * NX_full + (x + off_x);
                float u0 = ur0[src];
                float u1 = ur1[src];

                int dst = (tid * 3);
                ur[dst + 0] = u0;
                ur[dst + 1] = u1;
                ur[dst + 2] = 0.0f;
            }
            '''
            _STEP2A_CROP_KERNEL = _cp.RawKernel(crop_src, "turbo_step2a_crop")

        threads = 256
        n = int(NZ) * int(NX)
        blocks = (n + threads - 1) // threads

        _STEP2A_CROP_KERNEL(
            (blocks,),
            (threads,),
            (
                S.ur_full[0],
                S.ur_full[1],
                S.ur,
                _np.int32(NX),
                _np.int32(NZ),
                _np.int32(NX_full),
                _np.int32(off_x),
                _np.int32(off_z),
            ),
        )
    else:
        S.ur[:, :, 0] = S.ur_full[0, off_z:off_z + N, off_x:off_x + N]
        S.ur[:, :, 1] = S.ur_full[1, off_z:off_z + N, off_x:off_x + N]
        S.ur[:, :, 2] = 0.0


# ---------------------------------------------------------------------------
# NEXTDT — CFL based timestep
# ---------------------------------------------------------------------------

def compute_cflm(S: DnsState) -> Any:
    xp = S.xp
    ur = S.ur

    # CFLM = max(|u|) + max(|v|), match your previous usage
    umax = xp.max(xp.abs(ur[:, :, 0])) + xp.max(xp.abs(ur[:, :, 1]))
    return umax * float(S.Nbase)


def next_dt(S: DnsState) -> None:
    CFLM = compute_cflm(S)

    if S.backend == "gpu":
        CFLM = float(CFLM)  # one sync here, but only when next_dt is called

    if CFLM <= 0.0 or S.dt <= 0.0:
        return

    CFL = CFLM * S.dt * PI
    S.cn = 0.8 + 0.2 * (S.cflnum / CFL)
    S.dt = S.dt * S.cn


# ===============================================================
# Python equivalent of dnsCudaDumpFieldAsPGMFull
# ===============================================================

def uc0_phys(S: DnsState) -> Any:
    xp = S.xp

    # build full-grid uc_tmp from low band uc_full[0]
    uc_tmp = xp.zeros((S.NZ_full, S.NK_full), dtype=xp.complex64)
    NX_half = S.NX // 2
    NZ = S.NZ
    NX_full = S.NX_full
    NZ_full = S.NZ_full
    NK_full = S.NK_full

    uc_tmp[:NZ, :NX_half] = S.uc_full[0, :NZ, :NX_half]

    # zero high modes and fill de-alias region similar to step2a
    hi_start = S.Nbase // 2
    hi_end = min(3 * S.Nbase // 4, NK_full - 1)
    if hi_start <= hi_end:
        uc_tmp[:, hi_start:hi_end + 1] = xp.complex64(0.0 + 0.0j)

    halfN = S.Nbase // 2
    k_max = min(halfN, NK_full)
    if k_max > 0:
        z_mid_start = halfN
        z_mid_end = S.Nbase
        z_top_start = S.Nbase
        z_top_end = S.Nbase + halfN
        uc_tmp[z_top_start:z_top_end, :k_max] = uc_tmp[z_mid_start:z_mid_end, :k_max]
        uc_tmp[z_mid_start:z_mid_end, :k_max] = xp.complex64(0.0 + 0.0j)

    z_mid = NZ
    if z_mid < NZ_full:
        uc_tmp[z_mid, :NX_half] = xp.complex64(0.0 + 0.0j)

    fft = S.fft

    if S.backend == "cpu":
        phys = fft.irfft2(uc_tmp, s=(NZ_full, NX_full), axes=(0, 1), overwrite_x=True, workers=S.fft_workers)
    else:
        phys = fft.irfft2(uc_tmp, s=(NZ_full, NX_full), axes=(0, 1), overwrite_x=True)

    phys *= (NZ_full * NX_full)
    return xp.asarray(phys, dtype=xp.float32)


def dns_om2_phys(S: DnsState) -> None:
    band = S.om2
    # (kept as-is in your original file)
    return


# ===============================================================
# RUN LOOP
# ===============================================================

def run_dns(
    N: int,
    Re: float,
    K0: float,
    steps: int,
    CFL: float,
    backend: str = "auto",
    workers: Optional[int] = None,
) -> None:

    if _cp is not None:
        try:
            dev = _cp.cuda.runtime.getDevice()
            props = _cp.cuda.runtime.getDeviceProperties(dev)
            name = props.get("name", b"").decode("utf-8", "ignore")
            print("  GPU: ", name)
        except Exception:
            pass

    print("--- RUN DNS ---")
    print(f" N   = {int(N)}")
    print(f" Re  = {float(Re)}")
    print(f" K0  = {float(K0)}")
    print(f" Steps = {int(steps)}")
    print(f" CFL  = {float(CFL)}")
    print(f" requested = {backend}")

    S = create_dns_state(N=N, Re=Re, K0=K0, CFL=CFL, backend=backend, seed=149, workers=workers)

    print(f" backend:  {S.backend}")
    print(f" workers (CPU): {S.fft_workers}")

    if S.backend == "gpu":
        print("FFT plan_mod: cupyx.scipy.fft")

    print(f"--- INITIALIZING SciPy/CuPy --- {time.strftime('%Y-%m-%d %H:%M')}")
    print(f" N={int(N)}, K0={int(K0)}, Re={float(Re)}")
    print("Generate isotropic random spectrum... (Numba)")

    # Initial inverse FFT and CALCOM
    t0 = time.perf_counter()
    print(" vfft_full_inverse_uc_full_to_ur_full(S)")
    vfft_full_inverse_uc_full_to_ur_full(S)

    print(" dns_calcom_from_uc_full(S)")
    dns_calcom_from_uc_full(S)

    print(f" effective = {S.backend} (xp = {'cupy' if S.backend == 'gpu' else 'scipy'})")
    print(f" DNS INITIALIZATION took {time.perf_counter() - t0:.3f} seconds")
    print(f" scipy.fft workers in-context = {1 if S.backend == 'gpu' else S.fft_workers}")

    # NEXTDT INIT
    CFLM = compute_cflm(S)
    if S.backend == "gpu":
        CFLM = float(CFLM)

    S.dt = float(S.cflnum / (CFLM * PI))
    S.cn = 1.0
    S.cnm1 = 1.0

    print(f" [NEXTDT INIT] CFLM={CFLM:11.4f} DT={S.dt:11.7f} CN={S.cn:11.7f}")
    print(f" Initial DT={S.dt:11.7f} CN={S.cn:11.7f}")

    # Loop
    t_start = time.perf_counter()
    for it in range(1, steps + 1):
        S.iteration = it

        dns_step2b(S)
        dns_step3(S, fuse=True)
        dns_step2a(S)

        next_dt(S)
        S.t += S.dt

        if it == 1 or it == steps or (it % 100 == 0):
            CFLM = compute_cflm(S)
            if S.backend == "gpu":
                CFLM = float(CFLM)
            print(f" ITERATION {it:6d} T={S.t:.10f} DT={S.dt:.8f} CN={S.cn:.8f} CFLM={CFLM:.6f}")

    elapsed = time.perf_counter() - t_start
    print(f" Elapsed CPU time for {steps} steps (s) = {elapsed: .6f}")
    print(f" Final T={S.t:.7f}  CN={S.cn:8.5f}  DT={S.dt:.9f}")
    fps = (steps / elapsed) if elapsed > 0 else 0.0
    print(f" FPS = {fps:.6f}")


# ===============================================================
# CLI
# ===============================================================

def main():
    _print_env_info()

    parser = argparse.ArgumentParser()
    parser.add_argument("N", nargs="?", type=int, default=512)
    parser.add_argument("Re", nargs="?", type=float, default=1e4)
    parser.add_argument("K0", nargs="?", type=float, default=10.0)
    parser.add_argument("steps", nargs="?", type=int, default=101)
    parser.add_argument("CFL", nargs="?", type=float, default=0.75)
    parser.add_argument("backend", nargs="?", type=str, default="auto")
    parser.add_argument("workers", nargs="?", type=int, default=None)
    args = parser.parse_args()

    run_dns(
        N=args.N,
        Re=args.Re,
        K0=args.K0,
        steps=args.steps,
        CFL=args.CFL,
        backend=args.backend.lower(),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()