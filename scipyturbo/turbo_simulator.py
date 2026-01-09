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
  • alfa[NX_half], gamma[NZ]           — wave-number vectors
  • Time loop     : STEP2B → STEP3 → STEP2A → NEXTDT, like dns_all.cu

Backends:
  • CPU:  SciPy
  • GPU:  CuPy (if installed); same API used via the `xp` alias.

This is now a faithful structural port of dns_all.cu:

  • dnsCudaPaoHostInit  → dns_pao_host_init
  • dnsCudaCalcom       → dns_calcom_from_uc_full
  • dnsCudaStep2A/2B/3  → dns_step2a / dns_step2b / dns_step3
  • next_dt_gpu         → next_dt

The 3/2 de-aliasing, Crank–Nicolson update, and spectral vorticity
formulas follow the CUDA kernels line-by-line.
"""
from contextlib import nullcontext
import time
import datetime as _dt
import math
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as _np

try:
    import cupy as _cp
    #_cp.show_config()
    dev = _cp.cuda.Device()
    props = _cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode("utf-8") if isinstance(props["name"], (bytes, bytearray)) else str(props["name"])
    print(f"\r\nGPU: {name}")  # e.g. "NVIDIA GeForce RTX 3090"
    _cflm_max_abs_sum = None
    if _cp is not None:
        _cflm_max_abs_sum = _cp.ReductionKernel(
            in_params="float32 u, float32 w, float32 inv_dx",
            out_params="float32 out",
            map_expr="(fabsf(u) + fabsf(w)) * inv_dx",
            reduce_expr="max(a, b)",
            post_map_expr="out = a",
            identity="0.0f",
            name="cflm_max_abs_sum_inv_dx",
        )
except Exception:  # CuPy is optional
    _cp = None
    print("\r\nCPU: CuPy not installed")

import numpy as np  # in addition to your existing _np alias, this is fine

# ===============================================================
# Optional Numba acceleration (CPU-only) for PAO initialization
#
# Pattern:
#   - one PAO kernel implementation (NumPy)
#   - one dispatcher name used by dns_pao_host_init
#   - if numba exists: dispatcher points to njit() version
#   - else: dispatcher points to pure-Python version
#
# IMPORTANT: no duplicate PAO kernel code.
# ===============================================================
try:
    import numba as _nb  # type: ignore
except Exception:
    _nb = None

def _pao_build_ur_and_stats_impl(
    N: int,
    NE: int,
    K0: np.float32,
    Re: np.float32,
    seed_init: int,
    alfa: np.ndarray,
    gamma: np.ndarray,
):
    """
    Shared PAO core (single source of truth):

      - Generate isotropic random spectrum (Fortran DO 500/510 loops)
      - Hermitian symmetry in Z (Fortran DO 600)
      - Compute averages A(1..7), E110, Q2, W2, VISC (Fortran DO 800/810)
      - Reshuffle (Fortran DO 1000 block)

    IMPORTANT:
      - Keep SERIAL loop order to preserve deterministic RNG call sequence for a given seed.
      - No prints inside: must work for Numba and non-Numba.
    """
    ND2 = N // 2
    NED2 = NE // 2
    PI = np.float32(3.14159265358979)

    # ------------------------------------------------------------------
    # Fortran LCG used in PAO (same constants as frand()).
    # ------------------------------------------------------------------
    IMM = 420029
    IT = 2017
    ID = 5011

    seed = int(seed_init)

    # ------------------------------------------------------------------
    # Fortran random vector RANVEC(97)
    # ------------------------------------------------------------------
    RANVEC = np.zeros(97, dtype=np.float32)

    # "warm-up" 97 calls
    for _ in range(97):
        seed = (seed * IMM + IT) % ID

    # fill RANVEC
    for i in range(97):
        seed = (seed * IMM + IT) % ID
        RANVEC[i] = np.float32(seed) / np.float32(ID)

    NORM = PI * K0 * K0

    # ------------------------------------------------------------------
    # Host spectral UR: complex field UR(kx,z,comp)
    # comp=0 → u1, comp=1 → u3 (Fortran components 1 and 2)
    #
    #   UR[x,z,c]  where  x ∈ [0..ND2-1], z ∈ [0..NE-1], c ∈ {0,1}
    # ------------------------------------------------------------------
    UR = np.zeros((ND2, NE, 2), dtype=np.complex64)

    # ------------------------------------------------------------------
    # Generate isotropic random spectrum (Fortran DO 500/510 loops)
    # ------------------------------------------------------------------
    for z in range(NE):
        gz = gamma[z]
        for x in range(NED2):
            # frand()
            seed = (seed * IMM + IT) % ID
            r = np.float32(seed) / np.float32(ID)

            # random_from_vec(r)
            idx = int(float(r) * 97.0)
            if idx < 0:
                idx = 0
            if idx > 96:
                idx = 96
            v = RANVEC[idx]
            RANVEC[idx] = r

            th = np.float32(2.0) * PI * v
            ARG = np.complex64(np.cos(th) + 1j * np.sin(th))

            ax = alfa[x]
            K2 = np.float32(ax * ax + gz * gz)
            K = np.float32(np.sqrt(K2)) if K2 > 0.0 else np.float32(0.0)

            if ax == 0.0:
                # ALFA(X) == 0: purely u1 mode
                UR[x, z, 1] = np.complex64(0.0 + 0.0j)

                ABSU2 = np.float32(np.exp(- (K / K0) * (K / K0)) / NORM)
                amp = np.float32(np.sqrt(ABSU2))
                UR[x, z, 0] = np.complex64(amp) * ARG
            else:
                denom = np.float32(1.0) + (gz * gz) / (ax * ax)
                ABSW2 = np.float32(np.exp(- (K / K0) * (K / K0)) / (denom * NORM))
                ampw = np.float32(np.sqrt(ABSW2))

                w = np.complex64(ampw) * ARG
                u = np.complex64(- (gz / ax)) * w  # -GAMMA/ALFA * UR(.,.,2)

                UR[x, z, 1] = w
                UR[x, z, 0] = u

    # Special zero modes (UR(1,1,1)=0, UR(1,1,2)=0 in 1-based Fortran)
    UR[0, 0, 0] = np.complex64(0.0 + 0.0j)
    UR[0, 0, 1] = np.complex64(0.0 + 0.0j)

    # ------------------------------------------------------------------
    # Hermitian symmetry in Z (Fortran DO 600)
    # ------------------------------------------------------------------
    for z in range(1, NED2):
        UR[0, NE - z, 0] = np.conj(UR[0, z, 0])
        UR[0, NE - z, 1] = np.conj(UR[0, z, 1])

    # Zero at Z=NED2+1 (index NED2 in 0-based)
    for x in range(ND2):
        UR[x, NED2, 0] = np.complex64(0.0 + 0.0j)
        UR[x, NED2, 1] = np.complex64(0.0 + 0.0j)

    # ------------------------------------------------------------------
    # Compute averages A(1..7), E110, Q2, W2, VISC (Fortran DO 800/810)
    # ------------------------------------------------------------------
    A1 = 0.0
    A2 = 0.0
    A3 = 0.0
    A4 = 0.0
    A5 = 0.0
    A6 = 0.0
    A7 = 0.0
    E110 = 0.0

    for x in range(ND2):
        x1 = (x == 0)
        ax2 = float(alfa[x]) * float(alfa[x])

        for z in range(NE):
            U1 = UR[x, z, 0]
            U3 = UR[x, z, 1]

            # Keep this explicit (Numba-friendly, avoids complex abs)
            u1u1 = float(U1.real) * float(U1.real) + float(U1.imag) * float(U1.imag)
            u3u3 = float(U3.real) * float(U3.real) + float(U3.imag) * float(U3.imag)

            gz2 = float(gamma[z]) * float(gamma[z])
            K2f = ax2 + gz2
            m = 1.0 if x1 else 2.0

            A1 += m * u1u1
            A2 += m * u3u3
            A3 += m * u1u1 * ax2
            A4 += m * u1u1 * gz2
            A5 += m * u3u3 * ax2
            A6 += m * u3u3 * gz2
            A7 += m * (u1u1 + u3u3) * K2f * K2f

            if x1:
                E110 += u1u1

    Q2 = A1 + A2
    W2 = A3 + A4 + A5 + A6
    #visc = np.sqrt((Q2 * Q2) / (float(Re) * W2))
    visc = 1.0 / float(Re)

    # ------------------------------------------------------------------
    # Reshuffle (Fortran DO 1000 block)
    # ------------------------------------------------------------------
    for comp in range(2):
        for z in range(NED2 - 1, -1, -1):
            for x in range(ND2):
                # UR(X,N-NED2+Z,I) = UR(X,Z+NED2,I)
                UR[x, N - NED2 + z, comp] = UR[x, z + NED2, comp]

                # IF(Z.LE.(N-NE)) UR(X,Z+NED2,I) = NOLL
                if z <= (N - NE - 1):
                    UR[x, z + NED2, comp] = np.complex64(0.0 + 0.0j)

    return UR, seed, np.float32(visc), Q2, W2, E110, A1, A2, A3, A4, A5, A6, A7


# Dispatcher used by dns_pao_host_init (Numba if available; else Python).
if _nb is not None:
    _pao_build_ur_and_stats = _nb.njit(cache=True)(_pao_build_ur_and_stats_impl)
else:
    _pao_build_ur_and_stats = _pao_build_ur_and_stats_impl

# ===============================================================
# ONLY FFT selection (CPU: scipy.fft, GPU: cupyx.scipy.fft)
# ===============================================================
try:
    import scipy.fft as _spfft  # type: ignore
except Exception:
    _spfft = None

try:
    import cupyx.scipy.fft as _cpfft  # type: ignore
except Exception:
    _cpfft = None


def _fft_mod_for_state(S: "DnsState"):
    """
    ONLY FFT selection:
      - CPU: scipy.fft
      - GPU: cupyx.scipy.fft (fallback to cupy.fft if cupyx.scipy.fft is unavailable)
    """
    if S.backend == "gpu":
        if _cpfft is not None:
            return _cpfft
        return S.xp.fft
    return _spfft

# ===============================================================
# Fortran-style random generator used in PAO (port of frand)
# ===============================================================
def frand(seed_list):
    """
    Port of the Fortran LCG used in PAO:

      IMM = 420029
      IT  = 2017
      ID  = 5011

      seed = (seed*IMM + IT) mod ID
      r    = seed / ID

    `seed_list` is a 1-element list to mimic Fortran SAVE/INTENT(INOUT).
    """
    IMM = 420029
    IT = 2017
    ID = 5011

    seed_list[0] = (seed_list[0] * IMM + IT) % ID
    return np.float32(seed_list[0] / ID)


# ---------------------------------------------------------------------------
# Backend selection: xp = np (CPU) or cp (GPU, if available)
# ---------------------------------------------------------------------------
def get_xp(backend: Literal["cpu", "gpu", "auto"] = "auto"):
    """
    backend = "gpu"  → force CuPy cuFFT (error if not available)
    backend = "cpu"  → force SciPy FFT
    backend = "auto" → use CuPy if available and a GPU is present, else SciPy
    """
    # Auto-select: try GPU first
    if backend == "auto":
        if _cp is not None:
            return _cp
        return _np

    # Explicit GPU / CPU selection
    if backend == "gpu":
        if _cp is None:
            raise RuntimeError("CuPy is not installed, but backend='gpu' was requested.")
        return _cp

    # backend == "cpu"
    return _np


# ---------------------------------------------------------------------------
# Fortran-style random generator used in PAO, port of frand(seed)
# ---------------------------------------------------------------------------

class Frand:
    """
    Port of the tiny LCG from dns_all.cu:

      IMM = 420029
      IT  = 2017
      ID  = 5011

      seed = (seed*IMM + IT) % ID
      r    = seed / ID
    """
    IMM = 420029
    IT = 2017
    ID = 5011

    def __init__(self, seed: int = 1):
        self.seed = int(seed)

    def __call__(self) -> float:
        self.seed = (self.seed * self.IMM + self.IT) % self.ID
        return float(self.seed) / float(self.ID)


# ===============================================================
# Python equivalent of dnsCudaDumpUCFullCsv
# ===============================================================
def dump_uc_full_csv(S: "DnsState", UC_full, comp: int):
    """
    CSV dumper compatible with step2a_debug.py, but for SoA layout:

        UC_full: (3, NZ_full, NK_full)  # [comp, z, kx]

    We print NX_full rows, NZ_full columns:
      - For i < 2*ND2:
          kx       = i // 2
          imag_row = (i & 1) == 1
          value    = Re or Im of UC_full[comp, z, kx]
      - For i >= 2*ND2, we print 0.0 (as in the debug helper).
    """
    N = S.Nbase
    NX_full = S.NX_full
    NZ_full = S.NZ_full
    NK_full = S.NK_full
    ND2 = N // 2

    # Bring data to NumPy on CPU for printing
    if S.backend == "gpu":
        UC_local = _np.asarray(UC_full.get())
    else:
        UC_local = _np.asarray(UC_full)

    for i in range(NX_full):
        row_vals = []

        use_mode = (i < 2 * ND2)
        if use_mode:
            kx = i // 2           # 0..ND2-1
            imag_row = (i & 1) == 1
        else:
            kx = None
            imag_row = False  # unused

        for z in range(NZ_full):
            if use_mode and kx < NK_full:
                # SoA layout: [comp, z, kx]
                v = UC_local[comp, z, kx]
                val = float(v.imag if imag_row else v.real)
            else:
                val = 0.0

            row_vals.append(f"{val:10.5f}")

        print(",".join(row_vals))

    print(f"[CSV] Wrote UC_full, {NX_full}x{NZ_full}, comp={comp}")


# ---------------------------------------------------------------------------
# DNS state  (Python equivalent of DnsDeviceState)
# ---------------------------------------------------------------------------

@dataclass
class DnsState:
    xp: any                 # scipy or cupy module
    backend: str            # "cpu" or "gpu"

    Nbase: int              # Fortran NX=NZ
    NX: int
    NZ: int
    NK: int

    NX_full: int
    NZ_full: int
    NK_full: int

    Re: float
    K0: float
    visc: float             # viscosity
    cflnum: float           # CFL target
    seed_init: int = 1
    fft_workers: int = 1

    # Cached FFT module (scipy.fft or cupyx.scipy.fft)
    fft: any = None

    # Reusable cuFFT plans (GPU only)
    fft_plan_rfft2_ur_full: any = None
    fft_plan_irfft2_uc01: any = None

    # Precomputed grid constants for CFL computation (dx==dz==2*pi/N)
    inv_dx: float = 0.0

    # CFL scratch to avoid per-step allocations (full 3/2 grid)
    cfl_tmp: any = None
    cfl_absw: any = None

    # Time integration
    t: float = 0.0
    dt: float = 0.0
    cn: float = 1.0
    cnm1: float = 0.0
    it: int = 0

    # Spectral wavenumber vectors
    alfa: any = None        # shape (NX_half,)
    gamma: any = None       # shape (NZ,)

    # Compact grid (AoS)
    ur: any = None          # shape (NZ, NX, 3), real
    uc: any = None          # shape (NZ, NK, 3), complex

    # Full 3/2 grid (SoA)
    ur_full: any = None     # shape (3, NZ_full, NX_full), real
    uc_full: any = None     # shape (3, NZ_full, NK_full), complex

    # Vorticity and non-linear history
    om2: any = None         # shape (NZ, NX_half), complex
    fnm1: any = None        # shape (NZ, NX_half), complex

    scratch1: any = None
    scratch2: any = None

    # Precomputed index grids for STEP3 (avoid per-step allocations)
    step3_z_indices: any = None
    step3_kx_indices: any = None
    step3_z_spec: any = None

    # STEP3 scratch buffers & constants (avoid per-step allocations)
    step3_uc1_th: any = None
    step3_uc2_th: any = None
    step3_uc3_th: any = None

    step3_K2: any = None          # float32 (NZ, NX_half)
    step3_GA: any = None          # float32 (NZ, NX_half)
    step3_G2mA2: any = None       # float32 (NZ, NX_half)
    step3_invK2_sub: any = None   # float32 (NZ, NX_half-1)

    step3_ARG: any = None         # float32 (NZ, NX_half)
    step3_DEN: any = None         # float32 (NZ, NX_half)
    step3_NUM: any = None         # complex64 (NZ, NX_half)

    step3_mask_ix0: any = None    # bool (NZ,)
    step3_inv_gamma0: any = None  # float32 (NZ,)  precomputed 1/gamma for ix=0 branch (0 where invalid)
    step3_divxz: any = None       # float32 scalar

    def sync(self):
        """For a CuPy backend, force synchronization at convenient checkpoints."""
        if self.backend == "gpu":
            self.xp.cuda.Stream.null.synchronize()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helper to create a DnsState (dnsCudaInit equivalent)
# ---------------------------------------------------------------------------

def create_dns_state(
    N: int = 8,
    Re: float = 1e5,
    K0: float = 100.0,
    CFL: float = 0.75,
    backend: Literal["cpu", "gpu", "auto"] = "auto",
    seed: int = 1,
) -> DnsState:
    xp = get_xp(backend)

    if backend == "auto":
        effective_backend = "gpu" if (_cp is not None and xp is _cp) else "cpu"
    else:
        effective_backend = backend

    Nbase = N
    NX = N
    NZ = N

    # Your CUDA code uses 3*N/2 (full 3/2 grid)
    NX_full = 3 * NX // 2
    NZ_full = 3 * NZ // 2
    NK_full = NX_full // 2 + 1

    # Compact spectral NK:
    # For the original PAO/Calcom you used NK = 3*N/4 + 1; we keep that here.
    NK = 3 * NX // 4 + 1

    NX_half = NX // 2
    visc = 0

    state = DnsState(
        xp=xp,
        backend=effective_backend,
        Nbase=Nbase,
        NX=NX,
        NZ=NZ,
        NK=NK,
        NX_full=NX_full,
        NZ_full=NZ_full,
        NK_full=NK_full,
        Re=Re,
        K0=K0,
        visc=visc,
        cflnum=CFL,
        seed_init=int(seed),
        fft_workers=4,
    )

    # Cache FFT module for the chosen backend (avoid per-call selection)
    state.fft = _fft_mod_for_state(state)
    if state.backend == "cpu" and state.fft is None:
        raise RuntimeError("scipy.fft import failed; CPU backend requires SciPy.")

    # Precompute inverse grid spacing (dx==dz==2*pi/N)
    state.inv_dx = float(state.Nbase) / (2.0 * math.pi)

    # Allocate arrays
    state.ur = xp.zeros((NZ, NX, 3), dtype=xp.float32)
    state.uc = xp.zeros((NZ, NK, 3), dtype=xp.complex64)

    state.ur_full = xp.zeros((3, NZ_full, NX_full), dtype=xp.float32)
    state.uc_full = xp.zeros((3, NZ_full, NK_full), dtype=xp.complex64)

    # CFL scratch buffers (full 3/2 grid) to avoid per-step temporaries
    state.cfl_tmp = xp.empty((NZ_full, NX_full), dtype=xp.float32)
    state.cfl_absw = xp.empty((NZ_full, NX_full), dtype=xp.float32)

    state.om2 = xp.zeros((NZ, NX_half), dtype=xp.complex64)
    state.fnm1 = xp.zeros((NZ, NX_half), dtype=xp.complex64)

    state.alfa = xp.zeros((NX_half,), dtype=xp.float32)
    state.gamma = xp.zeros((NZ,), dtype=xp.float32)

    # Reusable cuFFT plans (GPU only)
    if state.backend == "gpu":
        plan_mod = None
        if _cpfft is not None and hasattr(_cpfft, "get_fft_plan"):
            plan_mod = _cpfft

        if plan_mod is not None:
            # Forward: rfft2 on real UR_full over (z,x) axes
            state.fft_plan_rfft2_ur_full = plan_mod.get_fft_plan(
                state.ur_full, axes=(1, 2), value_type="R2C"
            )
            # Inverse: irfft2 on UC_full[0:2] over (z,x) axes back to real
            state.fft_plan_irfft2_uc01 = plan_mod.get_fft_plan(
                state.uc_full[0:2],
                shape=(state.NZ_full, state.NX_full),
                axes=(1, 2),
                value_type="C2R",
            )

        if plan_mod is None:
            print("FFT plan_mod: None")
        else:
            print(f"FFT plan_mod: {plan_mod.__name__}")
    else:
        print(f"FFT workers (CPU): {state.fft_workers}")

    # PAO-style initialization (dnsCudaPaoHostInit)
    dns_pao_host_init(state)

    # DT and CN will be initialized in run_dns via CFL (like CUDA)
    state.dt = 0.0
    state.cn = 1.0
    state.cnm1 = 0.0

    state.scratch1 = xp.zeros((NZ, NX_half), dtype=xp.complex64)
    state.scratch2 = xp.zeros((NZ, NX_half), dtype=xp.complex64)

    # Precompute index grids used in STEP3 (avoid per-step allocations)
    NZ = state.NZ
    NX_half = state.NX // 2
    state.step3_z_indices = xp.arange(NZ, dtype=xp.int32)
    state.step3_kx_indices = xp.arange(NX_half, dtype=xp.int32)
    NZ_half = NZ // 2
    zi = state.step3_z_indices
    state.step3_z_spec = xp.where(
        zi <= (NZ_half - 1),
        zi,
        zi + NZ_half,
    )

    # STEP3: preallocate gather buffers for UC low-k band (avoid advanced-index allocs)
    state.step3_uc1_th = xp.empty((NZ, NX_half), dtype=xp.complex64)
    state.step3_uc2_th = xp.empty((NZ, NX_half), dtype=xp.complex64)
    state.step3_uc3_th = xp.empty((NZ, NX_half), dtype=xp.complex64)

    # STEP3: precompute constant spectral grids (float32) used each step
    ax = state.alfa[None, :]          # (1, NX_half)
    gz = state.gamma[:, None]         # (NZ, 1)
    ax2 = ax * ax
    gz2 = gz * gz

    state.step3_K2 = (ax2 + gz2).astype(xp.float32, copy=False)
    state.step3_GA = (gz * ax).astype(xp.float32, copy=False)
    state.step3_G2mA2 = (gz2 - ax2).astype(xp.float32, copy=False)

    if NX_half > 1:
        state.step3_invK2_sub = (xp.float32(1.0) / (state.step3_K2[:, 1:] + xp.float32(1.0e-30))).astype(xp.float32, copy=False)
    else:
        state.step3_invK2_sub = xp.empty((NZ, 0), dtype=xp.float32)

    # STEP3: per-step float/complex scratch (avoid allocating ARG/DEN/NUM each step)
    state.step3_ARG = xp.empty((NZ, NX_half), dtype=xp.float32)
    state.step3_DEN = xp.empty((NZ, NX_half), dtype=xp.float32)
    state.step3_NUM = xp.empty((NZ, NX_half), dtype=xp.complex64)

    # ix=0 branch mask (Z>=1 and GAMMA!=0), constant
    state.step3_mask_ix0 = (state.step3_z_indices >= 1) & (xp.abs(state.gamma) > 0.0)

    # Precompute safe inv_gamma for ix=0 (avoid xp.divide(where=...) which CuPy rejects here)
    mask0 = xp.asarray(state.step3_mask_ix0)  # stays on-GPU for CuPy
    safe_gamma = xp.where(mask0, state.gamma, xp.float32(1.0))  # no zeros in denominator
    inv_gamma0 = (xp.float32(1.0) / safe_gamma).astype(xp.float32, copy=False)
    inv_gamma0 *= mask0.astype(xp.float32, copy=False)  # zero out invalid lanes

    state.step3_mask_ix0 = mask0
    state.step3_inv_gamma0 = inv_gamma0

    # DIVXZ = 1/(3NX/2 * 3NZ/2), constant for fixed N
    NX32 = xp.float32(1.5) * xp.float32(state.Nbase)
    NZ32 = xp.float32(1.5) * xp.float32(state.Nbase)
    state.step3_divxz = xp.float32(1.0) / (NX32 * NZ32)

    return state

# ===============================================================
# Python/Numpy/Scipy port of dnsCudaPaoHostInit, wired into DnsState
# ===============================================================
def dns_pao_host_init(S: DnsState):
    xp = S.xp
    N = S.NX
    NE = S.NZ
    ND2 = N // 2
    NED2 = NE // 2
    PI = np.float32(3.14159265358979)

    DXZ = np.float32(2.0) * PI / np.float32(N)
    K0 = np.float32(S.K0)
    NORM = PI * K0 * K0

    print("--- INITIALIZING SciPy/CuPy ---", _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f" N={N}, K0={int(K0)}, Re={S.Re:.4e}")

    # ------------------------------------------------------------------
    # Build ALFA(N/2) and GAMMA(N)  (Fortran DALFA, DGAMMA, E1, E3)
    # ------------------------------------------------------------------
    alfa = np.zeros(ND2, dtype=np.float32)
    gamma = np.zeros(NE, dtype=np.float32)

    E1 = np.float32(1.0)
    E3 = np.float32(1.0) / E1

    DALFA = np.float32(1.0) / E1
    DGAMMA = np.float32(1.0) / E3

    for x in range(NED2):
        alfa[x] = np.float32(x) * DALFA

    gamma[0] = np.float32(0.0)
    for z in range(1, NED2 + 1):
        gamma[z] = np.float32(z) * DGAMMA
        gamma[NE - z] = -gamma[z]

    # ------------------------------------------------------------------
    # Host spectral UR: complex field UR(kx,z,comp)
    # comp=0 → u1, comp=1 → u3 (Fortran components 1 and 2)
    #
    #   UR[x,z,c]  where  x ∈ [0..ND2-1], z ∈ [0..NE-1], c ∈ {0,1}
    # ------------------------------------------------------------------
    UR = np.zeros((ND2, NE, 2), dtype=np.complex64)

    # ------------------------------------------------------------------
    # Fortran random vector RANVEC(97)
    # ------------------------------------------------------------------
    seed = [int(S.seed_init)]  # mimics ISEED SAVE

    # ------------------------------------------------------------------
    # Generate isotropic random spectrum (Fortran DO 500/510 loops)
    # ------------------------------------------------------------------
    print(" Generate isotropic random spectrum... " + ("(Numba)" if (_nb is not None) else "(Python)"))

    UR, seed_out, visc_f32, Q2, W2, E110, A1, A2, A3, A4, A5, A6, A7 = _pao_build_ur_and_stats(
        N=N,
        NE=NE,
        K0=np.float32(S.K0),
        Re=np.float32(S.Re),
        seed_init=int(S.seed_init),
        alfa=alfa,
        gamma=gamma,
    )

    seed[0] = int(seed_out)
    S.visc = np.float32(visc_f32)

    # ------------------------------------------------------------------
    # Extra diagnostics (Fortran WRITE block)
    # ------------------------------------------------------------------
    visc = float(S.visc)
    EP = visc * W2
    De = 2.0 * visc * visc * A7
    KOL = (visc * visc * visc / EP) ** 0.25
    NLAM = 0.0
    if E110 != 0.0:
        NLAM = 2.0 * A1 / E110

    a11 = 2.0 * A1 / Q2 - 1.0
    e11 = 2.0 * (A3 + A4) / W2 - 1.0
    tscale = 0.5 * Q2 / EP
    dxKol = float(DXZ) / KOL
    Lux = 2.0 * math.pi / math.sqrt(2.0 * A1 / A3)
    Luz = 2.0 * math.pi / math.sqrt(2.0 * A1 / A4)
    Lwx = 2.0 * math.pi / math.sqrt(2.0 * A2 / A5)
    Lwz = 2.0 * math.pi / math.sqrt(2.0 * A2 / A6)
    Ceps2 = 0.5 * Q2 * De / (EP * EP)

    # Print diagnostics exactly like the CUDA/Fortran version
    print(f" N           = {N:.8g}")
    print(f" Reynolds n. = {float(S.Re):.8g}")
    print(f" K0          = {K0:.8g}")
    print(f" Energy      = {Q2:.8g}")
    print(f" WiWi        = {W2:.8g}")
    #print(f" Epsilon     = {EP:.8g}")
    #print(f" a11         = {a11:.8g}")
    #print(f" e11         = {e11:.8g}")
    print(f" Time scale  = {tscale:.8g}")
    print(f" Kolmogorov  = {KOL:.8g}")
    print(f" Viscosity   = {visc:.8g}")
    print(f" dx/Kol.     = {dxKol:.8g}")
    #print(f" 2Pi/Nlamda  = {NLAM:.8g}")
    #print(f" 2Pi/Lux     = {Lux:.8g}")
    #print(f" 2Pi/Luz     = {Luz:.8g}")
    #print(f" 2Pi/Lwx     = {Lwx:.8g}")
    #print(f" 2Pi/Lwz     = {Lwz:.8g}")
    #print(f" Deps.       = {De:.8g}")
    #print(f" Ceps2       = {Ceps2:.8g}")
    #print(f" E1          = {float(E1):.8g}")
    #print(f" E3          = {float(E3):.8g}")
    print(f" PAO seed    = {seed[0]:.8g}")

    # ------------------------------------------------------------------
    # Scatter spectral UR → compact UC(kx,z,comp) buffer (current grid)
    #   UC: (NK, NE, 3) on host, but DnsState.uc is (NZ, NK, 3) in xp
    # ------------------------------------------------------------------
    NK = S.NK
    #print(f" UC_host = np.zeros(({NK}, {NE}, 3), dtype=np.complex64)")
    UC_host = np.zeros((NK, NE, 3), dtype=np.complex64)  # only comp 0,1 used

    for z in range(NE):
        for x in range(ND2):
            for c in range(2):
                UC_host[x, z, c] = UR[x, z, c]

    # ALSO build full 3/2-grid UC_full (Fortran-like layout)
    NK_full = S.NK_full
    NZ_full = S.NZ_full

    #print(f" UC_full_host = np.zeros(({NK_full}, {NZ_full}, 3), dtype=np.complex64)")
    UC_full_host = np.zeros((NK_full, NZ_full, 3), dtype=np.complex64)
    for z in range(NE):
        for x in range(ND2):
            for c in range(2):
                UC_full_host[x, z, c] = UR[x, z, c]

    print(f" PAO INITIALIZATION OK. VISC={float(S.visc):.8g}")

    # ------------------------------------------------------------------
    # Move alfa/gamma/UC/UC_full into DnsState (xp backend, SoA layout)
    # ------------------------------------------------------------------
    S.alfa = xp.asarray(alfa, dtype=xp.float32)
    S.gamma = xp.asarray(gamma, dtype=xp.float32)

    # compact UC: host (NK, NE, 3) → xp (NZ, NK, 3) with axes swap
    UC_xp = xp.asarray(UC_host)
    S.uc[...] = xp.transpose(UC_xp, (1, 0, 2))  # (NE,NK,3) == (NZ,NK,3)

    # full UC_full: host (NK_full, NZ_full, 3) → xp (3, NZ_full, NK_full)
    UC_full_xp = xp.asarray(UC_full_host)
    S.uc_full[...] = xp.transpose(UC_full_xp, (2, 1, 0))  # (3,NZ_full,NK_full)

    # ------------------------------------------------------------------
    # Build initial UR_full & om2 from UC_full (for the rest of the solver)
    # ------------------------------------------------------------------
    # Inverse transform UC_full → UR_full for diagnostics / STEP2B input
    print(f" vfft_full_inverse_uc_full_to_ur_full(S)")
    vfft_full_inverse_uc_full_to_ur_full(S)

    # Spectral vorticity from UC_full, like dnsCudaCalcom
    #print(f" dns_calcom_from_uc_full(S)")
    dns_calcom_from_uc_full(S)

    # No history yet
    S.fnm1[...] = xp.zeros_like(S.om2)


# ---------------------------------------------------------------------------
# FFT helpers (vfft_full_* equivalents)
# ---------------------------------------------------------------------------

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
    scale = xp.float32(S.NZ_full * S.NX_full)
    xp.multiply(ur01, scale, out=S.ur_full[0:2, :, :])
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
            UC = fft.rfft2(UR, s=(S.NZ_full, S.NX_full), axes=(1, 2), plan=plan, overwrite_x=True)
        else:
            UC = fft.rfft2(UR, s=(S.NZ_full, S.NX_full), axes=(1, 2), overwrite_x=True)

    # Assign back; uc_full is complex64, assignment will down-cast if needed
    S.uc_full[...] = UC


# ---------------------------------------------------------------------------
# CALCOM — spectral vorticity from UC_full (dnsCudaCalcom)
# ---------------------------------------------------------------------------

def dns_calcom_from_uc_full(S: DnsState) -> None:
    """
    Python/xp port of dnsCudaCalcom:

      OM2(ix,iz) = i * [ GAMMA(iz)*UC1(ix,iz) - ALFA(ix)*UC2(ix,iz) ]

    Uses:
      S.uc_full : (3, NZ_full, NK_full)  [comp,z,kx]
      S.alfa    : (NX_half,)
      S.gamma   : (NZ,)
    Writes:
      S.om2     : (NZ, NX_half)
    """
    xp = S.xp

    Nbase = int(S.Nbase)
    NX_full = int(S.NX_full)
    NZ_full = int(S.NZ_full)
    NK_full = int(S.NK_full)

    NX_half = Nbase // 2
    NZ = Nbase

    alfa_1d = S.alfa.astype(xp.float32)      # (NX_half,)
    gamma_1d = S.gamma.astype(xp.float32)     # (NZ,)

    # UC_full layout: [comp, z, kx]
    uc1_full = S.uc_full[0]                   # (NZ_full, NK_full)
    uc2_full = S.uc_full[1]                   # (NZ_full, NK_full)

    # We only use the first NZ rows and NX_half kx-modes
    uc1 = uc1_full[:NZ, :NX_half]             # (NZ, NX_half)
    uc2 = uc2_full[:NZ, :NX_half]             # (NZ, NX_half)

    ax = alfa_1d[None, :]                     # (1, NX_half)
    gz = gamma_1d[:, None]                    # (NZ, 1)

    # diff = GAMMA*UC1 - ALFA*UC2
    diff = gz * uc1 - ax * uc2                # (NZ, NX_half), complex

    # om = i * diff = (-Im(diff), Re(diff))
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
         (VRFFTF + VCFFTF in Fortran)
      3) Zero UC(X,NZ+1,I) for X<=NX/2, I=1..3
    """
    xp = S.xp

    # Geometry on the full 3/2 grid
    N = S.Nbase          # NX = NZ = Nbase (Fortran NX,NZ)
    NX_full = S.NX_full        # 3*N/2
    NZ_full = S.NZ_full        # 3*N/2
    NK_full = S.NK_full        # 3*N/4+1

    UR = S.ur_full
    UC = S.uc_full

    u = UR[0]   # (NZ_full, NX_full)
    w = UR[1]   # (NZ_full, NX_full)

    # Use a single elementwise GPU kernel to write all three products in one pass
    if S.backend == "gpu":
        global _STEP2B_MUL3_KERNEL
        if _STEP2B_MUL3_KERNEL is None:
            _STEP2B_MUL3_KERNEL = xp.ElementwiseKernel(
                "T u, T w",
                "T uw, T uu, T ww",
                "uw = u * w; uu = u * u; ww = w * w;",
                "turbo_step2b_mul3",
            )
        _STEP2B_MUL3_KERNEL(u, w, UR[2], UR[0], UR[1])
    else:
        # Use in-place multiplies to avoid temporaries
        xp.multiply(u, w, out=UR[2])  # u * w
        xp.multiply(u, u, out=UR[0])  # u^2
        xp.multiply(w, w, out=UR[1])  # w^2

    vfft_full_forward_ur_full_to_uc_full(S)

    NX_half = N // 2
    NZ = N
    z_mid = NZ

    kx_max = min(NX_half, NK_full)

    if z_mid < NZ_full and kx_max > 0:
        UC[0:3, z_mid, 0:kx_max] = xp.complex64(0.0 + 0.0j)


# ---------------------------------------------------------------------------
# STEP3 — vorticity update using om2 & fnm1
# ---------------------------------------------------------------------------
def dns_step3(S: DnsState, fuse: bool = True) -> None:
    xp = S.xp
    global _STEP3_UPDATE_KERNEL, _STEP3_BUILD_UC_KERNEL
    # Fast GPU path: fuse the heavy STEP3 arithmetic into a couple of custom kernels.
    # This avoids a large number of small elementwise launches (dominant in Scalene).
    if S.backend == "gpu" and _cp is not None and fuse:

        # Compile once per process
        if _STEP3_UPDATE_KERNEL is None:
            _STEP3_UPDATE_KERNEL = _cp.RawKernel(r'''
            #include <cupy/complex.cuh>
            extern "C" __global__
            void turbo_step3_update(
                const complex<float>* uc0, const complex<float>* uc1, const complex<float>* uc2,
                const int* z_spec,
                const float* GA, const float* G2mA2, const float* K2,
                complex<float>* om2, complex<float>* fnm1,
                int NK_full, int NX_half, int NZ,
                float divxz, float visc, float dt, float cnm1
            ) {
                int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                int n = NZ * NX_half;
                if (idx >= n) return;

                int z = idx / NX_half;
                int k = idx - z * NX_half;

                int zsrc = z_spec[z];

                complex<float> u0 = uc0[zsrc * NK_full + k];
                complex<float> u1 = uc1[zsrc * NK_full + k];
                complex<float> u2v = uc2[zsrc * NK_full + k];

                float ga = GA[idx];
                float g2ma2 = G2mA2[idx];

                complex<float> fn = (u0 - u1) * ga + u2v * g2ma2;
                fn *= divxz;

                float arg = K2[idx] * (0.5f * visc * dt);
                float den = 1.0f + arg;
                float invden = 1.0f / den;

                float c2 = 0.5f * dt * (2.0f + cnm1);
                float c3 = -0.5f * dt * cnm1;

                complex<float> om = om2[idx];
                complex<float> fprev = fnm1[idx];

                complex<float> num = om - om * arg + fn * c2 + fprev * c3;

                om2[idx] = num * invden;
                fnm1[idx] = fn;
            }
            ''', "turbo_step3_update")

        if _STEP3_BUILD_UC_KERNEL is None:
            _STEP3_BUILD_UC_KERNEL = _cp.RawKernel(r'''
            #include <cupy/complex.cuh>
            extern "C" __global__
            void turbo_step3_build_uc01(
                const complex<float>* om2,
                const float* invK2_sub,
                const float* gamma,
                const float* alfa,
                const float* inv_gamma0,
                complex<float>* out1,
                complex<float>* out2,
                int NX_half, int NZ
            ) {
                int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                int n = NZ * NX_half;
                if (idx >= n) return;

                int z = idx / NX_half;
                int k = idx - z * NX_half;

                complex<float> om = om2[idx];

                complex<float> o1(0.0f, 0.0f);
                complex<float> o2(0.0f, 0.0f);

                if (k == 0) {
                    float invg = inv_gamma0[z];
                    // (-i) * (a + i b) = b - i a
                    o1 = complex<float>(om.imag(), -om.real()) * invg;
                    o2 = complex<float>(0.0f, 0.0f);
                } else {
                    float invk2 = invK2_sub[z * (NX_half - 1) + (k - 1)];
                    float gz = gamma[z];
                    float ax = alfa[k];

                    // (-i) * om
                    complex<float> m1(om.imag(), -om.real());
                    // ( i) * om
                    complex<float> m2(-om.imag(), om.real());

                    o1 = m1 * (invk2 * gz);
                    o2 = m2 * (invk2 * ax);
                }

                out1[idx] = o1;
                out2[idx] = o2;
            }
            ''', "turbo_step3_build_uc01")

        # Geometry and constants
        Nbase = int(S.Nbase)
        NX_half = Nbase // 2
        NZ = Nbase

        uc_full = S.uc_full
        NK_full = int(S.NK_full)

        threads = 256
        n = NZ * NX_half
        blocks = (n + threads - 1) // threads

        # IMPORTANT: RawKernel scalar args must match the C signature types.
        # On 64-bit Python, passing plain Python ints/floats will typically be int64/float64,
        # which corrupts the kernel argument packing (and silently breaks the physics).
        NK_full_i32 = _np.int32(NK_full)
        NX_half_i32 = _np.int32(NX_half)
        NZ_i32 = _np.int32(NZ)
        divxz_f32 = _np.float32(S.step3_divxz)
        visc_f32 = _np.float32(S.visc)
        dt_f32 = _np.float32(S.dt)
        cnm1_f32 = _np.float32(S.cnm1)

        # UPDATE: compute FN, update om2, update fnm1
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

        # BUILD: out1/out2 (scratch1/2) from updated om2
        _STEP3_BUILD_UC_KERNEL(
            (blocks,),
            (threads,),
            (
                S.om2,
                S.step3_invK2_sub,
                S.gamma,
                S.alfa,
                S.step3_inv_gamma0,
                S.scratch1,
                S.scratch2,
                NX_half_i32, NZ_i32,
            ),
        )# Scatter into uc_full low-k band (strided in NK_full, keep the simple slice assign)
        uc_full[0, :NZ, :NX_half] = S.scratch1
        uc_full[1, :NZ, :NX_half] = S.scratch2

        S.cnm1 = float(S.cn)
        return

    om2 = S.om2
    fnm1 = S.fnm1
    alfa = S.alfa
    gamma = S.gamma
    uc_full = S.uc_full

    Nbase = int(S.Nbase)
    NX_half = Nbase // 2
    NZ = Nbase

    visc = xp.float32(S.visc)
    dt = xp.float32(S.dt)
    cn = xp.float32(S.cn)
    cnm1 = xp.float32(S.cnm1)

    z_spec = S.step3_z_spec
    divxz = S.step3_divxz
    GA = S.step3_GA
    G2mA2 = S.step3_G2mA2
    K2 = S.step3_K2

    uc0_low = uc_full[0, :, :NX_half]
    uc1_low = uc_full[1, :, :NX_half]
    uc2_low = uc_full[2, :, :NX_half]

    uc1_th = S.step3_uc1_th
    uc2_th = S.step3_uc2_th
    uc3_th = S.step3_uc3_th
    xp.take(uc0_low, z_spec, axis=0, out=uc1_th)
    xp.take(uc1_low, z_spec, axis=0, out=uc2_th)
    xp.take(uc2_low, z_spec, axis=0, out=uc3_th)

    tmp_FN = S.scratch1
    tmp_c = S.scratch2
    xp.subtract(uc1_th, uc2_th, out=tmp_FN)
    xp.multiply(tmp_FN, GA, out=tmp_FN)
    xp.multiply(uc3_th, G2mA2, out=tmp_c)
    xp.add(tmp_FN, tmp_c, out=tmp_FN)
    tmp_FN *= divxz

    VT = xp.float32(0.5) * visc * dt
    ARG = S.step3_ARG
    DEN = S.step3_DEN
    xp.multiply(K2, VT, out=ARG)
    xp.add(ARG, xp.float32(1.0), out=DEN)

    c2 = xp.float32(0.5) * dt * (xp.float32(2.0) + cnm1)
    c3 = -xp.float32(0.5) * dt * cnm1

    NUM = S.step3_NUM
    NUM[...] = om2
    xp.multiply(om2, ARG, out=tmp_c)
    NUM -= tmp_c

    xp.multiply(tmp_FN, c2, out=tmp_c)
    NUM += tmp_c
    xp.multiply(fnm1, c3, out=tmp_c)
    NUM += tmp_c

    xp.divide(NUM, DEN, out=om2)

    fnm1[...] = tmp_FN

    out1 = S.scratch1
    out2 = S.scratch2
    out1[...] = 0
    out2[...] = 0

    if NX_half > 1:
        invK2_sub = S.step3_invK2_sub

        out1[:, 1:] = om2[:, 1:]
        out1[:, 1:] *= invK2_sub
        out1[:, 1:] *= gamma[:, None]
        out1[:, 1:] *= xp.complex64(-1.0j)

        out2[:, 1:] = om2[:, 1:]
        out2[:, 1:] *= invK2_sub
        out2[:, 1:] *= alfa[1:][None, :]
        out2[:, 1:] *= xp.complex64(1.0j)

    # GPU-optimized ix=0 branch: no fancy indexing gather/scatter
    out1[:, 0] = 0
    out1[:, 0] = xp.complex64(-1.0j) * om2[:, 0] * S.step3_inv_gamma0

    uc_full[0, :NZ, :NX_half] = out1
    uc_full[1, :NZ, :NX_half] = out2

    S.cnm1 = float(cn)


# ===============================================================
# STEP2A core (dealias + reshuffle + inverse FFT)
# ===============================================================
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

def compute_cflm(S: DnsState):
    xp = S.xp
    NX3D2 = S.NX_full
    NZ3D2 = S.NZ_full

    u = S.ur_full[0, :NZ3D2, :NX3D2]
    w = S.ur_full[1, :NZ3D2, :NX3D2]

    if S.backend == "gpu" and _cflm_max_abs_sum is not None:
        CFLM = _cflm_max_abs_sum(u, w, xp.float32(S.inv_dx))  # GPU scalar (already scaled)
        return CFLM

    # CPU (or fallback): keep current code path
    tmp = S.cfl_tmp[:NZ3D2, :NX3D2]
    absw = S.cfl_absw[:NZ3D2, :NX3D2]
    xp.abs(u, out=tmp)
    xp.abs(w, out=absw)
    xp.add(tmp, absw, out=tmp)
    CFLM = xp.max(tmp) * S.inv_dx
    return float(CFLM) if S.backend == "cpu" else CFLM

def next_dt(S: DnsState) -> None:
    PI = math.pi
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
def dump_field_as_pgm_full(S: DnsState, comp: int, filename: str) -> None:
    NX_full = S.NX_full
    NZ_full = S.NZ_full

    if S.backend == "gpu":
        ur_full_host = _np.asarray(S.ur_full.get(), dtype=_np.float32)
    else:
        ur_full_host = _np.asarray(S.ur_full, dtype=_np.float32)

    field = ur_full_host[comp, :, :]

    minv = float(field.min())
    maxv = float(field.max())

    try:
        f = open(filename, "wb")
    except OSError as e:
        print(f"[DUMP] fopen failed for {filename!r}: {e}")
        return

    header = f"P5\n{NX_full} {NZ_full}\n255\n"
    f.write(header.encode("ascii"))

    rng = maxv - minv

    if abs(rng) <= 1.0e-12:
        c = bytes([128])
        row = c * NX_full
        for _ in range(NZ_full):
            f.write(row)
    else:
        for j in range(NZ_full):
            for i in range(NX_full):
                val = float(field[j, i])
                norm = (val - minv) / rng
                pixf = 1.0 + norm * 254.0
                pix = int(pixf + 0.5)
                if pix < 1:
                    pix = 1
                if pix > 255:
                    pix = 255
                f.write(bytes([pix]))

    f.close()
    print(f"[DUMP] Wrote {filename} (PGM, {NX_full}x{NZ_full}, "
          f"comp={comp}, min={minv:g}, max={maxv:g})")


# ---------------------------------------------------------------------------
# Helpers for visualization fields (energy, vorticity, streamfunction)
# ---------------------------------------------------------------------------

def dns_kinetic(S: DnsState) -> None:
    xp = S.xp

    u = S.ur_full[0, :, :]
    w = S.ur_full[1, :, :]

    ke = xp.sqrt(u * u + w * w)
    S.ur_full[2, :, :] = ke.astype(xp.float32)


def _spectral_band_to_phys_full_grid(S: DnsState, band) -> any:
    xp = S.xp

    N = S.Nbase
    NX_full = S.NX_full
    NZ_full = S.NZ_full
    NK_full = S.NK_full

    NX_half = N // 2
    NZ = N

    uc_tmp = xp.zeros((NZ_full, NK_full), dtype=xp.complex64)
    uc_tmp[:NZ, :NX_half] = band

    hi_start = N // 2
    hi_end = min(3 * N // 4, NK_full - 1)
    if hi_start <= hi_end:
        uc_tmp[:, hi_start:hi_end + 1] = xp.complex64(0.0 + 0.0j)

    halfN = N // 2
    k_max = min(halfN, NK_full)

    if k_max > 0:
        z_mid_start = halfN
        z_mid_end = halfN + halfN
        z_top_start = N
        z_top_end = N + halfN

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
    phys = _spectral_band_to_phys_full_grid(S, band)
    S.ur_full[2, :, :] = phys


def dns_stream_func(S: DnsState) -> None:
    xp = S.xp

    N = S.Nbase
    NX_half = N // 2
    NZ = N

    alfa_1d = S.alfa.astype(xp.float32)
    gamma_1d = S.gamma.astype(xp.float32)

    ax = alfa_1d[None, :]
    gz = gamma_1d[:, None]

    K2 = ax * ax + gz * gz
    K2 = K2 + xp.float32(1.0e-30)

    phi_hat = S.om2 / K2
    phys = _spectral_band_to_phys_full_grid(S, phi_hat)
    S.ur_full[2, :, :] = phys


# ---------------------------------------------------------------------------
# Main driver (Python version of main in dns_all.cu)
# ---------------------------------------------------------------------------
def run_dns(
    N: int = 8,
    Re: float = 100,
    K0: float = 10.0,
    STEPS: int = 2,
    CFL: float = 0.75,
    backend: Literal["cpu", "gpu", "auto"] = "auto",
) -> None:
    print("--- RUN DNS ---")
    print(f" N   = {N}")
    print(f" Re  = {Re}")
    print(f" K0  = {K0}")
    print(f" Steps = {STEPS}")
    print(f" CFL  = {CFL}")
    print(f" requested = {backend}")

    start =  time.perf_counter()
    S = create_dns_state(N=N, Re=Re, K0=K0, CFL=CFL, backend=backend)
    print(f" effective = {S.backend} (xp = {'cupy' if S.backend == 'gpu' else 'scipy'})")
    elapsed = time.perf_counter() - start
    print(f" DNS INITIALIZATION took {elapsed:.3f} seconds")

    if S.backend == "cpu" and _spfft is not None and S.fft_workers > 1:
        fft_ctx = _spfft.set_workers(S.fft_workers)
    else:
        fft_ctx = nullcontext()

    with fft_ctx:
        if _spfft is not None:
            print(f" scipy.fft workers in-context = {_spfft.get_workers()}")
        else:
            print(" scipy.fft workers in-context = n/a (gpu or scipy.fft missing)")

        dns_step2a(S)

        CFLM = compute_cflm(S)
        if S.backend == "gpu":
            CFLM0 = float(CFLM)  # one sync here at init (fine)
        else:
            CFLM0 = float(CFLM)

        S.dt = S.cflnum / (CFLM0 * math.pi)
        S.cn = 1.0
        S.cnm1 = 0.0
        S.t = 0.0

        print(f" [NEXTDT INIT] CFLM={CFLM0:11.4f} DT={S.dt:11.7f} CN={S.cn:11.7f}")
        print(f" Initial DT={S.dt:11.7f} CN={S.cn:11.7f}")

        S.sync()
        t0 = time.perf_counter()

        for it in range(1, STEPS + 1):
            S.it = it
            dt_old = S.dt
            dns_step2b(S)
            dns_step3(S)
            dns_step2a(S)
            if (it % 100) == 0 or it == 1 or it == STEPS:
                next_dt(S)
                print(f" ITERATION {it:6d} T={S.t:12.10f} DT={S.dt:10.8f} CN={S.cn:10.8f} CFLM={float(compute_cflm(S)):.6f}")
            S.t += dt_old

        S.sync()
        t1 = time.perf_counter()

        elap = t1 - t0
        fps = (STEPS / elap) if elap > 0 else 0.0

        print(f" Elapsed CPU time for {STEPS} steps (s) = {elap:.8g}")
        print(f" Final T={S.t:.8g}  CN={S.cn:.8g}  DT={S.dt:.8g}")
        print(f" FPS = {fps:.8g}")

def main():
    args = sys.argv[1:]
    N = int(args[0]) if len(args) > 0 else 512
    Re = float(args[1]) if len(args) > 1 else 10000
    K0 = float(args[2]) if len(args) > 2 else 10.0
    STEPS = int(args[3]) if len(args) > 3 else 101
    CFL = float(args[4]) if len(args) > 4 else 0.75

    BACK = args[5].lower() if len(args) > 5 else "auto"
    if BACK not in ("cpu", "gpu", "auto"):
        BACK = "auto"

    run_dns(N=N, Re=Re, K0=K0, STEPS=STEPS, CFL=CFL, backend=BACK)


if __name__ == "__main__":
    main()