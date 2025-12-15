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
import math
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as _np

try:
    print(" Checking CuPy...")
    import cupy as _cp
    _cp.show_config()
except Exception:  # CuPy is optional
    _cp = None
    print(" CuPy not installed")

import numpy as np  # in addition to your existing _np alias, this is fine

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

    print(f" backend:  {backend}")
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

    # Viscosity: in your original Fortran/CUDA this is computed more carefully
    # from PAO/Calcom; here we keep a standard DNS-ish scaling ν ~ 1/Re.
    visc = 1.0 / float(Re)

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
    print(f" workers (CPU): {state.fft_workers}")

    # Cache FFT module for the chosen backend (avoid per-call selection)
    state.fft = _fft_mod_for_state(state)

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

    # ix=0 branch mask (Z>=2 and GAMMA!=0), constant
    state.step3_mask_ix0 = (state.step3_z_indices >= 1) & (xp.abs(state.gamma) > 0.0)

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
    RANVEC = np.zeros(97, dtype=np.float32)

    # "warm-up" 97 calls
    for _ in range(99):
        frand(seed)

    # fill RANVEC
    for i in range(97):
        RANVEC[i] = frand(seed)

    def random_from_vec(r: np.float32) -> np.float32:
        idx = int(float(r) * 97.0)
        if idx < 0:
            idx = 0
        if idx > 96:
            idx = 96
        v = RANVEC[idx]
        RANVEC[idx] = r
        return v

    # ------------------------------------------------------------------
    # Generate isotropic random spectrum (Fortran DO 500/510 loops)
    # ------------------------------------------------------------------
    for z in range(NE):
        gz = gamma[z]
        for x in range(NED2):
            r = frand(seed)
            th = np.float32(2.0) * PI * random_from_vec(r)

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
    UR[:, NED2, 0] = 0.0 + 0.0j
    UR[:, NED2, 1] = 0.0 + 0.0j

    # ------------------------------------------------------------------
    # Compute averages A(1..7), E110, Q2, W2, VISC (Fortran DO 800/810)
    # ------------------------------------------------------------------
    A1 = A2 = A3 = A4 = A5 = A6 = A7 = 0.0
    E110 = 0.0

    for x in range(ND2):
        x1 = (x == 0)
        ax2 = float(alfa[x]) * float(alfa[x])

        for z in range(NE):
            U1 = UR[x, z, 0]
            U3 = UR[x, z, 1]

            u1u1 = float(np.abs(U1) ** 2)
            u3u3 = float(np.abs(U3) ** 2)

            gz2 = float(gamma[z]) * float(gamma[z])
            K2 = ax2 + gz2
            m = 1.0 if x1 else 2.0

            A1 += m * u1u1
            A2 += m * u3u3
            A3 += m * u1u1 * ax2
            A4 += m * u1u1 * gz2
            A5 += m * u3u3 * ax2
            A6 += m * u3u3 * gz2
            A7 += m * (u1u1 + u3u3) * K2 * K2

            if x1:
                E110 += u1u1

    Q2 = A1 + A2
    W2 = A3 + A4 + A5 + A6
    visc = math.sqrt(Q2 * Q2 / (float(S.Re) * W2))

    S.visc = np.float32(visc)

    # ------------------------------------------------------------------
    # Extra diagnostics (Fortran WRITE block)
    # ------------------------------------------------------------------
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
    print(f" N           ={N:12.0f}")
    print(f" Reynolds n. ={float(S.Re):12.1f}")
    print(f" K0          ={K0:12.0f}")
    print(f" Energy      ={Q2:12.4f}")
    print(f" WiWi        ={W2:12.4f}")
    print(f" Epsilon     ={EP:12.4f}")
    print(f" a11         ={a11:12.4f}")
    print(f" e11         ={e11:12.4f}")
    print(f" Time scale  ={tscale:12.4f}")
    print(f" Kolmogorov  ={KOL:12.4f}")
    print(f" Viscosity   ={visc:12.4f}")
    print(f" dx/Kol.     ={dxKol:12.4f}")
    print(f" 2Pi/Nlamda  ={NLAM:12.4f}")
    print(f" 2Pi/Lux     ={Lux:12.4f}")
    print(f" 2Pi/Luz     ={Luz:12.4f}")
    print(f" 2Pi/Lwx     ={Lwx:12.4f}")
    print(f" 2Pi/Lwz     ={Lwz:12.4f}")
    print(f" Deps.       ={De:12.4f}")
    print(f" Ceps2       ={Ceps2:12.4f}")
    print(f" E1          ={float(E1):12.4f}")
    print(f" E3          ={float(E3):12.4f}")
    print(f" PAO seed    ={seed[0]:12d}")

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
                    UR[x, z + NED2, comp] = 0.0 + 0.0j

    # ------------------------------------------------------------------
    # Scatter spectral UR → compact UC(kx,z,comp) buffer (current grid)
    #   UC: (NK, NE, 3) on host, but DnsState.uc is (NZ, NK, 3) in xp
    # ------------------------------------------------------------------
    NK = S.NK
    UC_host = np.zeros((NK, NE, 3), dtype=np.complex64)  # only comp 0,1 used

    for z in range(NE):
        for x in range(ND2):
            for c in range(2):
                UC_host[x, z, c] = UR[x, z, c]

    # ALSO build full 3/2-grid UC_full (Fortran-like layout)
    NK_full = S.NK_full
    NZ_full = S.NZ_full

    UC_full_host = np.zeros((NK_full, NZ_full, 3), dtype=np.complex64)
    for z in range(NE):
        for x in range(ND2):
            for c in range(2):
                UC_full_host[x, z, c] = UR[x, z, c]

    print(f" PAO initialization OK. VISC={float(S.visc):12.10f}")

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
    vfft_full_inverse_uc_full_to_ur_full(S)

    # Spectral vorticity from UC_full, like dnsCudaCalcom
    dns_calcom_from_uc_full(S)

    # No history yet
    S.fnm1[...] = xp.zeros_like(S.om2)


# ---------------------------------------------------------------------------
# FFT helpers (vfft_full_* equivalents)
# ---------------------------------------------------------------------------

def vfft_full_inverse_uc_full_to_ur_full(S: DnsState) -> None:
    """
    UC_full (3, NZ_full, NK_full) → UR_full (3, NZ_full, NX_full)

    Correct inverse:
      1) inverse FFT along z  (complex → complex)
      2) inverse real FFT along x (complex → real)

    ONLY CHANGE: use irfft2 on (z,x) axes.
    """
    xp = S.xp
    UC = S.uc_full
    fft = S.fft
    if fft is None:
        raise RuntimeError("scipy.fft is not available on CPU (import scipy.fft failed).")

    if S.backend == "cpu":
        # overwrite_x reduces temporary allocations on SciPy/pocketfft
        ur_full = fft.irfft2(UC, s=(S.NZ_full, S.NX_full), axes=(1, 2), overwrite_x=True)
    else:
        ur_full = fft.irfft2(UC, s=(S.NZ_full, S.NX_full), axes=(1, 2))

    S.ur_full[...] = xp.asarray(ur_full, dtype=xp.float32)


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
    if fft is None:
        raise RuntimeError("scipy.fft is not available on CPU (import scipy.fft failed).")

    if S.backend == "cpu":
        # overwrite_x is safe here (UR_full is overwritten later by STEP2A anyway)
        UC = fft.rfft2(UR, s=(S.NZ_full, S.NX_full), axes=(1, 2), overwrite_x=True)
    else:
        UC = fft.rfft2(UR, s=(S.NZ_full, S.NX_full), axes=(1, 2))

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
#
# Mirrors Fortran STEP2B:
#
#   1) Build uiuj in UR(x,z,1..3)
#   2) VRFFTF + VCFFTF on UR(.,.,I) for I=1..3  → UC(.,.,I)
#   3) Zero UC(X,NZ+1,I) for X<=NX/2, I=1..3
# ---------------------------------------------------------------------------
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

    # SoA layout:
    #   UR_full[comp, z, x]    with comp=0,1,2
    #   UC_full[comp, z, kx]
    UR = S.ur_full
    UC = S.uc_full

    # ------------------------------------------------------------------
    # 1) Build uiuj on the full 3/2 grid
    #
    # Before STEP2B (UR_full):
    #   comp 0 → u(x,z)
    #   comp 1 → w(x,z)
    #   comp 2 → (don't care)
    #
    # After this block (Fortran UIUJ build):
    #   UR(:,:,3) = u*w  → comp 2
    #   UR(:,:,1) = u*u  → comp 0
    #   UR(:,:,2) = w*w  → comp 1
    # ------------------------------------------------------------------
    u = UR[0]   # (NZ_full, NX_full)
    w = UR[1]   # (NZ_full, NX_full)

    # Use in-place multiplies to avoid temporaries
    xp.multiply(u, w, out=UR[2])  # u * w
    xp.multiply(u, u, out=UR[0])  # u^2
    xp.multiply(w, w, out=UR[1])  # w^2

    # ------------------------------------------------------------------
    # 2) Full-grid forward FFT: UR_full → UC_full (3 components)
    #
    # CUDA version calls:
    #   vfft_full_forward_ur_full_to_uc_full(S)
    # ------------------------------------------------------------------
    vfft_full_forward_ur_full_to_uc_full(S)

    # ------------------------------------------------------------------
    # 3) Zero the "middle" Fourier coefficient UC(X,NZ+1,I)
    #    for X <= NX/2 and I = 1..3
    #
    # Fortran:
    #   DO I=1,3
    #     DO X=1,NX/2
    #       UC(X,NZ+1,I) = 0
    #     END DO
    #   END DO
    #
    # 0-based Python/CUDA:
    #   NX_half = N/2
    #   NZ      = N
    #   z_mid   = NZ      (index for Z = NZ+1 in 1-based)
    # ------------------------------------------------------------------
    NX_half = N // 2
    NZ = N
    z_mid = NZ

    kx_max = min(NX_half, NK_full)

    if z_mid < NZ_full and kx_max > 0:
        UC[0:3, z_mid, 0:kx_max] = xp.complex64(0.0 + 0.0j)


# ---------------------------------------------------------------------------
# STEP3 — vorticity update using om2 & fnm1
# ---------------------------------------------------------------------------
def dns_step3(S: DnsState) -> None:
    """
    STEP3 — update spectral vorticity OM2 and non-linear term FNM1,
    then reconstruct the low-k velocity spectrum UC_full from OM2.

    This mirrors the CUDA:
      - k_step3_update_om2_and_fnm1
      - k_step3_update_uc_from_om2
      - dnsCudaStep3
    """
    xp = S.xp

    # ------------------------------------------------------------------
    # Aliases to match CUDA naming
    # ------------------------------------------------------------------
    om2 = S.om2         # (NZ, NX_half), complex64
    fnm1 = S.fnm1       # (NZ, NX_half), complex64
    alfa = S.alfa       # (NX_half,), float32
    gamma = S.gamma     # (NZ,),      float32
    uc_full = S.uc_full # (3, NZ_full, NK_full), complex64

    Nbase = int(S.Nbase)
    NX_half = Nbase // 2
    NZ = Nbase

    # Scalars (float32) to match CUDA
    visc = xp.float32(S.visc)
    dt = xp.float32(S.dt)
    cn = xp.float32(S.cn)
    cnm1 = xp.float32(S.cnm1)

    # ------------------------------------------------------------------
    # 1) Update OM2 and FNM1 (nonlinear vorticity forcing)
    #    Reuse preallocated scratch/constant grids to avoid per-step allocs.
    # ------------------------------------------------------------------
    z_spec = S.step3_z_spec                 # (NZ,)
    divxz = S.step3_divxz                   # float32 scalar
    GA = S.step3_GA                         # (NZ, NX_half), float32
    G2mA2 = S.step3_G2mA2                   # (NZ, NX_half), float32
    K2 = S.step3_K2                         # (NZ, NX_half), float32

    # Gather UC low-k band for each component into preallocated buffers:
    #   uc_full layout: [comp, z, kx]
    uc0_low = uc_full[0, :, :NX_half]       # (NZ_full, NX_half) view
    uc1_low = uc_full[1, :, :NX_half]
    uc2_low = uc_full[2, :, :NX_half]

    uc1_th = S.step3_uc1_th                 # (NZ, NX_half)
    uc2_th = S.step3_uc2_th
    uc3_th = S.step3_uc3_th
    xp.take(uc0_low, z_spec, axis=0, out=uc1_th)
    xp.take(uc1_low, z_spec, axis=0, out=uc2_th)
    xp.take(uc2_low, z_spec, axis=0, out=uc3_th)

    # FN = (GA*(UC1-UC2) + (G2-A2)*UC3) * DIVXZ
    # Compute into scratch1, keep fnm1 untouched until after OM2 update (needs old fnm1).
    tmp_FN = S.scratch1                     # (NZ, NX_half), complex64
    tmp_c = S.scratch2                      # (NZ, NX_half), complex64
    xp.subtract(uc1_th, uc2_th, out=tmp_FN)         # UC1-UC2
    xp.multiply(tmp_FN, GA, out=tmp_FN)             # GA*(UC1-UC2)
    xp.multiply(uc3_th, G2mA2, out=tmp_c)           # (G2-A2)*UC3
    xp.add(tmp_FN, tmp_c, out=tmp_FN)               # sum
    tmp_FN *= divxz

    # Crank–Nicolson in spectral space (Fortran STEP3), update OM2 in-place
    VT = xp.float32(0.5) * visc * dt
    ARG = S.step3_ARG                                 # (NZ, NX_half), float32
    DEN = S.step3_DEN                                 # (NZ, NX_half), float32
    xp.multiply(K2, VT, out=ARG)                      # ARG = VT*K2
    xp.add(ARG, xp.float32(1.0), out=DEN)             # DEN = 1 + ARG

    c2 = xp.float32(0.5) * dt * (xp.float32(2.0) + cnm1)
    c3 = -xp.float32(0.5) * dt * cnm1

    # NUM = (1-ARG)*OM2 + c2*FN + c3*FNM1
    NUM = S.step3_NUM                                 # (NZ, NX_half), complex64
    NUM[...] = om2                                    # NUM = OM2
    xp.multiply(om2, ARG, out=tmp_c)                  # tmp_c = ARG*OM2
    NUM -= tmp_c                                      # NUM = OM2 - ARG*OM2

    xp.multiply(tmp_FN, c2, out=tmp_c)                # tmp_c = c2*FN
    NUM += tmp_c
    xp.multiply(fnm1, c3, out=tmp_c)                  # tmp_c = c3*FNM1_old
    NUM += tmp_c

    xp.divide(NUM, DEN, out=om2)                      # OM2_new in-place

    # Store FN into FNM1 (overwriting old history)
    fnm1[...] = tmp_FN

    # ------------------------------------------------------------------
    # 2) Reconstruct UC low-k band from OM2 (spectral velocity reconstruction)
    #    Reuse scratch1/scratch2 as out buffers.
    # ------------------------------------------------------------------
    out1 = S.scratch1
    out2 = S.scratch2
    out1[...] = 0
    out2[...] = 0

    if NX_half > 1:
        invK2_sub = S.step3_invK2_sub                 # (NZ, NX_half-1), float32

        # out1[:,1:] = -i * GAMMA * (OM2/K2)
        out1[:, 1:] = om2[:, 1:]
        out1[:, 1:] *= invK2_sub
        out1[:, 1:] *= gamma[:, None]
        out1[:, 1:] *= xp.complex64(-1.0j)

        # out2[:,1:] =  i * ALFA * (OM2/K2)
        out2[:, 1:] = om2[:, 1:]
        out2[:, 1:] *= invK2_sub
        out2[:, 1:] *= alfa[1:][None, :]
        out2[:, 1:] *= xp.complex64(1.0j)

    # ix=0 branch:
    #   Z>=2: UC1 = -i*OM2/GAMMA, UC2=0
    out1[:, 0] = 0
    mask0 = S.step3_mask_ix0
    out1[mask0, 0] = xp.complex64(-1.0j) * (om2[mask0, 0] / gamma[mask0])
    # out2[:,0] remains zero

    # Scatter into UC_full low-k strip: z=0..NZ-1, kx=0..NX_half-1
    uc_full[0, :NZ, :NX_half] = out1
    uc_full[1, :NZ, :NX_half] = out2

    # ------------------------------------------------------------------
    # 3) CNM1 update (Fortran STEP3 does CNM1 = CN)
    # ------------------------------------------------------------------
    S.cnm1 = float(cn)


# ===============================================================
# STEP2A core (dealias + reshuffle + inverse FFT)
#   Python/CuPy version of dnsCudaStep2A_full
#   Operates on S.uc_full (3, NZ_full, NK_full) and S.ur_full.
# ===============================================================
def dns_step2a(S: DnsState) -> None:
    """
    Python/CuPy port of dnsCudaStep2A_full:
      1) Dealias high-kx band on UC_full (comp 0,1)
      2) Z-reshuffle low-kz strip (as in visasub.f STEP2A)
      3) Inverse FFT along Z in-place on UC_full (complex→complex)
      4) Inverse FFT along X (complex→real) into UR_full (3/2 grid)
      5) Copy centered N×N block of UR_full into compact UR (N×N)

    Uses the DnsState SoA layout:
      uc_full : (3, NZ_full, NK_full)
      ur_full : (3, NZ_full, NX_full)
      ur      : (NZ, NX, 3)
    """
    xp = S.xp
    N = S.Nbase
    NX = S.NX
    NZ = S.NZ
    NX_full = S.NX_full   # 3*N/2
    NZ_full = S.NZ_full   # 3*N/2
    NK_full = S.NK_full   # 3*N/4+1

    UC = S.uc_full  # shape (3, NZ_full, NK_full), complex64

    # ----------------------------------------------------------
    # 1) Dealias high-kx modes for comp 0,1
    #    (k_step2a_full_zero_highkx)
    #    In CUDA: high kx region [N/2 .. 3N/4] is zeroed.
    # ----------------------------------------------------------
    nx_start = N // 2           # N/2
    nx_end = 3 * N // 4       # 3N/4
    hi_start = nx_start
    hi_end = min(nx_end, NK_full - 1)

    if hi_start <= hi_end:
        # comps 0,1 correspond to U1, U3
        UC[0:2, :, hi_start:hi_end + 1] = xp.complex64(0.0 + 0.0j)

    # ----------------------------------------------------------
    # 2) Z-reshuffle low-kz strip (Fortran STEP2A-style)
    #
    # For the 3/2 grid (NZ_full = 3N/2) this is a contiguous block move:
    #
    #   z_mid = N/2 .. N-1
    #   z_top = N   .. 3N/2-1
    #
    # and only for kx < min(N/2, NK_full).
    # ----------------------------------------------------------
    halfN = N // 2
    k_max = min(halfN, NK_full)

    if k_max > 0:
        z_mid_start = halfN
        z_mid_end = halfN + halfN    # == N
        z_top_start = N
        z_top_end = N + halfN        # == 3N/2 == NZ_full

        UC[0:2, z_top_start:z_top_end, :k_max] = UC[0:2, z_mid_start:z_mid_end, :k_max]
        UC[0:2, z_mid_start:z_mid_end, :k_max] = xp.complex64(0.0 + 0.0j)

    # ----------------------------------------------------------
    # 3+4) Inverse transforms via irfft2, then scale to match CUFFT
    # ----------------------------------------------------------
    fft = S.fft
    if fft is None:
        raise RuntimeError("scipy.fft is not available on CPU (import scipy.fft failed).")

    # irfft2 includes 1/(NZ_full*NX_full); CUFFT inverse is unscaled
    if S.backend == "cpu":
        ur_full = fft.irfft2(UC, s=(NZ_full, NX_full), axes=(1, 2), overwrite_x=True)
    else:
        ur_full = fft.irfft2(UC, s=(NZ_full, NX_full), axes=(1, 2))

    ur_full *= (NZ_full * NX_full)
    S.ur_full[...] = ur_full

    # ----------------------------------------------------------
    # 5) Downmap compact N×N block (centered, like
    #    k_copy_ur_full_to_ur_centered)
    #
    # ur_full layout: (3, NZ_full, NX_full)
    # ur         : (NZ, NX, 3)
    # ----------------------------------------------------------
    off_x = (NX_full - NX) // 2
    off_z = (NZ_full - NZ) // 2

    S.ur[:, :, 0] = S.ur_full[0, off_z:off_z + N, off_x:off_x + N]
    S.ur[:, :, 1] = S.ur_full[1, off_z:off_z + N, off_x:off_x + N]
    S.ur[:, :, 2] = 0.0


# ---------------------------------------------------------------------------
# NEXTDT — CFL based timestep
# ---------------------------------------------------------------------------

def compute_cflm(S: DnsState) -> float:
    """
    Compute CFLM = max(|u|/dx + |w|/dz) on the full 3/2 grid,
    same definition as in CUDA.

    CPU optimization: reuse preallocated buffers to avoid per-step temporaries.
    """
    xp = S.xp

    NX3D2 = S.NX_full
    NZ3D2 = S.NZ_full

    u = S.ur_full[0, :NZ3D2, :NX3D2]
    w = S.ur_full[1, :NZ3D2, :NX3D2]

    # dx == dz == 2*pi/N → (|u|/dx + |w|/dz) == (|u| + |w|) * inv_dx
    tmp = S.cfl_tmp[:NZ3D2, :NX3D2]            # scratch plane (float32)
    absw = S.cfl_absw[:NZ3D2, :NX3D2]           # scratch plane (float32)

    xp.abs(u, out=tmp)
    xp.abs(w, out=absw)
    xp.add(tmp, absw, out=tmp)

    CFLM = float(xp.max(tmp) * S.inv_dx)
    return CFLM


def next_dt(S: DnsState) -> None:
    """
    Python version of next_dt_gpu.

    Uses UR_full to compute CFLM and adjusts DT and CN as in your CUDA code:
        CFL  = CFLM * DT * PI
        CN   = 0.8 + 0.2 * CFLNUM / CFL
        DT   = DT * CN
    """
    PI = math.pi

    CFLM = compute_cflm(S)
    if CFLM <= 0.0 or S.dt <= 0.0:
        return

    CFL = CFLM * S.dt * PI

    # Update CN and DT (CNM1 is updated in STEP3)
    S.cn = 0.8 + 0.2 * (S.cflnum / CFL)
    S.dt = S.dt * S.cn


# ===============================================================
# Python equivalent of dnsCudaDumpFieldAsPGMFull
# ===============================================================
def dump_field_as_pgm_full(S: DnsState, comp: int, filename: str) -> None:
    """
    Python/CuPy version of:

        void dnsCudaDumpFieldAsPGMFull(DnsDeviceState *S, int comp,
                                       const char *filename)

    Uses S.ur_full (3, NZ_full, NX_full), SoA layout:
      ur_full[comp, z, x]

    Writes an 8-bit binary PGM (P5) file with values mapped
    linearly from [minv, maxv] → [1, 255], same as the CUDA code.
    """
    NX_full = S.NX_full
    NZ_full = S.NZ_full

    # ------------------------------------------------------------
    # Bring UR_full to host as float32, layout [comp][z][x]
    # ------------------------------------------------------------
    if S.backend == "gpu":
        ur_full_host = _np.asarray(S.ur_full.get(), dtype=_np.float32)
    else:
        ur_full_host = _np.asarray(S.ur_full, dtype=_np.float32)

    # Selected component plane: shape (NZ_full, NX_full)
    field = ur_full_host[comp, :, :]

    # ------------------------------------------------------------
    # Compute min and max over the selected component
    # layout: [comp][z][x]
    # ------------------------------------------------------------
    minv = float(field.min())
    maxv = float(field.max())

    try:
        f = open(filename, "wb")
    except OSError as e:
        print(f"[DUMP] fopen failed for {filename!r}: {e}")
        return

    # P5 header: binary grayscale
    header = f"P5\n{NX_full} {NZ_full}\n255\n"
    f.write(header.encode("ascii"))

    rng = maxv - minv

    # ------------------------------------------------------------
    # Map [minv, maxv] → [1,255]
    # If nearly constant field, use mid-grey 128
    # ------------------------------------------------------------
    if abs(rng) <= 1.0e-12:
        # field is essentially constant
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
    """
    Fill S.ur_full[2, :, :] with the kinetic-energy magnitude |u|.

    Fortran FIELD2KIN computes:
        K = sqrt(UR(:,:,1)^2 + UR(:,:,2)^2)

    Here we compute the same quantity on the full 3/2 grid:
        ur_full[0] → u, ur_full[1] → w
        ur_full[2] ← sqrt(u^2 + w^2)
    """
    xp = S.xp

    u = S.ur_full[0, :, :]
    w = S.ur_full[1, :, :]

    ke = xp.sqrt(u * u + w * w)
    S.ur_full[2, :, :] = ke.astype(xp.float32)


def _spectral_band_to_phys_full_grid(S: DnsState, band) -> any:
    """
    Internal helper: take a compact spectral band band(z, kx) with
    shape (NZ, NX/2) and map it to a full 3/2-grid physical field
    using the same de-aliasing + reshuffle + inverse FFT sequence
    as STEP2A / OM2PHYS / STREAMFUNC.

    Returns a real array of shape (NZ_full, NX_full), dtype float32.
    """
    xp = S.xp

    N = S.Nbase
    NX_full = S.NX_full
    NZ_full = S.NZ_full
    NK_full = S.NK_full

    NX_half = N // 2
    NZ = N

    # 2D spectral buffer: layout [z, kx]
    uc_tmp = xp.zeros((NZ_full, NK_full), dtype=xp.complex64)

    # Copy compact band into low-k strip (Z=0..NZ-1, kx=0..NX/2-1)
    uc_tmp[:NZ, :NX_half] = band

    # ----------------------------------------------------------
    # Dealias high-kx: zero band [N/2 .. 3N/4] as in STEP2A
    # ----------------------------------------------------------
    hi_start = N // 2
    hi_end = min(3 * N // 4, NK_full - 1)
    if hi_start <= hi_end:
        uc_tmp[:, hi_start:hi_end + 1] = xp.complex64(0.0 + 0.0j)

    # ----------------------------------------------------------
    # Z-reshuffle low-kz strip (STEP2A-style)
    #
    # For the 3/2 grid (NZ_full = 3N/2) this is a contiguous block move:
    #
    #   z_mid = N/2 .. N-1
    #   z_top = N   .. 3N/2-1
    #
    # and only for kx < min(N/2, NK_full).
    # ----------------------------------------------------------
    halfN = N // 2
    k_max = min(halfN, NK_full)

    if k_max > 0:
        z_mid_start = halfN
        z_mid_end = halfN + halfN    # == N
        z_top_start = N
        z_top_end = N + halfN        # == 3N/2 == NZ_full

        uc_tmp[z_top_start:z_top_end, :k_max] = uc_tmp[z_mid_start:z_mid_end, :k_max]
        uc_tmp[z_mid_start:z_mid_end, :k_max] = xp.complex64(0.0 + 0.0j)

# Zero the "middle" Fourier coefficient Z = NZ+1 (1-based)
    z_mid = NZ  # 0-based index
    if z_mid < NZ_full:
        uc_tmp[z_mid, :NX_half] = xp.complex64(0.0 + 0.0j)

    # ONLY CHANGE: use irfft2 then scale to match CUFFT
    fft = S.fft
    if fft is None:
        raise RuntimeError("scipy.fft is not available on CPU (import scipy.fft failed).")

    if S.backend == "cpu":
        phys = fft.irfft2(uc_tmp, s=(NZ_full, NX_full), axes=(0, 1), overwrite_x=True)
    else:
        phys = fft.irfft2(uc_tmp, s=(NZ_full, NX_full), axes=(0, 1))

    phys *= (NZ_full * NX_full)
    return xp.asarray(phys, dtype=xp.float32)


def dns_om2_phys(S: DnsState) -> None:
    """
    Fill S.ur_full[2, :, :] with the physical vorticity ω(x,z).

    Fortran OM2PHYS does:
      - insert OM2 into UC(:,:,3)
      - de-alias + reshuffle
      - VCFFTB + VRFFTB → UR(:,:,3)

    Here we:
      - start from S.om2 (shape NZ × NX/2)
      - map that spectral band to a full 3/2-grid physical field
      - store in S.ur_full[2, :, :].
    """
    xp = S.xp

    # S.om2 has shape (NZ, NX_half) with NZ = Nbase
    band = S.om2
    phys = _spectral_band_to_phys_full_grid(S, band)

    S.ur_full[2, :, :] = phys


def dns_stream_func(S: DnsState) -> None:
    """
    Fill S.ur_full[2, :, :] with the streamfunction φ(x,z).

    Fortran STREAMFUNC does:
      - φ̂(X,Z) = OM2(X,Z) / (ALFA(X)^2 + GAMMA(Z)^2 + 1e-30)
      - insert φ̂ into UC(:,:,4)
      - de-alias + reshuffle
      - inverse 2D FFT → UR(:,:,4)

    Here we:
      - build K2 on the compact grid (NZ × NX/2)
      - φ̂ = OM2 / (K2 + 1e-30)
      - map φ̂ via the same spectral → physical helper
      - store result in S.ur_full[2, :, :].
    """
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
    print("--- INITIALIZING DNS_ALL PYTHON (SciPy/CuPy) ---")
    print(f" N   = {N}")
    print(f" Re  = {Re}")
    print(f" K0  = {K0}")
    print(f" Steps = {STEPS}")
    print(f" CFL  = {CFL}")
    print(f" requested = {backend}")

    S = create_dns_state(N=N, Re=Re, K0=K0, CFL=CFL, backend=backend)
    print(f" effective = {S.backend} (xp = {'cupy' if S.backend == 'gpu' else 'scipy'})")

    # Choose a context: SciPy FFT workers (CPU only), otherwise do nothing
    if S.backend == "cpu" and _spfft is not None and S.fft_workers > 1:
        fft_ctx = _spfft.set_workers(S.fft_workers)  # context manager
    else:
        fft_ctx = nullcontext()

    with fft_ctx:
        if _spfft is not None:
            print(f" scipy.fft workers in-context = {_spfft.get_workers()}")
        else:
            print(" scipy.fft workers in-context = n/a (gpu or scipy.fft missing)")

        dns_step2a(S)

        # ----------------------------------------
        # Match CUDA's NEXTDT INIT behaviour
        # ----------------------------------------
        CFLM = compute_cflm(S)
        # CUDA-style initial DT: CFLM * DT * PI = CFLNUM  →  DT = CFLNUM / (CFLM * PI)
        S.dt = S.cflnum / (CFLM * math.pi)
        S.cn = 1.0
        S.cnm1 = 0.0

        print(f" [NEXTDT INIT] CFLM={CFLM:11.4f} DT={S.dt:11.7f} CN={S.cn:11.7f}")
        print(f" Initial DT={S.dt:11.7f} CN={S.cn:11.7f}")

        S.sync()
        t0 = time.perf_counter()

        for it in range(1, STEPS + 1):
            S.it = it

            # --- save old dt (CUDA uses this for time advance) ---
            dt_old = S.dt

            # STEP2B
            dns_step2b(S)

            # STEP3
            dns_step3(S)

            # STEP2A
            dns_step2a(S)

            # NEXTDT: updates S.cn and S.dt (dt_new)
            next_dt(S)

            # Advance time with *old* dt, like CUDA/Fortran
            S.t += dt_old

            if (it % 100) == 0 or it == 1 or it == STEPS:
                print(f" ITERATION {it:6d} T={S.t:12.10f} DT={S.dt:10.8f} CN={S.cn:10.8f}")

        S.sync()
        t1 = time.perf_counter()

        elap = t1 - t0
        fps = (STEPS / elap) if elap > 0 else 0.0

        print(f" Elapsed CPU time for {STEPS} steps (s) = {elap:12.4f}")
        print(f" Frames per second (FPS)                 = {fps:12.4f}")
        print(f" Final T={S.t:12.10f}  CN={S.cn:12.10f}  DT={S.dt:12.10f}  VISC={S.visc:12.10f}")

def main():
    #
    #   dns_all N Re K0 STEPS CFL auto
    #   $ python dns_simulator.py 256 10000 10 301 0.75 cpu
    #
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