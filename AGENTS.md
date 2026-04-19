# AGENTS.md

## Setup

- Install dependencies: `uv sync` (CPU-only) or `uv sync --extra cuda` (GPU support)
- Requires Python 3.13+ only (`requires-python = ">=3.13,<3.14"`)
- GPU verification: `uv run python -c "import cupy as cp; x = cp.arange(5); print(x, x.device)"`

## Project overview

2D homogeneous incompressible turbulence DNS solver using pseudo-spectral methods (3/2 de-aliasing, Crank–Nicolson time integration). Supports CPU (NumPy/SciPy) or GPU (CuPy/CUDA 13) backends. Includes PySide6 GUI (`turbulence` command) and headless CLI variants.

## Entry points (from pyproject.toml)

```
uv run -- turbulence        # PySide6 GUI (interactive visualization)
uv run -- sim               # Headless DNS loop
uv run -- fps               # FPS-benchmark variant
uv run -- cufft             # cuFFT-plan variant (GPU optimized)
```

Headless CLI signature:
```
python -m scipyturbo.turbo_simulator N Re K0 STEPS CFL BACKEND
# BACKEND ∈ {cpu, gpu, auto}
```

## Critical conventions

- **Dtypes**: All arrays are `float32`/`complex64` for GPU throughput. Casting to float64 in hot paths silently halves GPU performance.
- **Grid layout**: Compact grid (AoS) uses `(NZ, NX, 3)` for `ur`, full 3/2 grid (SoA) uses `(3, NZ_full, NX_full)` for `ur_full`. `NX = NZ = N`, `NZ_full = 3*N/2`.
- **GPU sync discipline**: On GPU, `DnsSimulator.step()` only schedules `next_dt` (sets `_next_dt_pending = True`). Actual scalar sync happens in `get_frame_pixels()` before the unavoidable device→host frame transfer. Never call `next_dt` directly in `step()` on GPU.
- **PAO initialization**: `dns_pao_host_init` must remain **serial** — loop order preserves RNG call sequence. Numba-jit is auto-selected if available, but parallelization breaks determinism.
- **Scratch buffers**: Pre-allocate on `DnsState` (e.g., `step3_*`, `cfl_tmp`, `cfl_absw`) to avoid hot-path allocations.

## Profiling

```
python -m cProfile -o turbo_simulator.prof -m scipyturbo.turbo_simulator 256 10000 10 201 0.75 cpu
snakeviz turbo_simulator.prof
scalene -m scipyturbo.turbo_simulator 256 10000 10 201 0.75 cpu
```

## Architecture notes

- Single package `scipyturbo/` with four top-level modules; each `*_fps.py` / `*_cufft.py` is a registered entry point, not dead code.
- `DnsState` mirrors `DnsDeviceState` from CUDA source. All arrays live there; `xp` alias resolves once via `get_xp()`.
- Time loop: `STEP2B → STEP3 → STEP2A → NEXTDT` (matches `dns_all.cu`).
- GUI uses `QImage::Format_Indexed8` with preloaded LUTs for performance; `get_frame_pixels()` returns contiguous `uint8`.

## No test suite, linter, or CI

This project ships without automated tests or linting. Validation is manual (profiling, visual inspection, benchmark comparisons).
