# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

2D homogeneous incompressible turbulence DNS solver (pseudo-spectral, 3/2 de-aliasing, Crank–Nicolson time integration, PAO random-field init). Pure-Python structural port of `dns_all.cu`. Runs on CPU via SciPy/NumPy or GPU via CuPy (CUDA 13, `cupy-cuda13x`). Ships with a PySide6 GUI (`cupystorm`) for live visualization.

Python 3.13 only (`requires-python = ">=3.13,<3.14"`).

## Common commands

Environment / install:

    uv sync                 # CPU-only
    uv sync --extra cuda    # adds cupy-cuda13x + fastrlock

Entry points (defined in `pyproject.toml` `[project.scripts]`):

    uv run -- turbulence    # PySide6 GUI (scipyturbo.turbo_main:main)
    uv run -- sim           # Headless DNS loop (turbo_simulator:main)
    uv run -- fps           # FPS-benchmark variant (turbo_simulator_fps:main)
    uv run -- cufft         # cuFFT-plan variant (turbo_simulator_cufft:main)

Headless CLI positional args (all four `sim*` entry points share this shape):

    python -m scipyturbo.turbo_simulator N Re K0 STEPS CFL BACKEND
    # BACKEND ∈ {cpu, gpu, auto}

Verify GPU:

    uv run python -c "import cupy as cp; x = cp.arange(5); print(x, x.device)"

Profiling (from README):

    python -m cProfile -o turbo_simulator.prof -m scipyturbo.turbo_simulator
    snakeviz turbo_simulator.prof
    scalene -m scipyturbo.turbo_simulator 256 10000 10 201 0.75 cpu

macOS `.app` bundle: `uv run pyinstaller macos.spec` (see `macos.spec`).

Windows CUDA launch helpers: `CUDA.bat` / `CUDA.ps1` set `CUDA_PATH` to a `Briefcase\v13.1` install and run `python -m scipyturbo.turbo_main`.

There is no test suite, linter config, or CI in this repo.

## Architecture

Single package `scipyturbo/` with four large top-level modules. Do not treat the `*_fps.py` / `*_cufft.py` files as dead variants — each is its own entry point registered in `pyproject.toml`.

### Solver (`turbo_simulator.py`)

Core `DnsState` dataclass mirrors `DnsDeviceState` from the CUDA source. All arrays live on `DnsState` and the `xp` alias (`numpy`/`scipy.fft` on CPU, `cupy`/`cupyx.scipy.fft` on GPU) is resolved once in `get_xp()` / `create_dns_state()`.

Grid conventions (critical — many shape bugs come from mixing these):

- Compact grid (AoS): `ur` shape `(NZ, NX, 3)`, `uc` shape `(NZ, NK, 3)`
- Full 3/2 grid (SoA): `ur_full` shape `(3, NZ_full, NX_full)`, `uc_full` shape `(3, NZ_full, NK_full)`
- `NX = NZ = N`, `NX_full = NZ_full = 3*N/2`, `NK_full = NX_full/2 + 1`, `NK = 3*N/4 + 1`
- `ur_full`'s first axis is `comp` (0=u, 1=v, 2=scratch for derived fields like energy/omega/stream)

Time loop (matches `dns_all.cu`): `STEP2B → STEP3 → STEP2A → NEXTDT`, implemented as `dns_step2b → dns_step3 → dns_step2a → next_dt`. `next_dt` pulls a device scalar to host on GPU — this is a hard sync, keep it out of the hot loop (see GPU-sync note below).

Per-step scratch buffers (`step3_*`, `cfl_tmp`, `cfl_absw`) are preallocated on `DnsState` to avoid allocations in the hot path. When adding new temporaries, follow this pattern.

PAO initialization (`dns_pao_host_init` → `_pao_build_ur_and_stats_impl`) is deterministic for a given seed and **must remain serial** — loop order preserves the RNG call sequence. A Numba-jit variant is selected automatically if `numba` is importable. Do not parallelize this kernel.

### Wrapper (`turbo_wrapper.py`)

`DnsSimulator` is the GUI-facing adapter. Holds a `DnsState`, exposes `step(mod_next_dt)`, `set_N(N)`, `reset_field()`, and pixel extraction (`make_pixels_component`, `get_frame_pixels`). `VAR_U/V/ENERGY/OMEGA/STREAM` constants are the variable selector enum consumed by the GUI.

GPU sync discipline: `step()` only *schedules* `next_dt` via `_next_dt_pending = True`; the actual call happens inside `get_frame_pixels()` right before `cp.asnumpy(...)`, so the unavoidable device→host transfer for the frame also covers the `next_dt` scalar sync. Do not call `next_dt` from `step()` on GPU — it will add a second sync point per iteration.

CPU path wraps FFT calls in `scipy.fft.set_workers(self.fft_workers)` (default 4). GPU path never enters that context.

### GUI (`turbo_main.py`)

Single-file PySide6 app. `MainWindow` (≈1200 lines) renders `ur_full` slices as `QImage::Format_Indexed8` with preloaded LUTs (`_make_*_lut` functions at top of file) — this Indexed8 path is why `get_frame_pixels()` returns a contiguous `uint8` array. The worker thread calls `DnsSimulator.step()` in a loop; GUI timer ticks pull a frame via `get_frame_pixels()`.

Keyboard shortcuts (`_setup_shortcuts`): H stop, G start, Y reset, V/C/N/R/K/L/S/U cycle variable/colormap/grid/Re/K0/CFL/steps/update-interval. See README for the full list.

`main()` picks a default `N` based on backend: 256 on CPU, 1024 on GPU, and 1024 on `auto` iff CuPy is importable.

### Variants

- `turbo_simulator_fps.py`, `turbo_simulator_cufft.py`: alternate solver modules with the same public functions. `cufft` version uses persistent cuFFT plans (`fft_plan_rfft2_ur_full`, `fft_plan_irfft2_uc01` fields on `DnsState`) for GPU performance.
- `turbo_crop.py`, `turbo_wrapper_nextdt.py`: auxiliary utilities.

## Conventions that matter

- Dtypes are `float32` / `complex64` throughout for GPU throughput — preserve this when adding code. Casts to float64 in the hot path silently halve GPU perf.
- When touching anything on the GPU hot path, count syncs. `.item()`, `float(cp_array)`, and `cp.asnumpy()` each force a device→host sync.
- `DnsState.backend` is the source of truth for `"cpu"` vs `"gpu"`; don't re-probe CuPy availability elsewhere.
- The three-component layout of `ur_full[2, :, :]` is reused as scratch by `dns_kinetic`, `dns_om2_phys`, `dns_stream_func` — callers overwrite it and then snapshot immediately. Don't assume component 2 is stable across calls.
