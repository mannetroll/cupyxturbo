# turbo_wrapper.py
from pathlib import Path
from typing import Union

import numpy as np
import math
import os
from scipyturbo import turbo_simulator as dns_all
from PIL import Image

# --- ONLY: SciPy FFT threading control (CPU) ---
import scipy.fft as spfft


class DnsSimulator:
    """
    DnsSimulator side is assumed to expose:
      - dns_init(N, re, k0)
      - dns_step(t, dt, cn)
      - dns_kinetic(px, py)
      - dns_om2phys(px, py)
      - dns_streamfunc(px, py)
      - dns_set_ur_real(field, nx, ny)
    """

    # Variable selector constants (used by the Qt GUI)
    VAR_U = 0
    VAR_V = 1
    VAR_ENERGY = 2
    VAR_OMEGA = 3
    VAR_STREAM = 4

    def __init__(
        self,
        n: int = 384,
        re: float = 10000.0,
        k0: float = 15.0,
        cfl: float = 0.5,
        seed: int = 1,
    ):
        self.N = int(n)
        self.m = 3 * self.N
        self.re = float(re)
        self.k0 = float(k0)
        self.cfl = float(cfl)
        self.seed = int(seed)
        self.max_steps = 5000

        # --- ONLY: max SciPy FFT workers on CPU ---
        self.fft_workers = 4

        # UR dimensions from Fortran workspace: UR(2+3N/2, 3N/2, 3)
        # For the pure-Python solver, we use the full 3/2-grid from DnsState.
        #   ur_full has shape (3, NZ_full, NX_full)
        #   NZ_full = 3*N/2, NX_full = 3*N/2
        # We map these directly to (py, px) for the GUI.
        with spfft.set_workers(self.fft_workers):
            self.state = dns_all.create_dns_state(
                N=self.N,
                Re=self.re,
                K0=self.k0,
                CFL=self.cfl,
                backend="auto",  # GUI uses the CPU/NumPy/GPU/CuPy backend
                seed=self.seed,
            )

            self.nx = int(self.state.NZ_full)  # "height"
            self.ny = int(self.state.NX_full)  # "width"

            # expose for GUI sizing
            self.py = self.nx  # height
            self.px = self.ny  # width

            # time integration scalars
            self.t = float(self.state.t)
            self.dt = float(self.state.dt)
            self.cn = float(self.state.cn)
            self.iteration = 0

            # which field to visualize
            self.current_var = self.VAR_U

            # initialize Python DNS state (mirror dns_all.run_dns NEXTDT INIT)
            #   1) initial STEP2A from spectral to physical
            #   2) compute CFLM
            #   3) set DT and CN from CFL condition
            dns_all.dns_step2a(self.state)
            CFLM = dns_all.compute_cflm(self.state)
            # CFLM * DT * PI = CFLNUM  →  DT = CFLNUM / (CFLM * PI)
            self.state.dt = self.state.cflnum / (CFLM * math.pi)
            self.state.cn = 1.0
            self.state.cnm1 = 0.0

            # mirror into the simulator scalars
            self.t = float(self.state.t)
            self.dt = float(self.state.dt)
            self.cn = float(self.state.cn)

    # ------------------------------------------------------------------
    def step(self, mod_next_dt: int) -> None:
        """Advance one DNS step on the Fortran side."""
        # In the pure-Python version this mirrors dns_all.run_dns:
        #   dt_old = DT
        #   STEP2B
        #   STEP3
        #   STEP2A
        #   NEXTDT
        #   T = T + dt_old
        S = self.state

        dt_old = S.dt

        if S.backend == "cpu":
            with spfft.set_workers(self.fft_workers):
                dns_all.dns_step2b(S)
                dns_all.dns_step3(S)
                dns_all.dns_step2a(S)
        else:
            dns_all.dns_step2b(S)
            dns_all.dns_step3(S)
            dns_all.dns_step2a(S)

        # Call NEXTDT every mod_next_dt iterations
        if (self.iteration % mod_next_dt) == 0:
            dns_all.next_dt(S)

        S.t += dt_old

        self.t = float(S.t)
        self.dt = float(S.dt)
        self.cn = float(S.cn)
        self.iteration += 1

    def set_N(self, N: int) -> None:
        """Recreate the entire DNS state with a new grid size N."""
        self.N = int(N)
        self.m = 3 * self.N  # preserve original structure

        # Rebuild state exactly the same way __init__ does
        with spfft.set_workers(self.fft_workers):
            self.state = dns_all.create_dns_state(
                N=self.N,
                Re=self.re,
                K0=self.k0,
                CFL=self.cfl,
                backend="auto",
                seed=self.seed,
            )

        # DEBUG: print full-grid sizes
        '''
        try:
            print("DEBUG ur_full.shape =", self.state.ur_full.shape)
        except Exception as e:
            print("DEBUG ur_full.shape ERROR:", e)

        print("DEBUG NZ_full =", self.state.NZ_full)
        print("DEBUG NX_full =", self.state.NX_full)
        print("DEBUG N input =", self.N)
        print("-------------------------------------")
        '''

        # update px/py for GUI (3/2 rule inside Fortran/Python DNS)
        self.nx = int(self.state.NZ_full)
        self.ny = int(self.state.NX_full)
        self.py = self.nx
        self.px = self.ny

        # Reset integrator scalars like in reset_field()
        if self.state.backend == "cpu":
            with spfft.set_workers(self.fft_workers):
                dns_all.dns_step2a(self.state)
                CFLM = dns_all.compute_cflm(self.state)
        else:
            dns_all.dns_step2a(self.state)
            CFLM = dns_all.compute_cflm(self.state)

        self.state.dt = self.state.cflnum / (CFLM * math.pi)
        self.state.cn = 1.0
        self.state.cnm1 = 0.0

        self.t = float(self.state.t)
        self.dt = float(self.state.dt)
        self.cn = float(self.state.cn)
        self.iteration = 0

    # ------------------------------------------------------------------
    def reset_field(self) -> None:
        """Reinitialize the DNS state on the Fortran side."""
        self.t = np.float32(0.0)
        self.dt = np.float32(0.0)
        self.cn = np.float32(1.0)
        self.iteration = 0
        # Pick a fresh PAO seed each reset (LCG is mod 5011 → use 1..5010)
        seed = 1 + (int.from_bytes(os.urandom(8), "little") % 5010)

        # Recreate the Python DNS state and redo the NEXTDT INIT phase
        with spfft.set_workers(self.fft_workers):
            self.state = dns_all.create_dns_state(
                N=self.N,
                Re=self.re,
                K0=self.k0,
                CFL=self.cfl,
                backend="auto",
                seed=seed,
            )

        self.nx = int(self.state.NZ_full)
        self.ny = int(self.state.NX_full)
        self.py = self.nx
        self.px = self.ny

        if self.state.backend == "cpu":
            with spfft.set_workers(self.fft_workers):
                dns_all.dns_step2a(self.state)
                CFLM = dns_all.compute_cflm(self.state)
        else:
            dns_all.dns_step2a(self.state)
            CFLM = dns_all.compute_cflm(self.state)

        self.state.dt = self.state.cflnum / (CFLM * math.pi)
        self.state.cn = 1.0
        self.state.cnm1 = 0.0

        self.t = float(self.state.t)
        self.dt = float(self.state.dt)
        self.cn = float(self.state.cn)

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        return {"t": float(self.t), "dt": float(self.dt), "cn": float(self.cn)}

    # ------------------------------------------------------------------
    def _float_to_pixels(self, field: np.ndarray) -> np.ndarray:
        """
        Map a float field to 8-bit grayscale [1,255], like the PGM dumper.
        """
        fmin = float(field.min())
        fmax = float(field.max())
        rng = fmax - fmin

        if abs(rng) <= 1.0e-12:
            # essentially constant field → mid-grey
            return np.full(field.shape, 128, dtype=np.uint8)

        norm = (field - fmin) / rng    # 0..1
        pixf = 1.0 + norm * 254.0      # 1..255
        pix = np.clip(pixf, 1.0, 255.0)

        return pix.astype(np.uint8)

    # ------------------------------------------------------------------
    def _snapshot(self, comp: int) -> np.ndarray:
        """
        Raw snapshot from Fortran, now using dns_frame with 3× scale-up.
        """
        # For the pure-Python solver, we take component 'comp' from ur_full:
        #   comp = 1,2,3  → ur_full[0,1,2]
        S = self.state

        idx = int(comp) - 1
        if idx < 0 or idx > 2:
            idx = 0

        if S.backend == "gpu":
            import cupy as cp  # type: ignore
            field = cp.asnumpy(S.ur_full[idx, :, :])
        else:
            field = np.asarray(S.ur_full[idx, :, :])

        return self._float_to_pixels(field)

    # ------------------------------------------------------------------
    def make_pixels(self, comp: int = 1) -> np.ndarray:
        """
        Convenience: comp-based visualization.
        comp = 1,2,3 (UR components).
        """
        return self._snapshot(comp)

    # ------------------------------------------------------------------
    def make_pixels_component(self, var: int | None = None) -> np.ndarray:
        """
        High-level selector used by the GUI:
          VAR_U      -> UR(:,:,1)
          VAR_V      -> UR(:,:,2)
          VAR_ENERGY -> sqrt(u^2+v^2+w^2)
          VAR_OMEGA  -> vorticity
          VAR_STREAM -> streamfunction
        """
        if var is None:
            var = self.current_var

        S = self.state

        if var == self.VAR_U:
            plane = self._snapshot(1)
        elif var == self.VAR_V:
            plane = self._snapshot(2)
        elif var == self.VAR_ENERGY:
            # Use dns_all kinetic helper: fills ur_full[2,:,:]
            dns_all.dns_kinetic(S)
            if S.backend == "gpu":
                import cupy as cp  # type: ignore
                field = cp.asnumpy(S.ur_full[2, :, :])
            else:
                field = np.asarray(S.ur_full[2, :, :])
            plane = self._float_to_pixels(field)
        elif var == self.VAR_OMEGA:
            # Use dns_all omega→physical helper: fills ur_full[2,:,:]
            dns_all.dns_om2_phys(S)
            if S.backend == "gpu":
                import cupy as cp  # type: ignore
                field = cp.asnumpy(S.ur_full[2, :, :])
            else:
                field = np.asarray(S.ur_full[2, :, :])
            plane = self._float_to_pixels(field)
        elif var == self.VAR_STREAM:
            # Use dns_all stream-function helper: fills ur_full[2,:,:]
            dns_all.dns_stream_func(S)
            if S.backend == "gpu":
                import cupy as cp  # type: ignore
                field = cp.asnumpy(S.ur_full[2, :, :])
            else:
                field = np.asarray(S.ur_full[2, :, :])
            plane = self._float_to_pixels(field)
        else:
            plane = self._snapshot(1)

        return plane

    # ------------------------------------------------------------------
    def get_frame_pixels(self) -> np.ndarray:
        """Used by the Qt app worker thread.

        FIELD2PIX / dns_frame currently return 32-bit packed gray pixels
        (0x00LLLLLL). Here we reduce that once to an 8-bit contiguous
        array so the GUI can push it straight into a QImage.
        """
        plane = self.make_pixels_component(self.current_var)

        # plane is already uint8 [0..255]; we keep the original logic
        #   pixels32 & 0xFF  → pixels8
        pixels32 = np.asarray(plane, dtype=np.uint32)
        pixels8 = (pixels32 & 0xFF).astype(np.uint8)
        return np.ascontiguousarray(pixels8)

    def set_variable(self, var: int) -> None:
        """Select which variable the GUI should visualize."""
        self.current_var = int(var)

    def get_time(self) -> float:
        return float(self.t)

    def get_iteration(self) -> int:
        return self.iteration

    # ---------- PNG EXPORT -------------------------------------------
    def save_png(self, path: Union[str, Path], comp: int = 1) -> None:
        """
        Export current field component (UR(:,:,comp)) as grayscale PNG.

        comp = 1,2,3 (UR components).
        """
        pixels = self.make_pixels(comp)
        img = Image.fromarray(pixels, mode="L")  # L = 8-bit grayscale
        img.save(str(path))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    sim = DnsSimulator()

    print("Starting DNS simulation")
    for i in range(10):
        sim.step(1)
        d = sim.diagnostics()
        print(f"Step {i+1:2d}: T={d['t']:.6f}, DT={d['dt']:.6f}, CN={d['cn']:.6f}")

    pix = sim.make_pixels_component()
    print("Pixels:", pix.shape, pix[0, 0], pix[10, 10])

    '''
    arr = sim._snapshot(1)
    np.savetxt(
        "snapshot.csv",
        arr,
        fmt="%d",  # integer formatting
        delimiter=","
    )
    '''