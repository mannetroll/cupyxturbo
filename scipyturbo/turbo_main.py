# turbo_main.py
import colorsys
import os
import sys
import time
from typing import Optional

from pathlib import Path
from PySide6.QtCore import QSize, QTimer, Qt, QStandardPaths
from PySide6.QtGui import QIcon, QImage, QPixmap, QFontDatabase, qRgb, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QStatusBar,
    QCheckBox,
    QStyle,
    QLineEdit,
)
import numpy as np

from scipyturbo import turbo_simulator as dns_all
from scipyturbo.turbo_wrapper import DnsSimulator


# Simple helper: build a 256x3 uint8 LUT from color stops in 0..1
# stops: list of (pos, (r,g,b)) with pos in [0,1], r,g,b in [0,255]
def _make_lut_from_stops(stops, size: int = 256) -> np.ndarray:
    stops = sorted(stops, key=lambda s: s[0])
    lut = np.zeros((size, 3), dtype=np.uint8)

    positions = [int(round(p * (size - 1))) for p, _ in stops]
    colors = [np.array(c, dtype=np.float32) for _, c in stops]

    for i in range(len(stops) - 1):
        x0 = positions[i]
        x1 = positions[i + 1]
        c0 = colors[i]
        c1 = colors[i + 1]

        if x1 <= x0:
            lut[x0] = c0.astype(np.uint8)
            continue

        length = x1 - x0
        for j in range(length):
            t = j / float(length)
            c = (1.0 - t) * c0 + t * c1
            lut[x0 + j] = c.astype(np.uint8)

    # last entry
    lut[positions[-1]] = colors[-1].astype(np.uint8)
    return lut


def _make_gray_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lut[i] = (i, i, i)
    return lut


def _make_fire_lut() -> np.ndarray:
    """Approximate 'fire' palette via HSL ramp: red → yellow, brightening."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for x in range(256):
        # Hue 0..85 degrees
        h_deg = 85.0 * (x / 255.0)
        h = h_deg / 360.0
        s = 1.0
        # Lightness: 0..1 up to mid, then flat
        l = min(1.0, x / 128.0)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        lut[x] = (int(r * 255), int(g * 255), int(b * 255))
    return lut


def _make_doom_fire_lut() -> np.ndarray:
    """Classic Doom fire palette approximated as 256 RGB colors."""
    key_colors = np.array([
        [0, 0, 0],
        [7, 7, 7],
        [31, 7, 7],
        [47, 15, 7],
        [71, 15, 7],
        [87, 23, 7],
        [103, 31, 7],
        [119, 31, 7],
        [143, 39, 7],
        [159, 47, 7],
        [175, 63, 7],
        [191, 71, 7],
        [199, 71, 7],
        [223, 79, 7],
        [223, 87, 7],
        [223, 87, 7],
        [215, 95, 7],
        [215, 95, 7],
        [215, 103, 15],
        [207, 111, 15],
        [207, 119, 15],
        [207, 127, 15],
        [207, 135, 23],
        [199, 135, 23],
        [199, 143, 23],
        [199, 151, 31],
        [191, 159, 31],
        [191, 159, 31],
        [191, 167, 39],
        [191, 167, 39],
        [191, 175, 47],
        [183, 175, 47],
        [183, 183, 47],
        [183, 183, 55],
        [207, 207, 111],
        [223, 223, 159],
        [239, 239, 199],
        [255, 255, 255],
    ], dtype=np.uint8)

    stops = []
    n_keys = key_colors.shape[0]
    for i in range(n_keys):
        pos = i / (n_keys - 1)
        stops.append((pos, key_colors[i].tolist()))
    return _make_lut_from_stops(stops)


def _make_viridis_lut() -> np.ndarray:
    # Approximate viridis with a few key colors from the official palette
    stops = [
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ]
    return _make_lut_from_stops(stops)


def _make_inferno_lut() -> np.ndarray:
    stops = [
        (0.0, (0, 0, 4)),
        (0.25, (87, 15, 109)),
        (0.50, (187, 55, 84)),
        (0.75, (249, 142, 8)),
        (1.0, (252, 255, 164)),
    ]
    return _make_lut_from_stops(stops)


def _make_ocean_lut() -> np.ndarray:
    # Ocean: deep-blue → blue → cyan → turquoise → pale-aqua
    stops = [
        (0.0, (0, 5, 30)),  # deep navy
        (0.25, (0, 60, 125)),  # rich ocean blue
        (0.50, (0, 140, 190)),  # cyan-blue mix
        (0.75, (0, 200, 175)),  # turquoise
        (1.0, (180, 245, 240)),  # pale aqua
    ]
    return _make_lut_from_stops(stops)


def _make_cividis_lut() -> np.ndarray:
    stops = [
        (0.00, (0, 34, 77)),
        (0.25, (0, 68, 117)),
        (0.50, (60, 111, 130)),
        (0.75, (147, 147, 95)),
        (1.00, (250, 231, 33)),
    ]
    return _make_lut_from_stops(stops)


def _make_jet_lut() -> np.ndarray:
    stops = [
        (0.00, (0, 0, 131)),  # dark blue
        (0.35, (0, 255, 255)),  # cyan
        (0.66, (255, 255, 0)),  # yellow
        (1.00, (128, 0, 0)),  # dark red
    ]
    return _make_lut_from_stops(stops)


def _make_coolwarm_lut() -> np.ndarray:
    stops = [
        (0.00, (59, 76, 192)),  # deep blue
        (0.25, (127, 150, 203)),
        (0.50, (217, 217, 217)),  # near white (center)
        (0.75, (203, 132, 123)),
        (1.00, (180, 4, 38)),  # deep red
    ]
    return _make_lut_from_stops(stops)


def _make_rdbu_lut() -> np.ndarray:
    stops = [
        (0.00, (103, 0, 31)),  # dark red
        (0.25, (178, 24, 43)),
        (0.50, (247, 247, 247)),  # white center
        (0.75, (33, 102, 172)),
        (1.00, (5, 48, 97)),  # dark blue
    ]
    return _make_lut_from_stops(stops)


def _make_plasma_lut() -> np.ndarray:
    stops = [
        (0.0, (13, 8, 135)),
        (0.25, (126, 3, 167)),
        (0.50, (203, 71, 119)),
        (0.75, (248, 149, 64)),
        (1.0, (240, 249, 33)),
    ]
    return _make_lut_from_stops(stops)


def _make_magma_lut() -> np.ndarray:
    stops = [
        (0.0, (0, 0, 4)),
        (0.25, (73, 18, 99)),
        (0.50, (150, 50, 98)),
        (0.75, (226, 102, 73)),
        (1.0, (252, 253, 191)),
    ]
    return _make_lut_from_stops(stops)


def _make_turbo_lut() -> np.ndarray:
    # Approximate Google's Turbo colormap with a few key stops.
    stops = [
        (0.0, (48, 18, 59)),
        (0.25, (31, 120, 180)),
        (0.50, (78, 181, 75)),
        (0.75, (241, 208, 29)),
        (1.0, (133, 32, 26)),
    ]
    return _make_lut_from_stops(stops)


GRAY_LUT = _make_gray_lut()
INFERNO_LUT = _make_inferno_lut()
OCEAN_LUT = _make_ocean_lut()
VIRIDIS_LUT = _make_viridis_lut()
PLASMA_LUT = _make_plasma_lut()
MAGMA_LUT = _make_magma_lut()
TURBO_LUT = _make_turbo_lut()
FIRE_LUT = _make_fire_lut()
DOOM_FIRE_LUT = _make_doom_fire_lut()
CIVIDIS_LUT = _make_cividis_lut()
JET_LUT = _make_jet_lut()
COOLWARM_LUT = _make_coolwarm_lut()
RDBU_LUT = _make_rdbu_lut()

COLOR_MAPS = {
    "Gray": GRAY_LUT,
    "Inferno": INFERNO_LUT,
    "Ocean": OCEAN_LUT,
    "Viridis": VIRIDIS_LUT,
    "Plasma": PLASMA_LUT,
    "Magma": MAGMA_LUT,
    "Turbo": TURBO_LUT,
    "Fire": FIRE_LUT,
    "Doom": DOOM_FIRE_LUT,
    "Cividis": CIVIDIS_LUT,
    "Jet": JET_LUT,
    "Coolwarm": COOLWARM_LUT,
    "RdBu": RDBU_LUT,
}

DEFAULT_CMAP_NAME = "Inferno"

# ----------------------------------------------------------------------
# Option A: Qt Indexed8 + palette tables (avoid expanding to RGB in NumPy)
# ----------------------------------------------------------------------
QT_COLOR_TABLES = {
    name: [qRgb(int(rgb[0]), int(rgb[1]), int(rgb[2])) for rgb in lut]
    for name, lut in COLOR_MAPS.items()
}
QT_GRAY_TABLE = [qRgb(i, i, i) for i in range(256)]


def _setup_shortcuts(self):
    def sc(key, fn):
        s = QShortcut(QKeySequence(key), self)
        s.setContext(Qt.ShortcutContext.ApplicationShortcut)
        s.activated.connect(fn)  # type: ignore[attr-defined]
        return s

    self._sc_v = sc("V", lambda: self.variable_combo.setCurrentIndex(
        (self.variable_combo.currentIndex() + 1) % self.variable_combo.count()
    ))
    self._sc_c = sc("C", lambda: self.cmap_combo.setCurrentIndex(
        (self.cmap_combo.currentIndex() + 1) % self.cmap_combo.count()
    ))
    self._sc_n = sc("N", lambda: self.n_combo.setCurrentIndex(
        (self.n_combo.currentIndex() + 1) % self.n_combo.count()
    ))
    self._sc_r = sc("R", lambda: self.re_combo.setCurrentIndex(
        (self.re_combo.currentIndex() + 1) % self.re_combo.count()
    ))
    self._sc_k = sc("K", lambda: self.k0_combo.setCurrentIndex(
        (self.k0_combo.currentIndex() + 1) % self.k0_combo.count()
    ))
    self._sc_l = sc("L", lambda: self.cfl_combo.setCurrentIndex(
        (self.cfl_combo.currentIndex() + 1) % self.cfl_combo.count()
    ))
    self._sc_s = sc("S", lambda: self.steps_combo.setCurrentIndex(
        (self.steps_combo.currentIndex() + 1) % self.steps_combo.count()
    ))
    self._sc_u = sc("U", lambda: self.update_combo.setCurrentIndex(
        (self.update_combo.currentIndex() + 1) % self.update_combo.count()
    ))


class MainWindow(QMainWindow):
    def __init__(self, sim: DnsSimulator) -> None:
        super().__init__()

        self.sim = sim
        self.current_cmap_name = DEFAULT_CMAP_NAME

        # --- central image label ---
        self.image_label = QLabel()
        # allow shrinking when grid size becomes smaller
        self.image_label.setSizePolicy(
            self.image_label.sizePolicy().horizontalPolicy(),
            self.image_label.sizePolicy().verticalPolicy()
        )
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- small icon buttons ---
        style = QApplication.style()

        # Start button
        self.start_button = QPushButton()
        self.start_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_button.setToolTip("Start simulation")
        self.start_button.setFixedSize(28, 28)
        self.start_button.setIconSize(QSize(14, 14))

        # Stop button
        self.stop_button = QPushButton()
        self.stop_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_button.setToolTip("Stop simulation")
        self.stop_button.setFixedSize(28, 28)
        self.stop_button.setIconSize(QSize(14, 14))

        # Reset button
        self.reset_button = QPushButton()
        self.reset_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.reset_button.setToolTip("Reset simulation")
        self.reset_button.setFixedSize(28, 28)
        self.reset_button.setIconSize(QSize(14, 14))

        # Save button
        self.save_button = QPushButton()
        self.save_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_button.setToolTip("Save current frame")
        self.save_button.setFixedSize(28, 28)
        self.save_button.setIconSize(QSize(14, 14))

        # Folder button
        self.folder_button = QPushButton()
        self.folder_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        self.folder_button.setToolTip("Save files")
        self.folder_button.setFixedSize(28, 28)
        self.folder_button.setIconSize(QSize(14, 14))

        self._status_update_counter = 0
        self._update_intervall = 2

        # Variable selector
        self.variable_combo = QComboBox()
        self.variable_combo.setToolTip("V: Variable")
        self.variable_combo.addItems(["U", "V", "K", "Ω", "φ"])

        # Grid-size selector (N)
        self.n_combo = QComboBox()
        self.n_combo.setToolTip("N: Grid Size (N)")
        self.n_combo.addItems(["128", "192", "256", "384", "512", "768", "1024", "2048", "3072", "4096"])
        self.n_combo.setCurrentText(str(self.sim.N))

        # Reynolds selector (Re)
        self.re_combo = QComboBox()
        self.re_combo.setToolTip("R: Reynolds Number (Re)")
        self.re_combo.addItems(["1", "1000", "10000", "100000", "1E6", "1E9", "1E12", "1E15"])
        self.re_combo.setCurrentText(str(int(self.sim.re)))

        # K0 selector
        self.k0_combo = QComboBox()
        self.k0_combo.setToolTip("K: Initial energy peak wavenumber (K0)")
        self.k0_combo.addItems(["1", "5", "10", "15", "25", "35", "50"])
        self.k0_combo.setCurrentText(str(int(self.sim.k0)))

        # Colormap selector
        self.cmap_combo = QComboBox()
        self.cmap_combo.setToolTip("C: Colormaps")
        self.cmap_combo.addItems(list(COLOR_MAPS.keys()))
        idx = self.cmap_combo.findText(DEFAULT_CMAP_NAME)
        if idx >= 0:
            self.cmap_combo.setCurrentIndex(idx)

        # CFL selector
        self.cfl_combo = QComboBox()
        self.cfl_combo.setToolTip("L: Controlling Δt (CFL)")
        self.cfl_combo.addItems(["0.05", "0.15", "0.25", "0.5", "0.75", "0.95"])
        self.cfl_combo.setCurrentText(str(self.sim.cfl))

        # Steps selector
        self.steps_combo = QComboBox()
        self.steps_combo.setToolTip("S: Max steps before reset/stop")
        self.steps_combo.addItems(["2000", "5000", "10000", "25000", "50000", "1E5", "1E6"])
        self.steps_combo.setCurrentText("5000")

        # Update selector
        self.update_combo = QComboBox()
        self.update_combo.setToolTip("U: Update intervall")
        self.update_combo.addItems(["2", "5", "10", "20", "50", "100", "1000"])
        self.update_combo.setCurrentText("5")

        self.auto_reset_checkbox = QCheckBox()
        self.auto_reset_checkbox.setToolTip("If checked, simulation auto-resets")
        self.auto_reset_checkbox.setChecked(True)

        self._build_layout()

        # --- status bar ---
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.status.setFont(mono)

        self.threads_label = QLabel(self)
        self.status.addPermanentWidget(self.threads_label)

        # Timer-based simulation (no QThread)
        self.timer = QTimer(self)
        self.timer.setInterval(0)  # as fast as Qt allows
        self.timer.timeout.connect(self._on_timer)  # type: ignore[attr-defined]

        # signal connections
        self.start_button.clicked.connect(self.on_start_clicked)  # type: ignore[attr-defined]
        self.stop_button.clicked.connect(self.on_stop_clicked)  # type: ignore[attr-defined]
        self.reset_button.clicked.connect(self.on_reset_clicked)  # type: ignore[attr-defined]
        self.save_button.clicked.connect(self.on_save_clicked)  # type: ignore[attr-defined]
        self.folder_button.clicked.connect(self.on_folder_clicked)  # type: ignore[attr-defined]
        self.variable_combo.currentIndexChanged.connect(self.on_variable_changed)  # type: ignore[attr-defined]
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)  # type: ignore[attr-defined]
        self.n_combo.currentTextChanged.connect(self.on_n_changed)  # type: ignore[attr-defined]
        self.re_combo.currentTextChanged.connect(self.on_re_changed)  # type: ignore[attr-defined]
        self.k0_combo.currentTextChanged.connect(self.on_k0_changed)  # type: ignore[attr-defined]
        self.cfl_combo.currentTextChanged.connect(self.on_cfl_changed)  # type: ignore[attr-defined]
        self.steps_combo.currentTextChanged.connect(self.on_steps_changed)  # type: ignore[attr-defined]
        self.update_combo.currentTextChanged.connect(self.on_update_changed)  # type: ignore[attr-defined]

        # Ensure single-key shortcuts work regardless of which widget has focus (Win11 combos eat 'C')
        _setup_shortcuts(self)

        # window setup
        import importlib.util

        title_backend = "(SciPy)"
        if importlib.util.find_spec("cupy") is not None:
            import cupy as cp
            try:
                props = cp.cuda.runtime.getDeviceProperties(0)
                gpu_name = props["name"].decode(errors="replace")
                title_backend = f"(CuPyX) {gpu_name}"
            except (RuntimeError, OSError, ValueError, IndexError):
                pass

        self.setWindowTitle(f"2D Turbulence {title_backend} © Mannetroll")
        self.resize(self.sim.px + 40, self.sim.py + 120)

        # Keep-alive buffers for QImage wrappers
        self._last_pixels_rgb: Optional[np.ndarray] = None  # retained for compatibility
        self._last_pixels_u8: Optional[np.ndarray] = None

        # --- FPS from simulation start ---
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()

        # initial draw (omega mode)
        self.sim.set_variable(self.sim.VAR_OMEGA)
        self.variable_combo.setCurrentIndex(3)

        self._update_image(self.sim.get_frame_pixels())
        self._update_status(self.sim.get_time(), self.sim.get_iteration(), None)

        self.on_start_clicked()  # auto-start simulation immediately

    # ------------------------------------------------------------------

    @staticmethod
    def move_widgets(src_layout, dst_layout):
        """Move only widgets from src_layout into dst_layout (ignore spacers)."""
        while src_layout.count() > 0:
            item = src_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                dst_layout.addWidget(w)

    def _build_layout(self):
        """Rebuild the control layout based on the current N."""
        old = self.centralWidget()
        if old is not None:
            old.setParent(None)

        central = QWidget()
        main = QVBoxLayout(central)
        main.addWidget(self.image_label)

        # First row
        row1 = QHBoxLayout()
        row1.addWidget(self.start_button)
        row1.addWidget(self.stop_button)
        row1.addWidget(self.reset_button)
        row1.addWidget(self.save_button)
        row1.addWidget(self.folder_button)
        row1.addWidget(self.cmap_combo)
        row1.addWidget(self.steps_combo)
        row1.addWidget(self.update_combo)
        row1.addWidget(self.auto_reset_checkbox)

        # Second row
        row2 = QHBoxLayout()
        row2.addWidget(self.variable_combo)
        row2.addWidget(self.n_combo)
        row2.addWidget(self.re_combo)
        row2.addWidget(self.k0_combo)
        row2.addWidget(self.cfl_combo)

        # Combine into the single row if large N
        if self.sim.N >= 1024:
            single = QHBoxLayout()
            self.move_widgets(row1, single)
            self.move_widgets(row2, single)
            main.addLayout(single)
        else:
            main.addLayout(row1)
            main.addLayout(row2)

        self.setCentralWidget(central)

    def _maybe_downscale_u8(self, pix: np.ndarray) -> np.ndarray:
        """
        Downscale a 2D uint8 image for display only.
        Uses striding (nearest) to be very fast and avoid float work.
        """
        N = self.sim.N

        if N < 768:
            scale = 1
        elif N <= 1024:
            scale = 2
        elif N <= 3072:
            scale = 4
        else:
            scale = 6

        if scale == 1:
            return np.ascontiguousarray(pix)

        return np.ascontiguousarray(pix[::scale, ::scale])

    def _get_full_field(self, variable: str) -> np.ndarray:
        """
        Return a 2D float32 array (NZ_full × NX_full) for variable:
            'u', 'v', 'kinetic', 'omega'
        """
        S = self.sim.state

        # --------------------------
        # Direct velocity components
        # --------------------------
        if variable == "u":
            field = S.ur_full[0]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        if variable == "v":
            field = S.ur_full[1]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        # --------------------------
        # Kinetic energy |u|
        # --------------------------
        if variable == "kinetic":
            dns_all.dns_kinetic(S)  # fills ur_full[2,:,:]
            field = S.ur_full[2]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        # --------------------------
        # Physical vorticity ω
        # --------------------------
        if variable == "omega":
            dns_all.dns_om2_phys(S)  # fills ur_full[2,:,:]
            field = S.ur_full[2]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        # --------------------------
        # Unknown variable
        # --------------------------
        raise ValueError(f"Unknown variable: {variable}")

    def _update_run_buttons(self) -> None:
        """Enable/disable Start/Stop depending on the timer state."""
        running = self.timer.isActive()
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def on_start_clicked(self) -> None:
        # reset FPS baseline to "new simulation start"
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
        if not self.timer.isActive():
            self.timer.start()

        self._update_run_buttons()

    def on_stop_clicked(self) -> None:
        if self.timer.isActive():
            self.timer.stop()

        self._update_run_buttons()

    def on_step_clicked(self) -> None:
        self.sim.step()
        pixels = self.sim.get_frame_pixels()
        self._update_image(pixels)
        t = self.sim.get_time()
        it = self.sim.get_iteration()
        self._update_status(t, it, fps=None)

    def on_reset_clicked(self) -> None:
        self.on_stop_clicked()
        self.sim.reset_field()
        self._update_image(self.sim.get_frame_pixels())
        self._update_status(self.sim.get_time(), self.sim.get_iteration(), None)
        self.on_start_clicked()

    @staticmethod
    def sci_no_plus(x, decimals=0):
        x = float(x)
        s = f"{x:.{decimals}E}"
        return s.replace("E+", "E").replace("e+", "e")

    def on_folder_clicked(self) -> None:
        # --- Build the default folder name ---
        N = self.sim.N
        Re = self.sim.re
        K0 = self.sim.k0
        CFL = self.sim.cfl
        STEPS = self.sim.get_iteration()

        folder = f"cupyturbo_{N}_{self.sci_no_plus(Re)}_{K0}_{CFL}_{STEPS}"

        # Default root = Desktop
        desktop = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DesktopLocation
        )

        # --- Create non-native dialog (macOS compatible) ---
        dlg = QFileDialog(self)
        dlg.setWindowTitle(f"Case: {folder}")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setDirectory(desktop)

        # Prefill the directory edit field
        for lineedit in dlg.findChildren(QLineEdit):
            lineedit.setText(".")  # type: ignore[attr-defined]

        # Execute dialog
        if dlg.exec():
            base_dir = dlg.selectedFiles()[0]
        else:
            return

        # Build final path
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)

        print(f"[SAVE] Dumping fields to folder: {folder_path}")
        self._dump_pgm_full(self._get_full_field("u"), os.path.join(folder_path, "u_velocity.pgm"))
        self._dump_pgm_full(self._get_full_field("v"), os.path.join(folder_path, "v_velocity.pgm"))
        self._dump_pgm_full(self._get_full_field("kinetic"), os.path.join(folder_path, "kinetic.pgm"))
        self._dump_pgm_full(self._get_full_field("omega"), os.path.join(folder_path, "omega.pgm"))
        print("[SAVE] Completed.")

    def on_save_clicked(self) -> None:
        # determine variable name for filename
        var_name = self.variable_combo.currentText()
        cmap_name = self.cmap_combo.currentText()
        default_name = f"cupyturbo_{var_name}_{cmap_name}.png"

        # Default root = Desktop
        desktop = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DesktopLocation
        )
        initial_path = desktop + "/" + default_name

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save frame",
            initial_path,
            "PNG images (*.png);;All files (*)",
        )

        if path:
            pm = self.image_label.pixmap()
            if pm:
                pm.save(path, "PNG")

    def on_variable_changed(self, index: int) -> None:
        mapping = {
            0: self.sim.VAR_U,
            1: self.sim.VAR_V,
            2: self.sim.VAR_ENERGY,
            3: self.sim.VAR_OMEGA,
            4: self.sim.VAR_STREAM,
        }
        self.sim.set_variable(mapping.get(index, self.sim.VAR_U))
        self._update_image(self.sim.get_frame_pixels())

    def on_cmap_changed(self, name: str) -> None:
        if name in COLOR_MAPS:
            self.current_cmap_name = name
            self._update_image(self.sim.get_frame_pixels())

    def on_n_changed(self, value: str) -> None:
        N = int(value)
        self.sim.set_N(N)

        # 1) Update the image first
        self._update_image(self.sim.get_frame_pixels())

        # 2) Compute new geometry
        new_w = self.image_label.pixmap().width() + 40
        new_h = self.image_label.pixmap().height() + 120
        print("Resize to:", new_w, new_h)

        # 3) Allow the window to shrink (RESET constraints)
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)

        # 4) Now resize → Qt WILL shrink
        self.resize(new_w, new_h)

        # 5) Recenter
        screen = QApplication.primaryScreen().availableGeometry()
        g = self.geometry()
        g.moveCenter(screen.center())
        self.setGeometry(g)
        self._build_layout()
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()

    def _recenter_window(self):
        screen = QApplication.primaryScreen().availableGeometry()
        g = self.geometry()
        g.moveCenter(screen.center())
        self.move(g.topLeft())

    def on_re_changed(self, value: str) -> None:
        self.sim.re = float(value)
        self.sim.reset_field()
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
        self._update_image(self.sim.get_frame_pixels())

    def on_k0_changed(self, value: str) -> None:
        self.sim.k0 = float(value)
        self.sim.reset_field()
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
        self._update_image(self.sim.get_frame_pixels())

    def on_cfl_changed(self, value: str) -> None:
        self.sim.cfl = float(value)
        self.sim.reset_field()
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
        self._update_image(self.sim.get_frame_pixels())

    def on_steps_changed(self, value: str) -> None:
        self.sim.max_steps = int(float(value))

    def on_update_changed(self, value: str) -> None:
        self._update_intervall = int(float(value))

    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        # one DNS step per timer tick
        self.sim.step(self._update_intervall)

        # Count frames since the last GUI update
        self._status_update_counter += 1

        if self._status_update_counter >= self._update_intervall:
            pixels = self.sim.get_frame_pixels()
            self._update_image(pixels)

            # ---- FPS from simulation start ----
            now = time.time()
            elapsed = now - self._sim_start_time
            steps = self.sim.get_iteration() - self._sim_start_iter
            fps = None
            if elapsed > 0 and steps > 0:
                fps = steps / elapsed

            self._update_status(
                self.sim.get_time(),
                self.sim.get_iteration(),
                fps,
            )

            self._status_update_counter = 0

        # Optional auto-reset using STEPS combo
        if self.sim.get_iteration() >= self.sim.max_steps:
            if self.auto_reset_checkbox.isChecked():
                self.sim.reset_field()
                self._sim_start_time = time.time()
                self._sim_start_iter = self.sim.get_iteration()
            else:
                self.timer.stop()
                print("Max steps reached — simulation stopped (Auto-Reset OFF).")

    # ------------------------------------------------------------------
    @staticmethod
    def _dump_pgm_full(arr: np.ndarray, filename: str):
        """
        Write a single component field as PGM
        """
        h, w = arr.shape
        minv = float(arr.min())
        maxv = float(arr.max())

        rng = maxv - minv

        with open(filename, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())

            if rng <= 1e-12:  # constant field
                f.write(bytes([128]) * (w * h))
                return

            # scale to 1..255
            norm = (arr - minv) / rng
            pix = (1.0 + norm * 254.0).round().clip(1, 255).astype(np.uint8)
            f.write(pix.tobytes())

    def _update_image(self, pixels: np.ndarray) -> None:
        """
        Display H×W uint8 pixels using Qt Indexed8 + color table.
        Avoids expanding to H×W×3 RGB in NumPy.
        """
        pixels = np.asarray(pixels, dtype=np.uint8)
        if pixels.ndim != 2:
            return

        pixels = self._maybe_downscale_u8(pixels)
        h, w = pixels.shape

        # Keep numpy buffer alive for QImage
        self._last_pixels_u8 = pixels

        qimg = QImage(
            pixels.data,
            w,
            h,
            w,  # bytesPerLine: 1 byte per pixel
            QImage.Format.Format_Indexed8,
        )

        table = QT_COLOR_TABLES.get(self.current_cmap_name, QT_GRAY_TABLE)
        qimg.setColorTable(table)

        pix = QPixmap.fromImage(qimg, Qt.ImageConversionFlag.NoFormatConversion)
        self.image_label.setPixmap(pix)

    def _update_status(self, t: float, it: int, fps: Optional[float]) -> None:
        fps_str = f"{fps:4.1f}" if fps is not None else " N/a"

        # DPP = Display Pixel Percentage
        N = self.sim.N
        if N < 768:
            dpp = 100  # 1× scale
        elif N <= 1024:
            dpp = 50  # 2× downscale
        elif N <= 3072:
            dpp = 25  # 4× downscale
        else:
            dpp = 17  # 6× downscale (≈16.7%)

        # elapsed wall time since sim start (minutes)
        elapsed_min = (time.time() - self._sim_start_time) / 60.0

        # Viscosity from DNS state
        visc = float(self.sim.state.visc)

        txt = (
            f"FPS: {fps_str} | Iter: {it:5d} | T: {t:6.3f} "
            f"| DPP: {dpp}% | {elapsed_min:4.1f} min | Visc: {visc:12.10f}"
        )

        self.status.showMessage(txt)

    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()

        # rotate variable (V)
        if key == Qt.Key.Key_V:
            idx = self.variable_combo.currentIndex()
            count = self.variable_combo.count()
            self.variable_combo.setCurrentIndex((idx + 1) % count)
            return

        # rotate colormap (C)
        if key == Qt.Key.Key_C:
            idx = self.cmap_combo.currentIndex()
            count = self.cmap_combo.count()
            self.cmap_combo.setCurrentIndex((idx + 1) % count)
            return

        # rotate n_combo (N)
        if key == Qt.Key.Key_N:
            idx = self.n_combo.currentIndex()
            count = self.n_combo.count()
            self.n_combo.setCurrentIndex((idx + 1) % count)
            return

        # rotate Reynolds (R)
        if key == Qt.Key.Key_R:
            idx = self.re_combo.currentIndex()
            count = self.re_combo.count()
            self.re_combo.setCurrentIndex((idx + 1) % count)
            return

        # rotate K0 (K)
        if key == Qt.Key.Key_K:
            idx = self.k0_combo.currentIndex()
            count = self.k0_combo.count()
            self.k0_combo.setCurrentIndex((idx + 1) % count)
            return

        # rotate CFL (L)
        if key == Qt.Key.Key_L:
            idx = self.cfl_combo.currentIndex()
            count = self.cfl_combo.count()
            self.cfl_combo.setCurrentIndex((idx + 1) % count)
            return

        # rotate steps (S)
        if key == Qt.Key.Key_S:
            idx = self.steps_combo.currentIndex()
            count = self.steps_combo.count()
            self.steps_combo.setCurrentIndex((idx + 1) % count)
            return

        # update intervall (U)
        if key == Qt.Key.Key_U:
            idx = self.update_combo.currentIndex()
            count = self.update_combo.count()
            self.update_combo.setCurrentIndex((idx + 1) % count)
            return

        super().keyPressEvent(event)


# ----------------------------------------------------------------------
def main() -> None:
    app = QApplication(sys.argv)

    icon_path = Path(__file__).with_name("scipyturbo.icns")
    icon = QIcon(str(icon_path))
    app.setWindowIcon(icon)

    sim = DnsSimulator(n=384)
    window = MainWindow(sim)
    screen = app.primaryScreen().availableGeometry()
    g = window.geometry()
    g.moveCenter(screen.center())
    window.setGeometry(g)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()