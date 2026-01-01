# turboapp.spec
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
    collect_data_files,
)

hiddenimports = []
binaries = []
datas = []

# --- CuPy (GPU optional, but safe to include if installed in the build env) ---
hiddenimports += collect_submodules("cupy")
binaries += collect_dynamic_libs("cupy")
datas += collect_data_files("cupy")

# --- fastrlock (often present with cupy installs; include if installed) ---
# If fastrlock isn't installed, remove these 3 lines.
hiddenimports += collect_submodules("fastrlock")
binaries += collect_dynamic_libs("fastrlock")
datas += collect_data_files("fastrlock")

a = Analysis(
    ["scipyturbo/turbo_main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
)

pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="turboapp",
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="turboapp",
)