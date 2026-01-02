# macos.spec
# Build:
#   rm -rf build dist
#   uv pip install pyinstaller
#   uv run pyinstaller macos.spec
#   ./dist/cupyturbo.app/Contents/MacOS/scipyturbo
#   open -n ./dist/cupyturbo.app
#

a = Analysis(
    ["scipyturbo/turbo_main.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="scipyturbo",
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="cupyxturbo",
)

app = BUNDLE(
    coll,
    name="cupyxturbo.app",
    icon="scipyturbo/scipyturbo.icns",
    bundle_identifier="se.mannetroll.scipyturbo",
)