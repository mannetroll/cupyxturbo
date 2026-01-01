[app]
title = cupyxturbo

# repo root (where pyproject.toml lives)
project_dir = .

# entrypoint used by your cli script "turbulence"
input_file = scipyturbo/turbo_main.py

# where the output is generated (relative to project_dir)
exec_directory = .

# optional (leave empty)
project_file = 

# app icon (relative to project_dir)
icon = scipyturbo/scipyturbo.icns

[python]

# note = keep absolute (must be a real interpreter path)
python_path = /Users/drtobbe/github/cupyxturbo/.venv/bin/python
packages = Nuitka==2.8.9,ordered-set==4.1.0,zstandard==0.25.0

[qt]
qml_files = 
excluded_qml_plugins = 
modules = Core,DBus,Gui,Widgets
plugins = accessiblebridge,egldeviceintegrations,generic,iconengines,imageformats,platforminputcontexts,platforms,platforms/darwin,platformthemes,styles,xcbglintegrations

[android]
wheel_pyside = 
wheel_shiboken = 
plugins = 

[nuitka]
extra_args = 
	--include-package=scipyturbo
	--include-package=scipy
	--include-package-data=scipy
	--include-module=scipy._cyutility
	--noinclude-qt-translations
macos.permissions = 
mode = onefile

[buildozer]
mode = debug
recipe_dir = 
jars_dir = 
ndk_path = 
sdk_path = 
local_libs = 
arch = 

