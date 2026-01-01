[app]
title = cupyxturbo
project_dir = .
input_file = scipyturbo/turbo_main.py
exec_directory = .
project_file =
icon = scipyturbo/scipyturbo.icns

[python]
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
mode = onefile
macos.permissions =
extra_args = --include-package=scipyturbo --include-package=scipy --include-package-data=scipy --include-module=scipy._cyutility --noinclude-qt-translations

[buildozer]
mode = debug
recipe_dir =
jars_dir =
ndk_path =
sdk_path =
local_libs =
arch =