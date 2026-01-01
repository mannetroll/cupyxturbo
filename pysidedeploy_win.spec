[app]
title = cupyxturbo
project_dir = .
input_file = scipyturbo/turbo_main.py
exec_directory = .
project_file =
icon = scipyturbo/scipyturbo.ico

[python]
python_path = .venv/Scripts/python.exe
packages = Nuitka==2.8.9,ordered-set==4.1.0,zstandard==0.25.0

[qt]
qml_files =
excluded_qml_plugins =
modules = Core,DBus,Gui,Widgets
plugins = accessiblebridge,generic,iconengines,imageformats,platforminputcontexts,platforms,styles

[nuitka]
mode = onefile
extra_args = --include-package=scipyturbo --include-package=scipy --include-package-data=scipy --include-module=scipy._cyutility --noinclude-qt-translations --windows-icon-from-ico=scipyturbo/scipyturbo.ico --include-package=cupy --include-package=cupy_backends --include-package-data=cupy --include-package-data=cupy_backends --include-module=cupy_backends.cuda._softlink
