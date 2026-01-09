@echo off
setlocal

REM --- Find drive that contains Briefcase\v13.1\bin\nvcc.exe ---
for %%D in (C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
  if exist "%%D:\Briefcase\v13.1\bin\nvcc.exe" (
    set "DRV=%%D"
    goto :found
  )
)

echo ERROR: Could not find Briefcase\v13.1\bin\nvcc.exe on any drive.
exit /b 1

:found
set "CUDA_PATH=%DRV%:\Briefcase\v13.1"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"

REM --- Run entrypoint (no uv required) ---
"%~dp0\.venv\Scripts\python.exe" -m palinstrophy.turbo_main
