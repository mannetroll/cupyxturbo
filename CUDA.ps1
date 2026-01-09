$rel = "Briefcase\v13.1\bin\nvcc.exe"
$drv = Get-PSDrive -PSProvider FileSystem |
  Where-Object { Test-Path (Join-Path $_.Root $rel) } |
  Select-Object -First 1 -ExpandProperty Name

$env:CUDA_PATH = "$drv`:\Briefcase\v13.1"
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
$env:Path = "$env:CUDA_PATH\bin\x64;$env:Path"

.\.venv\Scripts\python.exe -m palinstrophy.turbo_main