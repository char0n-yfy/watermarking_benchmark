param(
  [string]$VenvDir = $(if ($env:WM_BENCH_VENV) { $env:WM_BENCH_VENV } else { ".venv" })
)

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RootDir

. (Join-Path $PSScriptRoot "import-dotenv.ps1")
$DotEnvPath = if ($env:WM_BENCH_DOTENV_PATH) { $env:WM_BENCH_DOTENV_PATH } else { Join-Path $RootDir ".env" }
Import-DotEnvFile -Path $DotEnvPath -OverwriteExisting | Out-Null

if (-not $PSBoundParameters.ContainsKey("VenvDir") -and $env:WM_BENCH_VENV) {
  $VenvDir = $env:WM_BENCH_VENV
}

$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

if ($env:WM_BENCH_INSTALL_PYTHON_DEPS -eq "0") {
  if (-not (Test-Path $PythonExe)) {
    throw "Missing Python virtual environment: $PythonExe. Unset WM_BENCH_INSTALL_PYTHON_DEPS or create the venv manually."
  }
  Write-Host "Python dependency install skipped: WM_BENCH_INSTALL_PYTHON_DEPS=0"
  return
}

if (-not (Test-Path $PythonExe)) {
  $VenvArgs = @("-m", "venv")
  if ($env:WM_BENCH_VENV_SYSTEM_SITE_PACKAGES -and $env:WM_BENCH_VENV_SYSTEM_SITE_PACKAGES -ne "0") {
    $VenvArgs += "--system-site-packages"
  }
  $VenvArgs += $VenvDir

  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 @VenvArgs
  } else {
    & python @VenvArgs
  }
}

& $PythonExe -m pip install --upgrade pip setuptools wheel

if (-not $env:WM_BENCH_INSTALL_SHARP_DEPS -or $env:WM_BENCH_INSTALL_SHARP_DEPS -ne "0") {
  & $PythonExe -m pip install -r requirements.txt -r requirements/sharp.txt
} else {
  & $PythonExe -m pip install -r requirements.txt
}

Write-Host "Python environment is ready: $PythonExe"
