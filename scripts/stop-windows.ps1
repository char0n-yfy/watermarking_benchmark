param(
  [int]$ApiPort = 8000,
  [int]$WebPort = 3000
)

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RootDir

. (Join-Path $PSScriptRoot "import-dotenv.ps1")
Import-DotEnvFile -Path (Join-Path $RootDir ".env") -OverwriteExisting | Out-Null

$RunsRoot = if ($env:WM_BENCH_RUNS_ROOT) { $env:WM_BENCH_RUNS_ROOT } else { Join-Path $RootDir "runs" }
$PidDir = if ($env:WM_BENCH_PID_DIR) { $env:WM_BENCH_PID_DIR } else { Join-Path $RunsRoot "pids" }
$RootEscaped = [regex]::Escape($RootDir.Path)

function Stop-PidFile {
  param([string]$Name)

  $PidFile = Join-Path $PidDir "$Name.pid"
  if (-not (Test-Path $PidFile)) {
    return
  }

  $ExistingPid = Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($ExistingPid -match "^\d+$") {
    Stop-Process -Id ([int]$ExistingPid) -Force -ErrorAction SilentlyContinue
  }
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

function Stop-RepoProcess {
  param([int]$ProcessId)

  $ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
  if (
    $ProcessInfo -and
    $ProcessInfo.CommandLine -and
    (
      $ProcessInfo.CommandLine -match $RootEscaped -or
      $ProcessInfo.CommandLine -match "local_worker\.py|uvicorn app\.main:app|@wm-bench/web dev|next dev"
    )
  ) {
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
  }
}

foreach ($Name in "api", "worker", "web") {
  Stop-PidFile $Name
}

Get-NetTCPConnection -LocalPort $WebPort, $ApiPort -State Listen -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-RepoProcess -ProcessId $_.OwningProcess }

Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
  Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match "local_worker\.py|uvicorn app\.main:app|@wm-bench/web dev|next dev" -and
    ($_.CommandLine -match $RootEscaped -or $_.CommandLine -match "apps[\\/]worker[\\/]local_worker\.py|uvicorn app\.main:app|@wm-bench/web dev|next dev")
  } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Write-Host "WM Bench local services stopped."
