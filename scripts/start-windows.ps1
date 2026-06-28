param(
  [int]$ApiPort = 8000,
  [int]$WebPort = 3000,
  [string]$ApiHost = "127.0.0.1",
  [string]$WebHost = "127.0.0.1",
  [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RootDir

$VenvDir = if ($env:WM_BENCH_VENV) { $env:WM_BENCH_VENV } else { ".venv" }
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$RunsRoot = if ($env:WM_BENCH_RUNS_ROOT) { $env:WM_BENCH_RUNS_ROOT } else { Join-Path $RootDir "runs\local" }
$LogDir = if ($env:WM_BENCH_LOG_DIR) { $env:WM_BENCH_LOG_DIR } else { Join-Path $RunsRoot "logs" }
$PidDir = if ($env:WM_BENCH_PID_DIR) { $env:WM_BENCH_PID_DIR } else { Join-Path $RunsRoot "pids" }
$DbPath = if ($env:WM_BENCH_DB_PATH) { $env:WM_BENCH_DB_PATH } else { Join-Path $RunsRoot "wmbench.sqlite" }
$ResourcesRoot = if ($env:WM_BENCH_RESOURCES_ROOT) { $env:WM_BENCH_RESOURCES_ROOT } else { Join-Path $RootDir "resources" }

New-Item -ItemType Directory -Force -Path `
  (Join-Path $ResourcesRoot "datasets"), `
  (Join-Path $ResourcesRoot "weights"), `
  $RunsRoot, `
  (Split-Path $DbPath), `
  $LogDir, `
  $PidDir | Out-Null

if (-not (Test-Path $PythonExe)) {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 -m venv $VenvDir
  } else {
    & python -m venv $VenvDir
  }
}

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r apps/api/requirements.txt
& $PythonExe -m pip install -r apps/worker/requirements.txt

& corepack enable
& corepack pnpm install

$env:APP_ENV = if ($env:APP_ENV) { $env:APP_ENV } else { "development" }
$env:WM_BENCH_DATA_ROOT = if ($env:WM_BENCH_DATA_ROOT) { $env:WM_BENCH_DATA_ROOT } else { "$RootDir" }
$env:WM_BENCH_RESOURCES_ROOT = "$ResourcesRoot"
$env:WM_BENCH_RUNS_ROOT = "$RunsRoot"
$env:WM_BENCH_DB_PATH = "$DbPath"
$env:WM_BENCH_DEVICE = if ($env:WM_BENCH_DEVICE) { $env:WM_BENCH_DEVICE } else { $Device }
$env:WM_BENCH_WORKER_POLL_SECONDS = if ($env:WM_BENCH_WORKER_POLL_SECONDS) { $env:WM_BENCH_WORKER_POLL_SECONDS } else { "2" }
$env:WM_BENCH_RUN_TIMEOUT_SECONDS = if ($env:WM_BENCH_RUN_TIMEOUT_SECONDS) { $env:WM_BENCH_RUN_TIMEOUT_SECONDS } else { "3600" }
$env:API_HOST = $ApiHost
$env:API_PORT = "$ApiPort"

function Stop-PidFile {
  param([string]$Name)
  $PidFile = Join-Path $PidDir "$Name.pid"
  if (Test-Path $PidFile) {
    $ExistingPid = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($ExistingPid) {
      Stop-Process -Id ([int]$ExistingPid) -Force -ErrorAction SilentlyContinue
      for ($i = 0; $i -lt 20; $i++) {
        $Process = Get-Process -Id ([int]$ExistingPid) -ErrorAction SilentlyContinue
        if (-not $Process) {
          break
        }
        Start-Sleep -Milliseconds 250
      }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
  }
}

function Wait-HttpOk {
  param(
    [string]$Name,
    [string]$Url,
    [string]$LogHint
  )
  for ($i = 0; $i -lt 60; $i++) {
    try {
      $Response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
      if ($Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500) {
        return
      }
    } catch {
      Start-Sleep -Seconds 1
    }
  }
  throw "$Name did not become reachable: $Url. Check logs: $LogHint"
}

Stop-PidFile "api"
Stop-PidFile "worker"
Stop-PidFile "web"

$ApiProcess = Start-Process -FilePath $PythonExe `
  -ArgumentList @("-m", "uvicorn", "app.main:app", "--app-dir", "apps/api", "--host", $ApiHost, "--port", "$ApiPort") `
  -RedirectStandardOutput (Join-Path $LogDir "api.out.log") `
  -RedirectStandardError (Join-Path $LogDir "api.err.log") `
  -WindowStyle Hidden `
  -PassThru
Set-Content -Path (Join-Path $PidDir "api.pid") -Value $ApiProcess.Id

$WorkerProcess = Start-Process -FilePath $PythonExe `
  -ArgumentList @("apps/worker/local_worker.py", "--poll-seconds", $env:WM_BENCH_WORKER_POLL_SECONDS) `
  -RedirectStandardOutput (Join-Path $LogDir "worker.out.log") `
  -RedirectStandardError (Join-Path $LogDir "worker.err.log") `
  -WindowStyle Hidden `
  -PassThru
Set-Content -Path (Join-Path $PidDir "worker.pid") -Value $WorkerProcess.Id

$env:NEXT_PUBLIC_API_BASE_URL = "http://localhost:$ApiPort"
$WebProcess = Start-Process -FilePath "cmd.exe" `
  -ArgumentList @("/c", "corepack pnpm --filter @wm-bench/web dev --hostname $WebHost --port $WebPort") `
  -RedirectStandardOutput (Join-Path $LogDir "web.out.log") `
  -RedirectStandardError (Join-Path $LogDir "web.err.log") `
  -WindowStyle Hidden `
  -PassThru
Set-Content -Path (Join-Path $PidDir "web.pid") -Value $WebProcess.Id

$CheckApiHost = if ($ApiHost -eq "0.0.0.0") { "127.0.0.1" } else { $ApiHost }
$CheckWebHost = if ($WebHost -eq "0.0.0.0") { "127.0.0.1" } else { $WebHost }

Wait-HttpOk "API" "http://$CheckApiHost`:$ApiPort/health" (Join-Path $LogDir "api.err.log")
Wait-HttpOk "Web UI" "http://$CheckWebHost`:$WebPort" (Join-Path $LogDir "web.err.log")

Write-Host "WM Bench local services started."
Write-Host ""
Write-Host "Web UI:     http://$CheckWebHost`:$WebPort"
Write-Host "API health: http://$CheckApiHost`:$ApiPort/health"
Write-Host ""
Write-Host "Logs:"
Write-Host "  $LogDir"
Write-Host ""
Write-Host "Stop:"
Write-Host "  Stop-Process -Id (Get-Content `"$PidDir\api.pid`"), (Get-Content `"$PidDir\worker.pid`"), (Get-Content `"$PidDir\web.pid`") -Force"
