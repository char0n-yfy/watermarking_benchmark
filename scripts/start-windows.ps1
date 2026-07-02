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

. (Join-Path $PSScriptRoot "import-dotenv.ps1")
Import-DotEnvFile -Path (Join-Path $RootDir ".env") -OverwriteExisting | Out-Null

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

if (-not $env:WM_BENCH_INSTALL_SHARP_DEPS) {
  $env:WM_BENCH_INSTALL_SHARP_DEPS = "1"
}

& (Join-Path $PSScriptRoot "bootstrap-python.ps1") -VenvDir $VenvDir

function Initialize-Pnpm {
  $PnpmStoreDir = Join-Path $RootDir ".pnpm-store"
  New-Item -ItemType Directory -Force -Path $PnpmStoreDir | Out-Null

  $PnpmVersion = ""
  try {
    $PnpmVersion = (& corepack pnpm --version 2>$null | Select-Object -First 1).ToString().Trim()
  } catch {
  }

  if (-not $PnpmVersion) {
    try {
      & corepack enable 2>&1 | ForEach-Object {
        if ($_ -match "EPERM|Internal Error") {
          Write-Host "corepack enable warning (non-fatal): $_"
        }
      }
    } catch {
      Write-Host "corepack enable warning (non-fatal): $_"
    }
  }

  $env:CI = "true"
  & corepack pnpm install --store-dir $PnpmStoreDir
  if ($LASTEXITCODE -ne 0) {
    throw "pnpm install failed"
  }
}

Initialize-Pnpm

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

function Enable-LibreHardwareMonitorWebServer {
  param([string]$ExecutablePath)

  if ([string]::IsNullOrWhiteSpace($ExecutablePath)) {
    return $false
  }

  $ParentDir = Split-Path -Path $ExecutablePath -Parent
  if ([string]::IsNullOrWhiteSpace($ParentDir)) {
    return $false
  }

  $ConfigPath = Join-Path $ParentDir "LibreHardwareMonitor.config"
  if (-not (Test-Path -LiteralPath $ConfigPath)) {
    return $false
  }

  $Config = Get-Content -LiteralPath $ConfigPath -Raw
  $Updated = $Config
  if ($Config -match 'key="runWebServerMenuItem"') {
    $Updated = $Config -replace 'key="runWebServerMenuItem" value="false"', 'key="runWebServerMenuItem" value="true"'
  } else {
    $Updated = $Config -replace '(</appSettings>)', "    <add key=`"runWebServerMenuItem`" value=`"true`" />`r`n  `$1"
  }

  if ($Updated -eq $Config) {
    return $false
  }

  Set-Content -LiteralPath $ConfigPath -Value $Updated -Encoding UTF8
  Write-Host "Enabled LibreHardwareMonitor web server in config."
  return $true
}

function Test-LibreHardwareMonitorHttpReady {
  param([int]$Port = 8085)

  try {
    $Payload = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/data.json" -TimeoutSec 2
    return [bool]$Payload
  } catch {
    return $false
  }
}

function Test-LibreHardwareMonitorWmiReady {
  try {
    $Namespaces = Get-CimInstance -Namespace root -ClassName __NAMESPACE -ErrorAction Stop
    if (-not ($Namespaces | Where-Object { $_.Name -eq "LibreHardwareMonitor" })) {
      return $null
    }

    $Sensors = Get-CimInstance -Namespace root/LibreHardwareMonitor -ClassName Sensor -ErrorAction Stop
    $Package = $Sensors |
      Where-Object { $_.SensorType -eq "Power" -and $_.Name -eq "CPU Package" } |
      Select-Object -First 1
    if ($Package -and $Package.Value -gt 0) {
      return [double]$Package.Value
    }
  } catch {
  }
  return $null
}

function Start-LibreHardwareMonitorIfAvailable {
  if ($env:WM_BENCH_SKIP_LHM -eq "1") {
    return
  }

  $Candidates = @()
  if ($env:WM_BENCH_LHM_PATH) {
    $Candidates += $env:WM_BENCH_LHM_PATH.Trim()
  }
  $Candidates += @(
    (Join-Path $RootDir "LibreHardwareMonitor\LibreHardwareMonitor.exe"),
    (Join-Path (Split-Path $RootDir -Parent) "LibreHardwareMonitor\LibreHardwareMonitor.exe"),
    (Join-Path $env:ProgramFiles "LibreHardwareMonitor\LibreHardwareMonitor.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "LibreHardwareMonitor\LibreHardwareMonitor.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\LibreHardwareMonitor\LibreHardwareMonitor.exe")
  )

  foreach ($Candidate in $Candidates) {
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
      continue
    }
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
      continue
    }

    $Resolved = $null
    try {
      $Resolved = (Resolve-Path -LiteralPath $Candidate -ErrorAction Stop).Path
    } catch {
      continue
    }
    if ([string]::IsNullOrWhiteSpace($Resolved)) {
      continue
    }

    $ConfigChanged = Enable-LibreHardwareMonitorWebServer -ExecutablePath $Resolved

    $Existing = @(Get-Process -Name "LibreHardwareMonitor" -ErrorAction SilentlyContinue)
    if ($Existing.Count -gt 1) {
      Write-Host "Multiple LibreHardwareMonitor instances detected; restarting a single instance."
      $Existing | Stop-Process -Force
      Start-Sleep -Seconds 1
      $Existing = @()
      $ConfigChanged = $true
    }

    if ($ConfigChanged) {
      if ($Existing) {
        $Existing | Stop-Process -Force
        Start-Sleep -Seconds 1
      }
      Start-Process -FilePath $Resolved -WindowStyle Minimized | Out-Null
      Start-Sleep -Seconds 2
      Write-Host "LibreHardwareMonitor started: $Resolved"
    } elseif (-not $Existing) {
      Start-Process -FilePath $Resolved -WindowStyle Minimized | Out-Null
      Start-Sleep -Seconds 2
      Write-Host "LibreHardwareMonitor started: $Resolved"
    } elseif (-not (Test-LibreHardwareMonitorHttpReady) -and -not (Test-LibreHardwareMonitorWmiReady)) {
      Write-Host "Restarting LibreHardwareMonitor to enable HTTP sensor API."
      $Existing | Stop-Process -Force
      Start-Sleep -Seconds 1
      Start-Process -FilePath $Resolved -WindowStyle Minimized | Out-Null
      Start-Sleep -Seconds 2
      Write-Host "LibreHardwareMonitor restarted: $Resolved"
    } else {
      Write-Host "LibreHardwareMonitor already running."
    }
    return
  }

  Write-Host "Tip: install LibreHardwareMonitor for accurate CPU package power readings."
}

function Wait-LibreHardwareMonitorReady {
  param([int]$TimeoutSeconds = 30)

  $Port = if ($env:WM_BENCH_LHM_PORT) { [int]$env:WM_BENCH_LHM_PORT } else { 8085 }

  if (Test-LibreHardwareMonitorHttpReady -Port $Port) {
    Write-Host "LibreHardwareMonitor HTTP API ready on port $Port."
    return
  }

  $ReadyWatts = Test-LibreHardwareMonitorWmiReady
  if ($null -ne $ReadyWatts) {
    Write-Host "LibreHardwareMonitor WMI ready (CPU Package: $([math]::Round($ReadyWatts, 1)) W)."
    return
  }

  for ($i = 0; $i -lt $TimeoutSeconds; $i++) {
    Start-Sleep -Seconds 1
    if (Test-LibreHardwareMonitorHttpReady -Port $Port) {
      Write-Host "LibreHardwareMonitor HTTP API ready on port $Port."
      return
    }
    $ReadyWatts = Test-LibreHardwareMonitorWmiReady
    if ($null -ne $ReadyWatts) {
      Write-Host "LibreHardwareMonitor WMI ready (CPU Package: $([math]::Round($ReadyWatts, 1)) W)."
      return
    }
  }

  Write-Host "LibreHardwareMonitor sensor API not ready yet; API will retry during startup."
  Write-Host "In LHM: Options -> Remote Web Server -> Run (default port 8085)."
}

Start-LibreHardwareMonitorIfAvailable
Wait-LibreHardwareMonitorReady

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
