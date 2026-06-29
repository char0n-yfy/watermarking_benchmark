param(
  [string]$EnvFile
)

function Import-DotEnvFile {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [switch]$OverwriteExisting
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    return $false
  }

  Get-Content -LiteralPath $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
      return
    }
    $parts = $line.Split("=", 2)
    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    if (-not $name) {
      return
    }
    if (-not $OverwriteExisting -and (Test-Path -Path "Env:$name")) {
      return
    }
    Set-Item -Path "Env:$name" -Value $value
  }

  return $true
}

if ($EnvFile) {
  Import-DotEnvFile -Path $EnvFile | Out-Null
}
