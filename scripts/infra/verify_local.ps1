param(
    [string]$EnvFile = "scripts/infra/.env.compose.example",
    [switch]$WithEdge
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$envPath = (Resolve-Path (Join-Path $repoRoot $EnvFile)).Path

Push-Location $repoRoot
try {
    $composeArgs = @("compose", "--env-file", $envPath)
    if ($WithEdge) {
        $composeArgs += @("--profile", "edge")
    }

    & docker @composeArgs config --quiet
    if ($LASTEXITCODE -ne 0) { throw "docker compose config failed" }

    $services = @("db", "redis")
    if ($WithEdge) { $services += "edge" }
    & docker @composeArgs up -d @services
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }

    $deadline = (Get-Date).AddSeconds(90)
    do {
        $rows = & docker @composeArgs ps --format json | ConvertFrom-Json
        $wanted = @($rows | Where-Object { $services -contains $_.Service })
        $healthy = $wanted.Count -eq $services.Count -and @($wanted | Where-Object { $_.Health -eq "healthy" }).Count -eq $services.Count
        if ($healthy) { break }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)

    if (-not $healthy) {
        & docker @composeArgs ps
        throw "Compose services did not become healthy within 90 seconds"
    }

    if ($WithEdge) {
        $edge = Invoke-RestMethod -Uri "http://127.0.0.1:8080/healthz" -TimeoutSec 5
        if ($edge.status -ne "ok") { throw "edge health endpoint returned an unexpected payload" }
    }

    Write-Host "[OK] Compose configuration is valid and requested services are healthy."
}
finally {
    Pop-Location
}
