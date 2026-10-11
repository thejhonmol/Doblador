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

        $sitePathValue = "../zhowplus/dist"
        $sitePathLine = Get-Content -LiteralPath $envPath | Where-Object { $_ -match '^\s*ZHOWPLUS_LANDING_PATH=' } | Select-Object -Last 1
        if ($sitePathLine) {
            $sitePathValue = ($sitePathLine -replace '^\s*ZHOWPLUS_LANDING_PATH=', '').Trim().Trim('"').Trim("'")
        }
        if ([System.IO.Path]::IsPathRooted($sitePathValue)) {
            $sitePath = [System.IO.Path]::GetFullPath($sitePathValue)
        } else {
            $sitePath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $sitePathValue))
        }

        $landingIndex = Join-Path $sitePath "index.html"
        $appIndex = Join-Path $sitePath "app\index.html"
        if (-not (Test-Path -LiteralPath $landingIndex -PathType Leaf) -or -not (Test-Path -LiteralPath $appIndex -PathType Leaf)) {
            $zhowplusRepo = Split-Path -Parent $sitePath
            throw @"
Bundled ZhowPlus site is missing at: $sitePath
Build it before starting the edge:
  npm --prefix "$repoRoot\frontend" run build
  python "$zhowplusRepo\scripts\build_site.py" --app-dist "$repoRoot\frontend\dist" --out "$sitePath"
Set ZHOWPLUS_LANDING_PATH to that dist directory in the ignored *.local env file.
"@
        }
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
        $landing = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8080/" -TimeoutSec 5
        if ($landing.StatusCode -ne 200) { throw "landing page did not return HTTP 200" }
        $app = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8080/app/" -TimeoutSec 5
        if ($app.StatusCode -ne 200) { throw "bundled app did not return HTTP 200" }
    }

    Write-Host "[OK] Compose configuration is valid and requested services are healthy."
}
finally {
    Pop-Location
}
