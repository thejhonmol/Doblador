<#
    prepare_ports.ps1

    Frees system ports (8000, 3000, 5173) before startup, safely distinguishing:
      1. An existing instance of OUR OWN service occupying the port (terminates it cleanly).
      2. An UNRELATED third-party process occupying the port (aborts to avoid killing user apps).

    Detection is verified via HTTP health signatures, never by process image name.
    Exit code: 0 if ports are free, 1 if manual intervention is required.
#>
[CmdletBinding()]
param(
    [int[]]$Ports = @(8000, 3000, 5173)
)

$ErrorActionPreference = 'Stop'

# Expected HTTP response signature for each service
$Signatures = @{
    8000 = @{ Path = '/';               Pattern = 'Python microservices are running' }
    3000 = @{ Path = '/api/health';     Pattern = 'doblador-orchestrator' }
    5173 = @{ Path = '/';               Pattern = 'AI Dubbing Studio' }
}

function Get-PortOwner {
    param([int]$Port)
    $line = netstat -aon 2>$null |
        Where-Object { $_ -match "^\s*TCP\s+\S+:$Port\s" -and $_ -match 'LISTENING' } |
        Select-Object -First 1
    if (-not $line) { return $null }
    $procId = [int](($line -split '\s+')[-1])
    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    $name = if ($proc) { $proc.ProcessName } else { 'unknown' }
    return [pscustomobject]@{ Pid = $procId; Name = $name }
}

function Test-OurService {
    param([int]$Port)
    $sig = $Signatures[$Port]
    if (-not $sig) { return $false }
    try {
        $r = Invoke-WebRequest -Uri ("http://127.0.0.1:{0}{1}" -f $Port, $sig.Path) -TimeoutSec 3 -UseBasicParsing
        return ($r.Content -match [regex]::Escape($sig.Pattern))
    } catch {
        return $false
    }
}

$problems = @()
$closed = @()

foreach ($port in $Ports) {
    $owner = Get-PortOwner -Port $port
    if (-not $owner) {
        Write-Host ("      -> Port {0} is free." -f $port) -ForegroundColor DarkGray
        continue
    }

    if (Test-OurService -Port $port) {
        Write-Host ("      -> Port {0} occupied by OUR service (PID {1}, {2}). Terminating..." -f $port, $owner.Pid, $owner.Name) -ForegroundColor Yellow
        try {
            Stop-Process -Id $owner.Pid -Force -ErrorAction Stop
            Start-Sleep -Milliseconds 900
            if (Get-PortOwner -Port $port) {
                $problems += "Port $port remains occupied after stopping PID $($owner.Pid)."
            } else {
                $closed += "$port (PID $($owner.Pid))"
            }
        } catch {
            $problems += "Could not stop PID $($owner.Pid) on port ${port}: $($_.Exception.Message)"
        }
    } else {
        $problems += "Port $port is in use by an EXTERNAL process: PID $($owner.Pid) ($($owner.Name)). Please close it or reconfigure the port."
    }
}

if ($closed.Count -gt 0) {
    Write-Host ("      -> Previous instances terminated: {0}" -f ($closed -join ', ')) -ForegroundColor Yellow
}

if ($problems.Count -gt 0) {
    Write-Host ""
    Write-Host "[ERROR] Cannot start system:" -ForegroundColor Red
    foreach ($p in $problems) { Write-Host ("        - {0}" -f $p) -ForegroundColor Red }
    exit 1
}

exit 0
