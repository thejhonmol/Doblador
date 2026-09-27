[CmdletBinding()]
param(
    [int[]]$Ports = @(8000, 3000, 5173)
)
& "$PSScriptRoot\prepare_ports.ps1" -Ports $Ports
exit $LASTEXITCODE
