<#
.SYNOPSIS
  Register (or remove) a Windows Task Scheduler task that runs Plume's idle-aware
  trainer at logon. The trainer itself pauses while a Steam game runs or another
  program uses the GPU heavily, so it never competes with games.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install_idle_trainer.ps1
  powershell -ExecutionPolicy Bypass -File scripts\install_idle_trainer.ps1 -Remove
#>
param(
  [switch]$Remove,
  [string]$TaskName = "Plume idle trainer"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

if ($Remove) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
  Write-Host "Removed scheduled task '$TaskName'."
  exit 0
}

$uv = (Get-Command uv).Source
$action = New-ScheduledTaskAction -Execute $uv `
  -Argument "run plume train --when-idle" `
  -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries:$false `
  -DontStopIfGoingOnBatteries:$false `
  -ExecutionTimeLimit (New-TimeSpan -Days 30) `
  -Priority 7
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
  -Settings $settings -Description "Trains Plume's PPO landing agent while the PC is idle." | Out-Null
Write-Host "Registered '$TaskName' (runs 'uv run plume train --when-idle' at logon in $repo)."
Write-Host "Progress: uv run plume train --status    Remove: -Remove"
