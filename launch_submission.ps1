<#
Start run_submission.ps1 fully detached, as a one-off Windows scheduled task
running as the current user. The task is owned by Task Scheduler, so it keeps
running after the terminal, Claude Code or the network go away. It is allowed
to start and keep running on battery and has no time limit.

Usage (from the project root):
    powershell -ExecutionPolicy Bypass -File launch_submission.ps1
    powershell -ExecutionPolicy Bypass -File launch_submission.ps1 -SampleSize 50000

Check progress:   Get-Content logs\submission-status.txt
Follow the log:   Get-Content (Get-ChildItem logs\submission-*.log | Sort-Object LastWriteTime | Select-Object -Last 1) -Tail 20 -Wait
Stop it:          Stop-ScheduledTask -TaskName EntityResolutionSubmission
Remove the task:  Unregister-ScheduledTask -TaskName EntityResolutionSubmission -Confirm:$false
#>
param(
    [int]$SampleSize = 100000,
    [string]$TaskName = "EntityResolutionSubmission"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $root "run_submission.ps1"
$python = (Get-Command python).Source

$action = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $root `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`" -SampleSize $SampleSize -Python `"$python`""
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Settings $settings -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 3
$info = Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo
Write-Host "Started task '$TaskName' (state: $((Get-ScheduledTask -TaskName $TaskName).State), last run: $($info.LastRunTime))"
Write-Host "Status file: $(Join-Path $root 'logs\submission-status.txt')"
