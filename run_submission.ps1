<#
Full submission run: train -> predict on the full test set -> validate.

Everything is logged to logs\submission-<timestamp>.log, and a one-line status
(RUNNING <step> / DONE / FAILED <step>) is kept in logs\submission-status.txt.

Keeps Windows awake while it runs (released automatically when the script
exits); it does not change power settings. Closing the laptop lid can still
sleep it, depending on your lid settings, so leave it open and plugged in.

Usage (from the project root):
    powershell -ExecutionPolicy Bypass -File run_submission.ps1
    powershell -ExecutionPolicy Bypass -File run_submission.ps1 -SampleSize 50000
#>
param(
    # S1 records used by train.py (80% train / 20% holdout for the threshold).
    [int]$SampleSize = 100000,
    [string]$Python = "python",
    [string]$Validator = "C:\Users\abhin\Downloads\6ab10eb3b23ba_student_resource\student_resource\utils\validate_submission.py",
    [string]$Model = "model.joblib",
    [string]$Output = "output"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
New-Item -ItemType Directory -Force logs | Out-Null

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $root "logs\submission-$stamp.log"
$status = Join-Path $root "logs\submission-status.txt"

function Write-Log([string]$message) {
    $line = "$(Get-Date -Format 's')  $message"
    Add-Content -Path $log -Value $line -Encoding utf8
    Write-Host $line
}

function Set-Status([string]$text) {
    Set-Content -Path $status -Value "$text  (at $(Get-Date -Format 's'); log: $log)" -Encoding utf8
}

function Invoke-Step([string]$name, [string]$arguments) {
    # Each step's stdout/stderr go to their own files (follow the .out.log
    # live), and are appended to the main log when the step ends.
    # Start-Process passes the argument string as-is; PowerShell 5.1 mangles
    # embedded quotes when calling native programs directly or via cmd /c.
    $out = Join-Path $root "logs\submission-$stamp-$name.out.log"
    $err = Join-Path $root "logs\submission-$stamp-$name.err.log"
    Write-Log "START $name : $Python $arguments"
    Write-Log "      output: $out"
    Set-Status "RUNNING $name (output: $out)"
    $started = Get-Date
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $process.Handle  # cache the handle so ExitCode is available (PowerShell 5.1 quirk)
    $process.WaitForExit()
    $code = $process.ExitCode
    foreach ($file in @($out, $err)) {
        if ((Test-Path $file) -and (Get-Item $file).Length -gt 0) {
            Add-Content -Path $log -Value "----- $name $(Split-Path -Leaf $file) -----" -Encoding utf8
            Add-Content -Path $log -Value (Get-Content $file -Encoding utf8) -Encoding utf8
        }
    }
    $minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    if ($code -ne 0) {
        Write-Log "FAILED $name (exit code $code) after $minutes min; see $err"
        Set-Status "FAILED $name (exit code $code)"
        exit $code
    }
    Write-Log "DONE $name in $minutes min"
}

# Keep the system awake while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
Add-Type -Namespace Native -Name Power -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
[Native.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

Write-Log "Submission run: sample size $SampleSize, model $Model, output $Output"
try { Write-Log "Python: $Python ($((& $Python --version) -join ' '))" } catch { Write-Log "Python version check failed: $_" }
try {
    Write-Log "Git commit: $((git rev-parse --short HEAD) -join '') (uncommitted files: $((git status --porcelain | Measure-Object).Count))"
} catch { Write-Log "Git info unavailable: $_" }

Invoke-Step "train" "-m src.train --data dataset/train --sample-size $SampleSize --model $Model"
Invoke-Step "predict" "-m src.predict --data dataset/test --model $Model --output $Output"
# Matching file only: the validator's candidate check needs ~8-10 GB on the full file.
Invoke-Step "validate" "`"$Validator`" --matching $Output/matching_results.tsv --candidate $Output/__skip__.tsv --test-dir dataset/test"
Invoke-Step "check-candidates" "-m src.check_candidates --candidates $Output/candidate_pairs.tsv --matching $Output/matching_results.tsv --test-dir dataset/test"

Write-Log "ALL DONE"
Set-Status "DONE"
