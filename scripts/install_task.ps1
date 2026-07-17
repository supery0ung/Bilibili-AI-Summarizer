# Run this script once (as admin) to register the scheduled task.
# It self-elevates (UAC) automatically.
#
# Registers ONE daily task (BilibiliSummarizer_Daily -> scheduled_entry.ps1),
# and disables the older duplicate (BilibiliSummarizerAutoRun -> run_pipeline.bat)
# if present. The duplicate caused two pipelines to run at 10:00 concurrently and
# fight over the 12 GB GPU -> hours-long ASR/Ollama stalls. Only ONE pipeline task
# must exist.

$ErrorActionPreference = "Stop"

# Self-elevate if not admin
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

# ── Remove the old duplicate task if it exists ──────────────────────────────
$dup = Get-ScheduledTask -TaskName "BilibiliSummarizerAutoRun" -ErrorAction SilentlyContinue
if ($dup) {
    Unregister-ScheduledTask -TaskName "BilibiliSummarizerAutoRun" -Confirm:$false
    Write-Host "Removed duplicate task: BilibiliSummarizerAutoRun" -ForegroundColor Yellow
}

$workdir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$wrapper = Join-Path $workdir "scripts\scheduled_entry.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper`"" `
    -WorkingDirectory $workdir

$trigger = New-ScheduledTaskTrigger -Daily -At "10:00AM"

$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

# Interactive: task runs in the logged-on user's session (needed for idle
# detection and the Playwright browser upload).
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName "BilibiliSummarizer_Daily" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Daily 10AM: fetch+process Bilibili videos, then hibernate if idle" `
    -Force

Write-Host ""
Write-Host "Task registered successfully!" -ForegroundColor Green
Write-Host "Name   : BilibiliSummarizer_Daily (the ONLY pipeline task)"
Write-Host "Trigger: Daily at 10:00 AM (wakes PC from sleep/hibernate)"
Write-Host "Action : powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$wrapper`""
Write-Host ""
Write-Host "IMPORTANT: The PC must be in Sleep or Hibernate (not fully shut down)"
Write-Host "for the wake-up to work. Use Sleep/Hibernate instead of Shut Down."
Write-Host ""
Read-Host "Press Enter to close"
