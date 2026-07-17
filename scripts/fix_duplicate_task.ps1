# One-time fix: disable the DUPLICATE scheduled task.
#
# Two tasks were both firing at 10:00 and both launching the full pipeline
# (`main.py run`), so two processes fought over the 12 GB GPU every morning ->
# VRAM overflow -> the hours-long ASR/Ollama stalls. Keep the newer task
# (BilibiliSummarizer_Daily = run_and_hibernate.py, which also hibernates) and
# disable the older one (BilibiliSummarizerAutoRun = run_pipeline.bat).
#
# Disable (not delete) so it is reversible. Run once; it self-elevates (UAC).

$ErrorActionPreference = "Stop"

# Self-elevate if not admin
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

$dup = "BilibiliSummarizerAutoRun"
$keep = "BilibiliSummarizer_Daily"

$t = Get-ScheduledTask -TaskName $dup -ErrorAction SilentlyContinue
if ($t) {
    Disable-ScheduledTask -TaskName $dup | Out-Null
    Write-Host "Disabled duplicate task: $dup" -ForegroundColor Green
} else {
    Write-Host "Duplicate task '$dup' not found (already removed?)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Current state of the project's tasks:"
Get-ScheduledTask | Where-Object { $_.TaskName -like 'Bilibili*' } |
    Select-Object TaskName, State | Format-Table -Auto

Write-Host ""
Write-Host "Done. Only '$keep' should be Ready now; '$dup' should be Disabled." -ForegroundColor Green
Read-Host "Press Enter to close"
