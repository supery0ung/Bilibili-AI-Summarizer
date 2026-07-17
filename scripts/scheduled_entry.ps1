param(
    [switch]$SelfTest
)

# Scheduled Task entrypoint.
#
# Task Scheduler can report success even when the Python action exits before
# run_and_hibernate.py initializes its own daily log. Keep this wrapper tiny and
# write a launch log before invoking Python so wake-time startup failures leave
# evidence.

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$LogDir = Join-Path $ProjectRoot "logs"
$LaunchLog = Join-Path $LogDir ("scheduled_launch_{0:yyyy-MM-dd}.log" -f (Get-Date))

$Python = $env:BILIBILI_SUMMARIZER_PYTHON
if (-not $Python) {
    $LocalPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $LocalPython) {
        $Python = $LocalPython
    }
    else {
        $Python = "python.exe"
    }
}
$Script = Join-Path $ProjectRoot "scripts\run_and_hibernate.py"

function Write-LaunchLog {
    param([string]$Message)
    $line = "{0:yyyy-MM-dd HH:mm:ss.fff} {1}" -f (Get-Date), $Message
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        try {
            $stream = [System.IO.File]::Open(
                $LaunchLog,
                [System.IO.FileMode]::Append,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::ReadWrite
            )
            try {
                $writer = [System.IO.StreamWriter]::new($stream, [System.Text.Encoding]::UTF8)
                try {
                    $writer.WriteLine($line)
                    return
                }
                finally {
                    $writer.Dispose()
                }
            }
            finally {
                $stream.Dispose()
            }
        }
        catch {
            if ($attempt -eq 10) {
                throw
            }
            Start-Sleep -Milliseconds 250
        }
    }
}

try {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

    Write-LaunchLog "=== Scheduled wrapper starting ==="
    Write-LaunchLog "User=$env:USERNAME Computer=$env:COMPUTERNAME"
    Write-LaunchLog "ProjectRoot=$ProjectRoot"
    Write-LaunchLog "Python=$Python Exists=$(Test-Path -LiteralPath $Python)"
    Write-LaunchLog "Script=$Script Exists=$(Test-Path -LiteralPath $Script)"

    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Python executable not found: $Python"
    }
    if (-not (Test-Path -LiteralPath $Script)) {
        throw "Scheduled Python script not found: $Script"
    }

    Set-Location -LiteralPath $ProjectRoot
    if ($SelfTest) {
        Write-LaunchLog "SelfTest requested; not invoking pipeline."
        Write-LaunchLog "=== Scheduled wrapper self-test OK ==="
        exit 0
    }

    Write-LaunchLog "Invoking Python..."

    & $Python $Script *>> $LaunchLog
    $exitCode = $LASTEXITCODE

    Write-LaunchLog "Python exited with code $exitCode"
    exit $exitCode
}
catch {
    Write-LaunchLog "FATAL: $($_.Exception.Message)"
    Write-LaunchLog "=== Scheduled wrapper failed ==="
    exit 1
}
