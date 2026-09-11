# Anzar — one-line installer for Windows (PowerShell 5.1+).
#
# Usage:
#   irm https://raw.githubusercontent.com/ghaaith/anzar-agent/main/scripts/install.ps1 | iex
#
# Ensures Python 3 (installs via winget if missing), bootstraps pipx, and
# installs the `anzar` CLI into ~/.local/bin, adding it to your user PATH.
# Safe to re-run: it upgrades an existing install.

$ErrorActionPreference = 'Stop'
$Package = 'anzar-agent'

function Write-Step([string]$Msg) { Write-Host "==> $Msg" -ForegroundColor Cyan }
function Write-Warn([string]$Msg) { Write-Host "!!  $Msg" -ForegroundColor Yellow }

# --- 1. Find Python 3 -------------------------------------------------------
$Python = $null
if (Get-Command python -ErrorAction SilentlyContinue) { $Python = (Get-Command python).Source }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $Python = (Get-Command py).Source }

$ver = if ($Python) { & $Python --version 2>&1 } else { '' }
if (-not $Python -or $LASTEXITCODE -ne 0 -or "$ver" -notmatch 'Python 3') {
    Write-Warn 'Python 3 was not found. Installing it via winget...'
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements | Out-Host
    $Python = (Get-Command python -ErrorAction SilentlyContinue).Source
    $ver = if ($Python) { & $Python --version 2>&1 } else { '' }
}
if (-not $Python -or "$ver" -notmatch 'Python 3') {
    throw 'Could not find Python 3. Install it from https://www.python.org/downloads/ and re-run.'
}

# --- 2. Bootstrap pipx ------------------------------------------------------
$pvOk = $false
$saveEA = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $Python -m pipx --version 2>$null
$pvOk = $LASTEXITCODE -eq 0
if (-not $pvOk) {
    Write-Step 'Installing pipx...'
    & $Python -m pip install --quiet --user pipx 2>$null
    & $Python -m pipx --version 2>$null
    $pvOk = $LASTEXITCODE -eq 0
    if (-not $pvOk) { $ErrorActionPreference = $saveEA; throw 'Could not install pipx.' }
}
$ErrorActionPreference = $saveEA

# --- 3. Install/upgrade anzar -----------------------------------------------
$ErrorActionPreference = 'Continue'
$listed = & $Python -m pipx list --short 2>$null
$listedOk = $LASTEXITCODE -eq 0
if ($listedOk -and $listed -match "^$Package\b") {
    Write-Step "Upgrading $Package..."
    & $Python -m pipx upgrade $Package 2>$null
    if ($LASTEXITCODE -ne 0) { $ErrorActionPreference = $saveEA; throw "Failed to upgrade $Package." }
}
else {
    Write-Step "Installing $Package..."
    & $Python -m pipx install $Package 2>$null
    if ($LASTEXITCODE -ne 0) { $ErrorActionPreference = $saveEA; throw "Failed to install $Package." }
}
$ErrorActionPreference = $saveEA

# --- 4. Make sure `anzar` is on the user PATH -------------------------------
if (Get-Command anzar -ErrorAction SilentlyContinue) {
    $env:Path = "$env:Path;$(Join-Path $env:USERPROFILE '.local\bin')"
}
else {
    $binDir = Join-Path $env:USERPROFILE '.local\bin'
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userPath -notlike "*$binDir*") {
        [Environment]::SetEnvironmentVariable('Path', "$userPath;$binDir", 'User')
        Write-Step "Added $binDir to your user PATH."
    }
    $env:Path = "$env:Path;$binDir"
}

Write-Host ''
Write-Host 'Done! Anzar is installed.' -ForegroundColor Green
if (Get-Command anzar -ErrorAction SilentlyContinue) {
    Write-Host '  Run:    anzar' -ForegroundColor Green
    Write-Host '  One-shot: anzar "explain this codebase"' -ForegroundColor Green
}
else {
    Write-Host '  Open a NEW terminal, then run: anzar' -ForegroundColor Green
    Write-Host '  One-shot: anzar "explain this codebase"' -ForegroundColor Green
}