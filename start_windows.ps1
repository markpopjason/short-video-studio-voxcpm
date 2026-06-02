$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($Arguments -join ' ')"
    }
}

$RuntimeDir = Join-Path $PSScriptRoot ".runtime"
$UvDir = Join-Path $RuntimeDir "uv"
$UvExe = Join-Path $UvDir "uv.exe"
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $UvDir | Out-Null

if (-not (Test-Path $UvExe)) {
    Write-Step "Downloading uv portable runtime"
    $ZipPath = Join-Path $RuntimeDir "uv.zip"
    $UvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    Invoke-WebRequest -Uri $UvUrl -OutFile $ZipPath
    Expand-Archive -Path $ZipPath -DestinationPath $RuntimeDir -Force
    $DownloadedUv = Get-ChildItem -Path $RuntimeDir -Recurse -Filter "uv.exe" | Select-Object -First 1
    if ($null -eq $DownloadedUv) {
        throw "uv.exe was not found after extracting $ZipPath"
    }
    Copy-Item -Path $DownloadedUv.FullName -Destination $UvExe -Force
}

if (-not (Test-Path $VenvPython)) {
    Write-Step "Installing managed Python 3.11 and creating local environment"
    Invoke-Checked $UvExe @("python", "install", "3.11")
    Invoke-Checked $UvExe @("venv", ".venv", "--python", "3.11")
}

Invoke-Checked $VenvPython @("-c", "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)")

Write-Step "Installing PyTorch into the local environment"
$TorchBackend = $env:VOXCPM_TORCH_BACKEND
if ([string]::IsNullOrWhiteSpace($TorchBackend)) {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $TorchBackend = "cu121"
    } else {
        $TorchBackend = "cpu"
    }
}

switch ($TorchBackend.ToLowerInvariant()) {
    "cpu" {
        Invoke-Checked $UvExe @("pip", "install", "--python", $VenvPython, "torch>=2.5.0", "--index-url", "https://download.pytorch.org/whl/cpu")
    }
    "cu121" {
        Invoke-Checked $UvExe @("pip", "install", "--python", $VenvPython, "torch>=2.5.0", "--index-url", "https://download.pytorch.org/whl/cu121")
    }
    "cu124" {
        Invoke-Checked $UvExe @("pip", "install", "--python", $VenvPython, "torch>=2.5.0", "--index-url", "https://download.pytorch.org/whl/cu124")
    }
    default {
        throw "Unsupported VOXCPM_TORCH_BACKEND=$TorchBackend. Use cpu, cu121, or cu124."
    }
}

Write-Step "Installing VoxCPM Studio dependencies"
Invoke-Checked $UvExe @("pip", "install", "--python", $VenvPython, "-r", "requirements.txt")

Write-Step "Starting VoxCPM Studio"
Invoke-Checked $VenvPython @("-m", "voxcpm_studio", "serve", "--host", "127.0.0.1", "--port", "8808", "--device", "auto")
