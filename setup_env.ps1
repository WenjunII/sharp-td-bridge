# Robust Environment Setup for SHARP + Bridge
$ErrorActionPreference = "Stop"

Write-Host "--- Starting Setup ---"

# 1. Conda Env
Write-Host "[1/4] Ensuring conda env 'sharp' exists..."
if (-not (conda env list | Select-String "sharp")) {
    conda create -n sharp python=3.13 -y
}

# 2. Clone SHARP
$sharpDir = Join-Path (Split-Path $PSScriptRoot) "ml-sharp"
if (-not (Test-Path $sharpDir)) {
    Write-Host "[2/4] Cloning ml-sharp..."
    git clone https://github.com/apple/ml-sharp.git $sharpDir
}

# 3. Install SHARP deps
Write-Host "[3/4] Installing SHARP deps (this may take a few minutes)..."
Push-Location $sharpDir
try {
    # We must run pip from within the ml-sharp directory because of '-e .' in requirements.txt
    conda run -n sharp pip install -r requirements.txt
} finally {
    Pop-Location
}

# 4. Install Bridge deps
Write-Host "[4/4] Installing bridge deps..."
conda run -n sharp pip install -r "$PSScriptRoot\requirements.txt"

# Create directories
New-Item -ItemType Directory -Force -Path "$PSScriptRoot\frames" | Out-Null
New-Item -ItemType Directory -Force -Path "$PSScriptRoot\splats" | Out-Null

Write-Host "--- Setup Complete! ---"
