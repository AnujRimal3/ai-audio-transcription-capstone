Write-Host ""
Write-Host "===================================="
Write-Host " LocalTranscriber Development Setup "
Write-Host "===================================="
Write-Host ""

# ----------------------------------------------------
# Check uv
# ----------------------------------------------------

$uv = Get-Command uv -ErrorAction SilentlyContinue

if (-not $uv) {
    Write-Host "ERROR: uv is required but was not found."
    Write-Host ""
    Write-Host "Install uv with:"
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
}

Write-Host "uv detected."
Write-Host ""

# ----------------------------------------------------
# Create Python 3.12 virtual environment
# ----------------------------------------------------

if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment with Python 3.12..."
    uv venv --python 3.12 --seed .venv
}
else {
    Write-Host "Virtual environment already exists."
}

Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

# ----------------------------------------------------
# Verify Python version
# ----------------------------------------------------

$version = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

if ($version -ne "3.12") {
    Write-Host "ERROR: LocalTranscriber requires Python 3.12."
    Write-Host "Detected Python version inside .venv: $version"
    exit 1
}

Write-Host "Python 3.12 detected in virtual environment."
Write-Host ""

# ----------------------------------------------------
# Install dependencies
# ----------------------------------------------------

Write-Host "Installing Python dependencies..."
uv pip install -r requirements.txt

# ----------------------------------------------------
# Run bootstrap
# ----------------------------------------------------

Write-Host ""
Write-Host "Downloading required assets..."
python scripts/bootstrap.py

Write-Host ""
Write-Host "===================================="
Write-Host " Setup Complete "
Write-Host "===================================="
Write-Host ""
Write-Host "Activate the virtual environment later with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run the application with:"
Write-Host "  python main.py"
Write-Host ""