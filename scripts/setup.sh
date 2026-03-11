#!/usr/bin/env bash

set -e

echo "===================================="
echo " LocalTranscriber Development Setup "
echo "===================================="
echo

# ----------------------------------------------------
# Check uv
# ----------------------------------------------------

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is required but was not found."
    echo
    echo "Install uv with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "uv detected."
echo

# ----------------------------------------------------
# Create Python 3.12 virtual environment
# ----------------------------------------------------

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.12..."
    uv venv --python 3.12 --seed .venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

# ----------------------------------------------------
# Verify Python version
# ----------------------------------------------------

PY_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if [ "$PY_VERSION" != "3.12" ]; then
    echo "ERROR: LocalTranscriber requires Python 3.12."
    echo "Detected Python version inside .venv: $PY_VERSION"
    exit 1
fi

echo "Python 3.12 detected in virtual environment."
echo

# ----------------------------------------------------
# Install dependencies
# ----------------------------------------------------

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

# ----------------------------------------------------
# Run bootstrap
# ----------------------------------------------------

echo
echo "Downloading required assets..."
python scripts/bootstrap.py

echo
echo "===================================="
echo " Setup Complete "
echo "===================================="
echo
echo "Activate the virtual environment later with:"
echo "  source .venv/bin/activate"
echo
echo "Run the application with:"
echo "  python main.py"
echo