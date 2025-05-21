#!/usr/bin/env bash
# Simple setup script for Strategic Counsel Gen 3
set -euo pipefail

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional test verification
pytest --version >/dev/null 2>&1 && echo "pytest installed"
