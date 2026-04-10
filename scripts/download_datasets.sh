#!/bin/bash
# Download datasets wrapper script
# Usage: ./download_datasets.sh [dataset_name]
#
# Examples:
#   ./download_datasets.sh                 # Download all datasets
#   ./download_datasets.sh helmholtz2D     # 2D Helmholtz
#   ./download_datasets.sh helmholtz3D      # 3D Helmholtz
#   ./download_datasets.sh wave2d1          # 2D wave (time-dependent)
#   ./download_datasets.sh --list          # List available datasets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Run the Python download script with uv
uv run python scripts/download_datasets.py "$@"

