# Efficient Physics Learning Datasets

A JAX-based dataset utilities package for PDE (Partial Differential Equations) learning research.

## Overview

This repository provides tools and utilities for downloading, processing, and working with physics-informed machine learning datasets. It is designed to work seamlessly with JAX and Hugging Face's datasets library.

## Available Datasets

| Dataset | Description |
|---------|-------------|
| `helmholtz2D` | 2D Helmholtz equation data |
| `helmholtz3D` | 3D Helmholtz equation data |
| `burgers` | Burgers equation data |
| `allen_cahn` | Allen-Cahn equation data |
| `flow_mixing` | Flow mixing simulation data |
| `convection` | Convection equation data |
| `hlrp_cdr` | HLRP Convection-Diffusion-Reaction data |
| `hlrp_convection` | HLRP Convection data |
| `hlrp_diffusion` | HLRP Diffusion data |
| `hlrp_helmholtz` | HLRP Helmholtz data |
| `hlrp_reaction` | HLRP Reaction data |

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Clone the repository
git clone <repository-url>
cd eff-physics-learn-dataset

# Create environment and install dependencies
uv sync
```

## Usage

### Downloading Datasets

List available datasets:
```bash
uv run python scripts/download_datasets.py --list
```

Download all datasets:
```bash
uv run python scripts/download_datasets.py
```

Download a specific dataset:
```bash
uv run python scripts/download_datasets.py -d helmholtz2D
```

Download to a custom directory:
```bash
uv run python scripts/download_datasets.py -o /path/to/datasets -d burgers
```

### Using the Shell Script

```bash
# Make the script executable
chmod +x scripts/download_datasets.sh

# Run it
./scripts/download_datasets.sh --list
./scripts/download_datasets.sh helmholtz2D
```

## Dependencies

- **JAX/JAXLib**: High-performance numerical computing
- **Hugging Face Datasets**: Dataset loading and processing
- **gdown**: Google Drive file downloads
- **NumPy/SciPy**: Numerical operations
- **h5py**: HDF5 file support

## Project Structure

```
eff-physics-learn-dataset/
├── configs/
│   └── datasets/
│       └── dataset_links.toml    # Dataset Google Drive IDs
├── scripts/
│   ├── download_datasets.py      # Python download script
│   └── download_datasets.sh      # Shell wrapper
├── src/                          # Source code (TBD)
├── pyproject.toml               # Project configuration
└── README.md
```

## License

MIT License - See [LICENSE](LICENSE) for details.

