# Dataset Download Guide

## Quick Start

```bash
# List available datasets
uv run python scripts/download_datasets.py --list

# Download a specific dataset
uv run python scripts/download_datasets.py -d helmholtz2D
uv run python scripts/download_datasets.py -d helmholtz3D
uv run python scripts/download_datasets.py -d wave2d1

# Download all datasets
uv run python scripts/download_datasets.py
```

## Command Options

| Option | Short | Description |
|--------|-------|-------------|
| `--list` | `-l` | List available datasets |
| `--dataset NAME` | `-d` | Download specific dataset |
| `--output-dir PATH` | `-o` | Output directory (default: `./datasets`) |
| `--config PATH` | `-c` | Custom config file path |
| `--no-extract` | | Keep zip files without extracting |

## Examples

```bash
# Download to custom directory
uv run python scripts/download_datasets.py -d burgers -o /data/physics

# Using the shell wrapper
./scripts/download_datasets.sh -d helmholtz2D
./scripts/download_datasets.sh -d helmholtz3D
./scripts/download_datasets.sh -d wave2d1
./scripts/download_datasets.sh --list
```

## Output Structure

Downloaded datasets are extracted to:

```
datasets/{dataset_name}/ground_truth/
```

## Available Datasets

| Dataset | Description |
|---------|-------------|
| `helmholtz2D` | 2D Helmholtz equation |
| `helmholtz3D` | 3D Helmholtz equation |
| `wave2d1` | 2D wave equation (time-dependent) |
| `burgers` | Burgers equation |
| `allen_cahn` | Allen-Cahn equation |
| `convection` | Convection equation |
| `ks` | Kuramoto–Sivashinsky (1D + time) |
| `flow_mixing` | Flow mixing simulation |
| `hlrp_cdr` | HLRP CDR |
| `hlrp_convection` | HLRP Convection |
| `hlrp_diffusion` | HLRP Diffusion |
| `hlrp_helmholtz` | HLRP Helmholtz |
| `hrlp_reaction` | HLRP Reaction |

