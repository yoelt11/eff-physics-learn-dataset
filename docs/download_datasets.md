# Dataset Download Guide

## Quick Start

```bash
# List available datasets
uv run python scripts/download_datasets.py --list

# Download a specific dataset
uv run python scripts/download_datasets.py -d helmholtz2D

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
| `burgers` | Burgers equation |
| `allen_cahn` | Allen-Cahn equation |
| `flow_mixing` | Flow mixing simulation |
| `convection` | Convection equation |
| `hlrp_cdr` | HLRP CDR |
| `hlrp_convection` | HLRP Convection |
| `hlrp_diffusion` | HLRP Diffusion |
| `hlrp_helmholtz` | HLRP Helmholtz |
| `hrlp_reaction` | HLRP Reaction |

