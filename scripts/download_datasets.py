#!/usr/bin/env python3
"""Download datasets from Google Drive using gdown.

This script reads dataset links from the TOML configuration file
and downloads them to the specified output directory.
"""

import argparse
import sys
from pathlib import Path

from eff_physics_learn_dataset.download import download_dataset, load_dataset_links


def main():
    parser = argparse.ArgumentParser(
        description="Download physics learning datasets from Google Drive"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Specific dataset to download (default: all datasets)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./datasets"),
        help="Output directory for downloaded datasets (default: ./datasets)"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to dataset links TOML configuration file (default: packaged config)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets without downloading"
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Do not extract zip files after download"
    )
    
    args = parser.parse_args()
    
    # Load dataset links
    if args.config is not None and not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    datasets = load_dataset_links(args.config)
    
    # List mode
    if args.list:
        print("\nAvailable datasets:")
        print("-" * 40)
        for name in sorted(datasets.keys()):
            print(f"  • {name}")
        print(f"\nTotal: {len(datasets)} datasets")
        return
    
    # Determine which datasets to download
    if args.dataset:
        if args.dataset not in datasets:
            print(f"Error: Dataset '{args.dataset}' not found.")
            print("Available datasets:", ", ".join(sorted(datasets.keys())))
            sys.exit(1)
        to_download = {args.dataset: datasets[args.dataset]}
    else:
        to_download = datasets
    
    # Download datasets
    print(f"\nDownloading {len(to_download)} dataset(s) to: {args.output_dir.absolute()}")
    
    successful = 0
    failed = 0
    
    for name, file_id in to_download.items():
        if download_dataset(name, file_id, args.output_dir, extract=not args.no_extract):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Download Summary")
    print(f"{'='*60}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {successful + failed}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

