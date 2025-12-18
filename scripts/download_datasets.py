#!/usr/bin/env python3
"""Download datasets from Google Drive using gdown.

This script reads dataset links from the TOML configuration file
and downloads them to the specified output directory.
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    import tomli
except ImportError:
    import tomllib as tomli

import gdown


def load_dataset_links(config_path: Path) -> dict[str, str]:
    """Load dataset links from TOML configuration file."""
    with open(config_path, "rb") as f:
        config = tomli.load(f)
    
    # Filter out comments (keys starting with #)
    return {k: v for k, v in config.items() if not k.startswith("#")}


def _flatten_extracted_dir(extract_dir: Path) -> None:
    """Flatten nested directory structure after extraction.
    
    If the extracted content has a single subdirectory (e.g., helmholtz2d/),
    move its contents up one level to get: datasets/{name}/ground_truth/
    instead of: datasets/{name}/helmholtz2d/ground_truth/
    """
    # Remove __MACOSX directory if present
    macosx_dir = extract_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)
    
    # Get all items in the extract directory
    items = list(extract_dir.iterdir())
    
    # If there's exactly one directory, flatten it
    if len(items) == 1 and items[0].is_dir():
        nested_dir = items[0]
        print(f"Flattening: {nested_dir.name}/ -> {extract_dir.name}/")
        
        # Move all contents from nested directory up one level
        for item in nested_dir.iterdir():
            dest = extract_dir / item.name
            shutil.move(str(item), str(dest))
        
        # Remove the now-empty nested directory
        nested_dir.rmdir()


def download_dataset(
    name: str, 
    file_id: str, 
    output_dir: Path, 
    extract: bool = True
) -> bool:
    """Download a single dataset from Google Drive.
    
    Args:
        name: Name of the dataset
        file_id: Google Drive file ID
        output_dir: Directory to save the downloaded file
        extract: Whether to extract zip files after download
        
    Returns:
        True if download was successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = output_dir / f"{name}.zip"
    
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"File ID: {file_id}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    
    try:
        gdown.download(url, str(output_path), quiet=False)
        
        if extract and output_path.exists():
            import zipfile
            extract_dir = output_dir / name
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Extracting to: {extract_dir}")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Flatten the directory structure
            _flatten_extracted_dir(extract_dir)
            
            # Remove the zip file after extraction
            output_path.unlink()
            print(f"Removed: {output_path}")
            
        print(f"✓ Successfully downloaded: {name}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {name}: {e}")
        return False


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
        default=Path(__file__).parent.parent / "configs" / "datasets" / "dataset_links.toml",
        help="Path to dataset links TOML configuration file"
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
    if not args.config.exists():
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

