"""Dataset download utilities."""

import shutil
import tempfile
from pathlib import Path
from typing import Dict

try:
    import tomli
except ImportError:
    import tomllib as tomli

import gdown


def load_dataset_links(config_path: Path | str) -> Dict[str, str]:
    """Load dataset links from TOML configuration file.
    
    Args:
        config_path: Path to the TOML configuration file
        
    Returns:
        Dictionary mapping dataset names to Google Drive file IDs
    """
    config_path = Path(config_path)
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
    output_dir: Path | str, 
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
    output_dir = Path(output_dir)
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

