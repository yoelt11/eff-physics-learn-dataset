"""Dataset download utilities."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import IO, Dict

try:
    import tomli
except ImportError:
    import tomllib as tomli

import gdown


def _open_default_dataset_links() -> IO[bytes]:
    """Open the packaged default dataset links TOML.

    When installed via pip, hatch puts it at eff_physics_learn_dataset/configs/...
    When running from a repo checkout, it lives at repo_root/configs/...
    """

    # Installed wheel: config is at site-packages/eff_physics_learn_dataset/configs/...
    try:
        import eff_physics_learn_dataset as _pkg

        _base = Path(_pkg.__file__).resolve().parent
        _candidate = _base / "configs" / "datasets" / "dataset_links.toml"
        if _candidate.exists():
            return _candidate.open("rb")
    except Exception:
        pass

    # Dev fallback: repo root (this file is src/download.py -> parent.parent = repo root)
    here = Path(__file__).resolve()
    return (here.parent.parent / "configs" / "datasets" / "dataset_links.toml").open("rb")


def load_dataset_links(config_path: Path | str | None = None) -> Dict[str, str]:
    """Load dataset links from a TOML configuration file.

    Resolution order:
    - Explicit `config_path`
    - `EFF_PHYSICS_LEARN_DATASET_LINKS` env var
    - Packaged default config shipped with the library

    Returns:
        Dictionary mapping dataset names to Google Drive file IDs.
    """

    env_path = os.environ.get("EFF_PHYSICS_LEARN_DATASET_LINKS")
    if config_path is None and env_path:
        config_path = env_path

    if config_path is None:
        f = _open_default_dataset_links()
        with f:
            config = tomli.load(f)
    else:
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
        # gdown caches cookie/home paths at import time (see gdown/download.py).
        # If ~/.cache/gdown isn't writable, redirect gdown to write into our
        # output directory instead.
        try:
            import importlib

            gdown_cached_download = importlib.import_module("gdown.cached_download")
            gdown_download = importlib.import_module("gdown.download")

            # Redirect all gdown cache/cookies writes away from ~/
            # (sandbox environments often block writes to it).
            new_home = str(output_dir.resolve())
            gdown_download.home = new_home
            gdown_cached_download.cache_root = os.path.join(new_home, ".cache", "gdown")

            # Ensure the destination cookie/cache dirs exist.
            (output_dir.resolve() / ".cache" / "gdown").mkdir(
                parents=True, exist_ok=True
            )
        except Exception:
            # If we can't redirect gdown's internal paths, proceed; gdown
            # will raise a helpful error.
            pass

        try:
            # Retry without cookies first.
            gdown.download(url, str(output_path), quiet=False, use_cookies=False)
        except Exception:
            # Retry with cookies enabled for cases where gdown needs them.
            gdown.download(url, str(output_path), quiet=False, use_cookies=True)

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

