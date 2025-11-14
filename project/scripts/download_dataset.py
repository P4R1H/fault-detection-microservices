#!/usr/bin/env python3
"""
Download RCAEval RE2-TrainTicket Dataset from Zenodo

Dataset DOI: 10.5281/zenodo.14590730
Source: WWW'25 and ASE 2024 benchmark

This script downloads the 270-case multimodal failure dataset including:
- Metrics (77-376 per case, 5-min granularity)
- Logs (8.6-26.9M lines with structure)
- Traces (39.6-76.7M distributed traces)
- Ground truth labels (root cause service + indicator)

Total size: ~4.21GB compressed
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib


# Zenodo record configuration
ZENODO_RECORD_ID = "14590730"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Dataset configuration
DATASET_DIR = Path(__file__).parent.parent / "data" / "RCAEval"
DATASETS = {
    "TrainTicket": {
        "filename": "train-ticket_dataset.zip",
        "size_mb": 1500,  # Approximate
        "description": "41-service microservice system"
    },
    "SockShop": {
        "filename": "sock-shop_dataset.zip",
        "size_mb": 1400,
        "description": "13-service e-commerce system"
    },
    "OnlineBoutique": {
        "filename": "online-boutique_dataset.zip",
        "size_mb": 1300,
        "description": "12-service demo application"
    }
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dest_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            size = f.write(chunk)
            pbar.update(size)

    return dest_path


def verify_checksum(file_path: Path, expected_md5: str = None):
    """Verify file integrity using MD5 checksum"""
    if not expected_md5:
        return True

    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)

    computed = md5_hash.hexdigest()
    if computed != expected_md5:
        print(f"❌ Checksum mismatch!")
        print(f"   Expected: {expected_md5}")
        print(f"   Got:      {computed}")
        return False

    print(f"✅ Checksum verified: {computed}")
    return True


def get_zenodo_files():
    """Fetch file information from Zenodo API"""
    print(f"📡 Fetching dataset information from Zenodo...")
    response = requests.get(ZENODO_API_URL)
    response.raise_for_status()

    record = response.json()
    files = record.get('files', [])

    print(f"✅ Found {len(files)} files in Zenodo record")
    return files


def download_rcaeval_dataset(systems=['TrainTicket'], extract=True):
    """
    Download RCAEval dataset for specified systems

    Args:
        systems: List of system names to download ('TrainTicket', 'SockShop', 'OnlineBoutique')
        extract: Whether to extract zip files after download
    """
    print("=" * 70)
    print("RCAEval RE2 Dataset Downloader")
    print("=" * 70)
    print(f"DOI: 10.5281/zenodo.{ZENODO_RECORD_ID}")
    print(f"Destination: {DATASET_DIR}")
    print(f"Systems to download: {', '.join(systems)}")
    print("=" * 70)

    # Create data directory
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Get file list from Zenodo
        zenodo_files = get_zenodo_files()

        # Download each requested system
        for system in systems:
            if system not in DATASETS:
                print(f"⚠️  Unknown system: {system}")
                continue

            config = DATASETS[system]
            filename = config['filename']

            # Find matching file in Zenodo record
            matching_file = next(
                (f for f in zenodo_files if filename in f.get('key', '')),
                None
            )

            if not matching_file:
                print(f"⚠️  File not found in Zenodo: {filename}")
                print(f"   Available files: {[f['key'] for f in zenodo_files]}")
                continue

            download_url = matching_file['links']['self']
            checksum = matching_file.get('checksum', '').replace('md5:', '')
            file_size_mb = matching_file['size'] / (1024 * 1024)

            dest_file = DATASET_DIR / filename

            # Check if already downloaded
            if dest_file.exists():
                print(f"\n📦 {system} ({config['description']})")
                print(f"   File already exists: {dest_file}")
                print(f"   Size: {dest_file.stat().st_size / (1024**2):.1f} MB")

                if verify_checksum(dest_file, checksum):
                    print(f"   ✅ Verified - skipping download")
                    continue
                else:
                    print(f"   ⚠️  Checksum failed - re-downloading")

            # Download file
            print(f"\n📥 Downloading {system} ({config['description']})")
            print(f"   Size: {file_size_mb:.1f} MB")
            print(f"   URL: {download_url[:60]}...")

            download_file(download_url, dest_file, desc=f"{system}")

            # Verify download
            if checksum:
                verify_checksum(dest_file, checksum)

            # Extract if requested
            if extract:
                print(f"   📂 Extracting {filename}...")
                import zipfile
                extract_dir = DATASET_DIR / system
                extract_dir.mkdir(exist_ok=True)

                with zipfile.ZipFile(dest_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                print(f"   ✅ Extracted to {extract_dir}")

        print("\n" + "=" * 70)
        print("✅ Download complete!")
        print(f"📁 Dataset location: {DATASET_DIR}")
        print("=" * 70)

        # Show dataset summary
        print("\nDataset Summary:")
        for system in systems:
            system_dir = DATASET_DIR / system
            if system_dir.exists():
                files = list(system_dir.rglob('*'))
                size = sum(f.stat().st_size for f in files if f.is_file())
                print(f"  {system}: {len(files)} files, {size / (1024**2):.1f} MB")

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error downloading from Zenodo: {e}")
        print(f"   Please check your internet connection and try again")
        print(f"   Or download manually from: https://zenodo.org/record/{ZENODO_RECORD_ID}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download RCAEval RE2 Dataset from Zenodo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download only TrainTicket (default)
  python download_dataset.py

  # Download all three systems
  python download_dataset.py --all

  # Download specific systems
  python download_dataset.py --systems TrainTicket SockShop

  # Download without extracting
  python download_dataset.py --no-extract
        """
    )

    parser.add_argument(
        '--systems',
        nargs='+',
        choices=['TrainTicket', 'SockShop', 'OnlineBoutique'],
        default=['TrainTicket'],
        help='Systems to download (default: TrainTicket)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all three systems'
    )

    parser.add_argument(
        '--no-extract',
        action='store_true',
        help='Do not extract zip files'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DATASET_DIR,
        help=f'Download directory (default: {DATASET_DIR})'
    )

    args = parser.parse_args()

    # Determine which systems to download
    if args.all:
        systems = list(DATASETS.keys())
    else:
        systems = args.systems

    # Update data directory if specified
    global DATASET_DIR
    DATASET_DIR = args.data_dir

    # Download dataset
    success = download_rcaeval_dataset(
        systems=systems,
        extract=not args.no_extract
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
