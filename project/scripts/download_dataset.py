#!/usr/bin/env python3
"""
Download RCAEval Dataset from Zenodo

Dataset DOI: 10.5281/zenodo.14590730
Source: WWW'25 and ASE 2024 benchmark

This script downloads multimodal failure datasets including:
- Metrics (77-376 per case, 5-min granularity)
- Logs (8.6-26.9M lines with structure)
- Traces (39.6-76.7M distributed traces)
- Ground truth labels (root cause service + indicator)

Available datasets:
- RE1: Round 1 data collection (90 cases per system)
- RE2: Round 2 data collection (90 cases per system)
- RE3: Round 3 data collection (90 cases per system)
- Systems: TrainTicket (TT), SockShop (SS), OnlineBoutique (OB)
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib
import zipfile
import argparse


# Zenodo record configuration
ZENODO_RECORD_ID = "14590730"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Dataset configuration
# Mapping from friendly names to zip file prefixes
SYSTEM_MAPPING = {
    "TrainTicket": "TT",
    "SockShop": "SS",
    "OnlineBoutique": "OB"
}

RE_VERSIONS = ["RE1", "RE2", "RE3"]

SYSTEM_INFO = {
    "TrainTicket": {
        "code": "TT",
        "description": "41-service train ticket booking system",
        "services": 41
    },
    "SockShop": {
        "code": "SS",
        "description": "13-service e-commerce demo",
        "services": 13
    },
    "OnlineBoutique": {
        "code": "OB",
        "description": "12-service Google demo application",
        "services": 12
    }
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading from {url}: {e}")
        return None

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(dest_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    size = f.write(chunk)
                    pbar.update(size)
    except Exception as e:
        print(f"❌ Error writing file {dest_path}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return None

    return dest_path


def verify_checksum(file_path: Path, expected_md5: str = None):
    """Verify file integrity using MD5 checksum"""
    if not expected_md5:
        return True

    print(f"   Verifying checksum...")
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)

    computed = md5_hash.hexdigest()
    if computed != expected_md5:
        print(f"   ❌ Checksum mismatch!")
        print(f"      Expected: {expected_md5}")
        print(f"      Got:      {computed}")
        return False

    print(f"   ✅ Checksum verified: {computed[:16]}...")
    return True


def extract_zip(zip_path: Path, extract_base_dir: Path, system: str, re_version: str):
    """
    Extract zip file to proper directory structure

    Args:
        zip_path: Path to zip file
        extract_base_dir: Base directory for extraction (data/RCAEval)
        system: System name (TrainTicket, SockShop, OnlineBoutique)
        re_version: RE version (RE1, RE2, RE3)

    Extracts to: {extract_base_dir}/{System}/{RE-Version}/
    """
    extract_dir = extract_base_dir / system / re_version
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"   📂 Extracting to {extract_dir}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total uncompressed size
            total_size = sum(info.file_size for info in zip_ref.infolist())

            # Extract with progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="   Extracting") as pbar:
                for member in zip_ref.infolist():
                    zip_ref.extract(member, extract_dir)
                    pbar.update(member.file_size)

        print(f"   ✅ Extracted successfully")
        return extract_dir

    except zipfile.BadZipFile as e:
        print(f"   ❌ Bad zip file: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Extraction error: {e}")
        return None


def get_zenodo_files():
    """Fetch file information from Zenodo API"""
    print(f"📡 Fetching dataset information from Zenodo...")
    try:
        response = requests.get(ZENODO_API_URL, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Zenodo: {e}")
        return None

    try:
        record = response.json()
        files = record.get('files', [])
        print(f"✅ Found {len(files)} files in Zenodo record")
        return files
    except Exception as e:
        print(f"❌ Error parsing Zenodo response: {e}")
        return None


def download_rcaeval_dataset(
    systems: list,
    re_versions: list,
    data_dir: Path,
    extract: bool = True,
    skip_existing: bool = True
):
    """
    Download RCAEval dataset for specified systems and RE versions

    Args:
        systems: List of system names ('TrainTicket', 'SockShop', 'OnlineBoutique')
        re_versions: List of RE versions to download ('RE1', 'RE2', 'RE3')
        data_dir: Base directory for data storage
        extract: Whether to extract zip files after download
        skip_existing: Skip download if file already exists and checksum matches
    """
    print("=" * 80)
    print("RCAEval Dataset Downloader")
    print("=" * 80)
    print(f"DOI: 10.5281/zenodo.{ZENODO_RECORD_ID}")
    print(f"Destination: {data_dir}")
    print(f"Systems: {', '.join(systems)}")
    print(f"RE Versions: {', '.join(re_versions)}")
    print("=" * 80)

    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get file list from Zenodo
    zenodo_files = get_zenodo_files()
    if zenodo_files is None:
        return False

    # Create download directory for zip files
    download_dir = data_dir / "downloads"
    download_dir.mkdir(exist_ok=True)

    downloaded_count = 0
    extracted_count = 0
    skipped_count = 0

    # Download each requested combination
    for system in systems:
        if system not in SYSTEM_INFO:
            print(f"⚠️  Unknown system: {system}")
            continue

        system_code = SYSTEM_INFO[system]['code']
        system_desc = SYSTEM_INFO[system]['description']

        for re_version in re_versions:
            # Construct expected filename: RE1-TT.zip, RE2-SS.zip, etc.
            zip_filename = f"{re_version}-{system_code}.zip"

            print(f"\n{'='*80}")
            print(f"📦 {system} - {re_version}")
            print(f"   {system_desc}")
            print(f"   File: {zip_filename}")
            print(f"{'='*80}")

            # Find matching file in Zenodo record
            matching_file = None
            for f in zenodo_files:
                file_key = f.get('key', '')
                if zip_filename == file_key or zip_filename in file_key:
                    matching_file = f
                    break

            if not matching_file:
                print(f"⚠️  File not found in Zenodo: {zip_filename}")
                print(f"   Available files: {[f['key'] for f in zenodo_files]}")
                continue

            download_url = matching_file['links']['self']
            checksum = matching_file.get('checksum', '').replace('md5:', '')
            file_size_mb = matching_file['size'] / (1024 * 1024)

            dest_file = download_dir / zip_filename

            # Check if already downloaded and verified
            if skip_existing and dest_file.exists():
                print(f"   📦 File exists: {dest_file.name} ({file_size_mb:.1f} MB)")
                if verify_checksum(dest_file, checksum):
                    print(f"   ✅ Verified - skipping download")
                    skipped_count += 1

                    # Extract if needed
                    if extract:
                        extract_target = data_dir / system / re_version
                        if extract_target.exists() and any(extract_target.iterdir()):
                            print(f"   ✅ Already extracted to {extract_target}")
                        else:
                            extract_dir = extract_zip(dest_file, data_dir, system, re_version)
                            if extract_dir:
                                extracted_count += 1
                    continue
                else:
                    print(f"   ⚠️  Checksum failed - re-downloading")
                    dest_file.unlink()

            # Download file
            print(f"   📥 Downloading {zip_filename} ({file_size_mb:.1f} MB)")
            print(f"   URL: {download_url[:70]}...")

            downloaded_file = download_file(download_url, dest_file, desc=f"   {re_version}-{system_code}")

            if downloaded_file is None:
                print(f"   ❌ Download failed")
                continue

            downloaded_count += 1

            # Verify download
            if checksum:
                if not verify_checksum(downloaded_file, checksum):
                    print(f"   ❌ Checksum verification failed - file may be corrupted")
                    continue

            # Extract if requested
            if extract:
                extract_dir = extract_zip(downloaded_file, data_dir, system, re_version)
                if extract_dir:
                    extracted_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("✅ Download Complete!")
    print("=" * 80)
    print(f"📊 Summary:")
    print(f"   Downloaded: {downloaded_count} files")
    print(f"   Skipped (already verified): {skipped_count} files")
    print(f"   Extracted: {extracted_count} archives")
    print(f"   Total files processed: {downloaded_count + skipped_count}")
    print(f"\n📁 Dataset location: {data_dir}")
    print("=" * 80)

    # Show directory structure
    print(f"\n📂 Directory Structure:")
    for system in systems:
        system_dir = data_dir / system
        if system_dir.exists():
            print(f"\n  {system}/")
            for re_version in re_versions:
                re_dir = system_dir / re_version
                if re_dir.exists():
                    files = list(re_dir.rglob('*'))
                    file_count = sum(1 for f in files if f.is_file())
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    print(f"    ├── {re_version}/ ({file_count} files, {total_size / (1024**2):.1f} MB)")

    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Download RCAEval Dataset from Zenodo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all systems, all RE versions (recommended)
  python download_dataset.py --all

  # Download only RE2 versions (270 cases - main benchmark)
  python download_dataset.py --all --reversions RE2

  # Download specific system and version
  python download_dataset.py --systems TrainTicket --reversions RE2

  # Download TrainTicket, all versions
  python download_dataset.py --systems TrainTicket

  # Download without extracting
  python download_dataset.py --all --no-extract

File naming convention:
  - RE1-TT.zip = Round 1, TrainTicket
  - RE2-SS.zip = Round 2, SockShop
  - RE3-OB.zip = Round 3, OnlineBoutique

Extracted structure:
  data/RCAEval/
    ├── TrainTicket/
    │   ├── RE1/
    │   ├── RE2/
    │   └── RE3/
    ├── SockShop/
    │   ├── RE1/
    │   ├── RE2/
    │   └── RE3/
    └── OnlineBoutique/
        ├── RE1/
        ├── RE2/
        └── RE3/
        """
    )

    parser.add_argument(
        '--systems',
        nargs='+',
        choices=['TrainTicket', 'SockShop', 'OnlineBoutique'],
        help='Systems to download (default: all if --all specified)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all three systems'
    )

    parser.add_argument(
        '--reversions',
        nargs='+',
        choices=['RE1', 'RE2', 'RE3'],
        default=['RE1', 'RE2', 'RE3'],
        help='RE versions to download (default: all three)'
    )

    parser.add_argument(
        '--no-extract',
        action='store_true',
        help='Do not extract zip files'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/RCAEval'),
        help='Download directory (default: data/RCAEval)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download even if files exist'
    )

    args = parser.parse_args()

    # Determine which systems to download
    if args.all:
        systems = list(SYSTEM_INFO.keys())
    elif args.systems:
        systems = args.systems
    else:
        # Default to all systems if neither specified
        print("No systems specified. Use --systems or --all")
        print("Example: python download_dataset.py --all")
        return 1

    # Download dataset
    success = download_rcaeval_dataset(
        systems=systems,
        re_versions=args.reversions,
        data_dir=args.data_dir,
        extract=not args.no_extract,
        skip_existing=not args.force
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
