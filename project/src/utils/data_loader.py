"""
RCAEval Data Loader - Robust Version

Handles the actual RCAEval dataset structure:
- Double nesting: RE2/RE2-TT/, RE1/RE1-OB/, etc.
- Inconsistent filenames: metrics.csv (RE2/RE3) vs data.csv (RE1)
- Labels encoded in folder names: {service}_{fault}/1/

Example paths:
  data/RCAEval/TrainTicket/RE2/RE2-TT/ts-auth-service_cpu/1/metrics.csv
  data/RCAEval/OnlineBoutique/RE1/RE1-OB/adservice_cpu/1/data.csv
  data/RCAEval/SockShop/RE3/RE3-SS/carts_f1/1/metrics.csv
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import Counter


@dataclass
class FailureCase:
    """Single failure case from RCAEval dataset"""
    case_id: str
    system: str  # 'TrainTicket', 'SockShop', 'OnlineBoutique'
    fault_type: str  # 'CPU', 'MEM', 'DISK', 'SOCKET', 'DELAY', 'LOSS'
    root_cause_service: str
    root_cause_indicator: str

    # Data modalities
    metrics: Optional[pd.DataFrame] = None
    logs: Optional[pd.DataFrame] = None
    traces: Optional[pd.DataFrame] = None

    # Metadata
    re_version: Optional[str] = None  # RE1, RE2, or RE3
    case_number: Optional[int] = None
    timestamp: Optional[pd.Timestamp] = None
    duration_minutes: Optional[int] = None


class RCAEvalDataLoader:
    """
    Robust data loader for RCAEval benchmark dataset

    Handles actual dataset structure with:
    - Recursive discovery of failure cases
    - Dynamic label parsing from folder names
    - Multiple metric filename formats
    - Fault code mapping

    Usage:
        loader = RCAEvalDataLoader(data_dir='data/RCAEval')
        cases = loader.load_all_cases()
        train, val, test = loader.load_splits(train_ratio=0.6, val_ratio=0.2)
    """

    # Fault code mappings (observed in SockShop dataset)
    FAULT_CODE_MAP = {
        'cpu': 'CPU',
        'mem': 'MEM',
        'memory': 'MEM',
        'disk': 'DISK',
        'delay': 'DELAY',
        'loss': 'LOSS',
        'packet_loss': 'LOSS',
        'socket': 'SOCKET',
        # SockShop specific codes
        'f1': 'CPU',
        'f2': 'MEM',
        'f3': 'DISK',
        'f4': 'DELAY',
        'f5': 'LOSS',
        'f6': 'SOCKET',
    }

    # Metric filename priority (try in order)
    METRIC_FILENAMES = ['metrics.csv', 'data.csv', 'simple_metrics.csv']

    def __init__(self, data_dir: str = 'data/RCAEval'):
        """
        Initialize data loader

        Args:
            data_dir: Path to RCAEval dataset directory
        """
        self.data_dir = Path(data_dir)
        self.systems = ['TrainTicket', 'SockShop', 'OnlineBoutique']

        # Verify dataset exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"RCAEval dataset not found at {self.data_dir}\n"
                f"Please run: python scripts/download_dataset.py --all"
            )

    def load_all_cases(self, systems: List[str] = None, re_versions: List[str] = None) -> List[FailureCase]:
        """
        Load all failure cases from specified systems

        Args:
            systems: List of system names to load (default: all)
            re_versions: List of RE versions to load (default: all)

        Returns:
            List of FailureCase objects
        """
        if systems is None:
            systems = self.systems

        if re_versions is None:
            re_versions = ['RE1', 'RE2', 'RE3']

        cases = []

        for system in systems:
            system_dir = self.data_dir / system
            if not system_dir.exists():
                print(f"⚠️  System directory not found: {system_dir}")
                continue

            # Scan for RE versions
            for re_version in re_versions:
                re_version_dir = system_dir / re_version
                if not re_version_dir.exists():
                    continue

                # Recursively find all failure case directories
                case_dirs = self._discover_case_directories(re_version_dir)

                for case_dir in case_dirs:
                    case = self._load_single_case(system, re_version, case_dir)
                    if case is not None:
                        cases.append(case)

        if len(cases) == 0:
            print("❌ No cases loaded! Please check dataset structure.")
            print(f"   Expected structure: {self.data_dir}/{{System}}/{{RE_Version}}/")
        else:
            print(f"✅ Loaded {len(cases)} failure cases")

        return cases

    def _discover_case_directories(self, re_version_dir: Path) -> List[Path]:
        """
        Recursively discover directories that contain failure case data

        A directory is a failure case if it contains any of:
        - metrics.csv, data.csv, or simple_metrics.csv

        Args:
            re_version_dir: Path to RE version directory (e.g., TrainTicket/RE2/)

        Returns:
            List of paths to failure case directories
        """
        case_dirs = []

        # Recursively walk through directory tree
        for item in re_version_dir.rglob('*'):
            if item.is_dir():
                # Check if this directory contains a metrics file
                has_metrics = any((item / filename).exists() for filename in self.METRIC_FILENAMES)

                if has_metrics:
                    case_dirs.append(item)

        return case_dirs

    def _parse_folder_labels(self, case_dir: Path) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse service and fault type from folder name

        Expected format: {service}_{fault}/1/ or {service}_{fault}/
        Examples:
          - ts-auth-service_cpu -> service='ts-auth-service', fault='CPU'
          - adservice_cpu -> service='adservice', fault='CPU'
          - carts_f1 -> service='carts', fault='CPU' (via mapping)

        Args:
            case_dir: Path to case directory

        Returns:
            (service, fault_type) or (None, None) if parsing fails
        """
        # Get the parent folder name (skip '1' if it exists)
        folder_name = case_dir.name
        if folder_name.isdigit():
            folder_name = case_dir.parent.name

        # Split by underscore (last part is fault code)
        parts = folder_name.rsplit('_', 1)

        if len(parts) != 2:
            return None, None

        service, fault_code = parts

        # Map fault code to standard fault type
        fault_type = self.FAULT_CODE_MAP.get(fault_code.lower())

        if fault_type is None:
            # Try exact match
            if fault_code.upper() in ['CPU', 'MEM', 'DISK', 'DELAY', 'LOSS', 'SOCKET']:
                fault_type = fault_code.upper()
            else:
                print(f"⚠️  Unknown fault code: {fault_code} in {folder_name}")
                return service, None

        return service, fault_type

    def _find_metric_file(self, case_dir: Path) -> Optional[Path]:
        """
        Find metrics file in directory (handles different naming conventions)

        Tries in priority order: metrics.csv, data.csv, simple_metrics.csv

        Args:
            case_dir: Path to case directory

        Returns:
            Path to metrics file or None
        """
        for filename in self.METRIC_FILENAMES:
            file_path = case_dir / filename
            if file_path.exists():
                return file_path
        return None

    def _load_single_case(self, system: str, re_version: str, case_dir: Path) -> Optional[FailureCase]:
        """
        Load a single failure case with all modalities

        Args:
            system: System name (TrainTicket, SockShop, OnlineBoutique)
            re_version: RE version (RE1, RE2, RE3)
            case_dir: Path to case directory

        Returns:
            FailureCase object or None if loading fails
        """
        try:
            # Parse labels from folder name
            service, fault_type = self._parse_folder_labels(case_dir)

            if service is None or fault_type is None:
                return None

            # Generate case ID
            case_id = f"{system}_{re_version}_{service}_{fault_type}_{case_dir.name}"

            # Load metrics (required)
            metrics_file = self._find_metric_file(case_dir)
            if metrics_file is None:
                return None

            try:
                metrics = pd.read_csv(metrics_file)
            except Exception as e:
                print(f"⚠️  Error reading metrics from {metrics_file}: {e}")
                return None

            # Load logs (optional)
            logs_file = case_dir / 'logs.csv'
            logs = None
            if logs_file.exists():
                try:
                    logs = pd.read_csv(logs_file)
                except Exception as e:
                    print(f"⚠️  Error reading logs from {logs_file}: {e}")

            # Load traces (optional)
            traces_file = case_dir / 'traces.csv'
            traces = None
            if traces_file.exists():
                try:
                    traces = pd.read_csv(traces_file)
                except Exception as e:
                    print(f"⚠️  Error reading traces from {traces_file}: {e}")

            # Root cause indicator (heuristic: fault type mapped to metric)
            indicator_map = {
                'CPU': 'cpu_usage_percent',
                'MEM': 'memory_usage_mb',
                'DISK': 'disk_usage_percent',
                'DELAY': 'latency_ms',
                'LOSS': 'packet_loss_rate',
                'SOCKET': 'socket_count'
            }
            root_cause_indicator = indicator_map.get(fault_type, f"{fault_type.lower()}_metric")

            # Extract case number from directory name
            case_number = None
            if case_dir.name.isdigit():
                case_number = int(case_dir.name)

            # Create failure case object
            case = FailureCase(
                case_id=case_id,
                system=system,
                fault_type=fault_type,
                root_cause_service=service,
                root_cause_indicator=root_cause_indicator,
                metrics=metrics,
                logs=logs,
                traces=traces,
                re_version=re_version,
                case_number=case_number,
                timestamp=None,  # Not available in folder structure
                duration_minutes=60  # Default assumption
            )

            return case

        except Exception as e:
            print(f"⚠️  Error loading case from {case_dir}: {e}")
            return None

    def load_splits(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        random_seed: int = 42,
        stratify_by: str = 'fault_type'
    ) -> Tuple[List[FailureCase], List[FailureCase], List[FailureCase]]:
        """
        Load dataset with stratified train/val/test splits

        Args:
            train_ratio: Proportion for training (default: 0.6)
            val_ratio: Proportion for validation (default: 0.2)
            random_seed: Random seed for reproducibility
            stratify_by: Stratify by 'fault_type', 'system', or None

        Returns:
            (train_cases, val_cases, test_cases)
        """
        cases = self.load_all_cases()

        if len(cases) == 0:
            print("❌ No cases loaded - cannot create splits")
            return [], [], []

        # Stratified split
        if stratify_by == 'fault_type':
            return self._stratified_split_by_fault(cases, train_ratio, val_ratio, random_seed)
        elif stratify_by == 'system':
            return self._stratified_split_by_system(cases, train_ratio, val_ratio, random_seed)
        else:
            return self._random_split(cases, train_ratio, val_ratio, random_seed)

    def _random_split(
        self,
        cases: List[FailureCase],
        train_ratio: float,
        val_ratio: float,
        random_seed: int
    ) -> Tuple[List[FailureCase], List[FailureCase], List[FailureCase]]:
        """Random split without stratification"""
        np.random.seed(random_seed)
        indices = np.random.permutation(len(cases))

        n_train = int(len(cases) * train_ratio)
        n_val = int(len(cases) * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        train_cases = [cases[i] for i in train_idx]
        val_cases = [cases[i] for i in val_idx]
        test_cases = [cases[i] for i in test_idx]

        self._print_split_stats(train_cases, val_cases, test_cases)
        return train_cases, val_cases, test_cases

    def _stratified_split_by_fault(
        self,
        cases: List[FailureCase],
        train_ratio: float,
        val_ratio: float,
        random_seed: int
    ) -> Tuple[List[FailureCase], List[FailureCase], List[FailureCase]]:
        """Stratified split ensuring balanced fault types"""
        np.random.seed(random_seed)

        # Group by fault type
        fault_groups = {}
        for case in cases:
            if case.fault_type not in fault_groups:
                fault_groups[case.fault_type] = []
            fault_groups[case.fault_type].append(case)

        train_cases, val_cases, test_cases = [], [], []

        # Split each fault type group
        for fault_type, group_cases in fault_groups.items():
            indices = np.random.permutation(len(group_cases))

            n_train = int(len(group_cases) * train_ratio)
            n_val = int(len(group_cases) * val_ratio)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]

            train_cases.extend([group_cases[i] for i in train_idx])
            val_cases.extend([group_cases[i] for i in val_idx])
            test_cases.extend([group_cases[i] for i in test_idx])

        # Shuffle again to mix fault types
        np.random.shuffle(train_cases)
        np.random.shuffle(val_cases)
        np.random.shuffle(test_cases)

        self._print_split_stats(train_cases, val_cases, test_cases)
        return train_cases, val_cases, test_cases

    def _stratified_split_by_system(
        self,
        cases: List[FailureCase],
        train_ratio: float,
        val_ratio: float,
        random_seed: int
    ) -> Tuple[List[FailureCase], List[FailureCase], List[FailureCase]]:
        """Stratified split ensuring balanced systems"""
        np.random.seed(random_seed)

        # Group by system
        system_groups = {}
        for case in cases:
            if case.system not in system_groups:
                system_groups[case.system] = []
            system_groups[case.system].append(case)

        train_cases, val_cases, test_cases = [], [], []

        # Split each system group
        for system, group_cases in system_groups.items():
            indices = np.random.permutation(len(group_cases))

            n_train = int(len(group_cases) * train_ratio)
            n_val = int(len(group_cases) * val_ratio)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]

            train_cases.extend([group_cases[i] for i in train_idx])
            val_cases.extend([group_cases[i] for i in val_idx])
            test_cases.extend([group_cases[i] for i in test_idx])

        # Shuffle again to mix systems
        np.random.shuffle(train_cases)
        np.random.shuffle(val_cases)
        np.random.shuffle(test_cases)

        self._print_split_stats(train_cases, val_cases, test_cases)
        return train_cases, val_cases, test_cases

    def _print_split_stats(
        self,
        train_cases: List[FailureCase],
        val_cases: List[FailureCase],
        test_cases: List[FailureCase]
    ):
        """Print statistics about dataset splits"""
        total = len(train_cases) + len(val_cases) + len(test_cases)

        print(f"\n📊 Dataset splits:")
        print(f"   Train: {len(train_cases)} cases ({len(train_cases)/total*100:.1f}%)")
        print(f"   Val:   {len(val_cases)} cases ({len(val_cases)/total*100:.1f}%)")
        print(f"   Test:  {len(test_cases)} cases ({len(test_cases)/total*100:.1f}%)")

    def get_fault_type_distribution(self, cases: List[FailureCase]) -> Dict[str, int]:
        """Get distribution of fault types in dataset"""
        return dict(Counter(case.fault_type for case in cases))

    def get_system_distribution(self, cases: List[FailureCase]) -> Dict[str, int]:
        """Get distribution of systems in dataset"""
        return dict(Counter(case.system for case in cases))

    def get_service_distribution(self, cases: List[FailureCase]) -> Dict[str, int]:
        """Get distribution of root cause services"""
        return dict(Counter(case.root_cause_service for case in cases))


def load_rcaeval_dataset(
    data_dir: str = 'data/RCAEval',
    split: str = 'all'
) -> List[FailureCase]:
    """
    Convenience function to load RCAEval dataset

    Args:
        data_dir: Path to dataset directory
        split: 'all', 'train', 'val', or 'test'

    Returns:
        List of FailureCase objects
    """
    loader = RCAEvalDataLoader(data_dir)

    if split == 'all':
        return loader.load_all_cases()
    else:
        train, val, test = loader.load_splits()
        if split == 'train':
            return train
        elif split == 'val':
            return val
        elif split == 'test':
            return test
        else:
            raise ValueError(f"Unknown split: {split}")


# Example usage
if __name__ == '__main__':
    print("=" * 80)
    print("RCAEval Data Loader - Testing")
    print("=" * 80)

    # Load dataset
    loader = RCAEvalDataLoader('data/RCAEval')

    # Test loading all cases
    print("\n1. Loading all cases...")
    cases = loader.load_all_cases()

    if cases:
        # Get splits
        print("\n2. Creating stratified splits...")
        train, val, test = loader.load_splits(stratify_by='fault_type')

        # Show statistics
        print("\n3. Fault Type Distribution (All Cases):")
        for fault_type, count in sorted(loader.get_fault_type_distribution(cases).items()):
            print(f"   {fault_type}: {count} cases")

        print("\n4. System Distribution (All Cases):")
        for system, count in sorted(loader.get_system_distribution(cases).items()):
            print(f"   {system}: {count} cases")

        print("\n5. Top 10 Root Cause Services:")
        service_dist = loader.get_service_distribution(cases)
        for service, count in sorted(service_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {service}: {count} cases")

        # Show sample case
        print("\n6. Sample Case Details:")
        sample = cases[0]
        print(f"   Case ID: {sample.case_id}")
        print(f"   System: {sample.system}")
        print(f"   RE Version: {sample.re_version}")
        print(f"   Fault Type: {sample.fault_type}")
        print(f"   Root Cause Service: {sample.root_cause_service}")
        print(f"   Root Cause Indicator: {sample.root_cause_indicator}")

        if sample.metrics is not None:
            print(f"\n   Metrics: {sample.metrics.shape} (timesteps × features)")
            print(f"      Columns (first 5): {list(sample.metrics.columns)[:5]}")
        else:
            print(f"   Metrics: Not available")

        if sample.logs is not None:
            print(f"   Logs: {len(sample.logs)} entries")
        else:
            print(f"   Logs: Not available")

        if sample.traces is not None:
            print(f"   Traces: {len(sample.traces)} spans")
        else:
            print(f"   Traces: Not available")

        print("\n" + "=" * 80)
        print("✅ Data loader test complete!")
        print("=" * 80)
    else:
        print("\n❌ No cases loaded - check dataset structure")
