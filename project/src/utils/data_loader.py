"""
RCAEval Data Loader

Loads and preprocesses the RCAEval RE2-TrainTicket dataset including:
- Metrics (77-376 per case at 5-min granularity)
- Logs (8.6-26.9M lines with structure)
- Traces (39.6-76.7M distributed traces with call graphs)
- Ground truth labels (root cause service + indicator)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


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
    timestamp: Optional[pd.Timestamp] = None
    duration_minutes: Optional[int] = None


class RCAEvalDataLoader:
    """
    Data loader for RCAEval RE2 benchmark dataset

    Usage:
        loader = RCAEvalDataLoader(data_dir='project/data/RCAEval')
        train, val, test = loader.load_splits(train_ratio=0.6, val_ratio=0.2)

        for case in train:
            metrics = case.metrics  # pd.DataFrame with 77-376 metrics
            logs = case.logs        # pd.DataFrame with log entries
            traces = case.traces    # pd.DataFrame with distributed traces
            label = case.root_cause_service
    """

    def __init__(self, data_dir: str = 'project/data/RCAEval'):
        """
        Initialize data loader

        Args:
            data_dir: Path to RCAEval dataset directory
        """
        self.data_dir = Path(data_dir)
        self.systems = ['TrainTicket', 'SockShop', 'OnlineBoutique']
        self.fault_types = ['CPU', 'MEM', 'DISK', 'SOCKET', 'DELAY', 'LOSS']

        # Verify dataset exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"RCAEval dataset not found at {self.data_dir}\n"
                f"Please run: python scripts/download_dataset.py"
            )

    def load_all_cases(self, systems: List[str] = None) -> List[FailureCase]:
        """
        Load all failure cases from specified systems

        Args:
            systems: List of system names to load (default: all)

        Returns:
            List of FailureCase objects (270 total: 90 per system)
        """
        if systems is None:
            systems = self.systems

        cases = []
        for system in systems:
            system_dir = self.data_dir / system
            if not system_dir.exists():
                print(f"⚠️  System directory not found: {system_dir}")
                continue

            # Load ground truth labels
            labels_file = system_dir / 'ground_truth.csv'
            if not labels_file.exists():
                print(f"⚠️  Ground truth not found for {system}")
                continue

            labels_df = pd.read_csv(labels_file)

            # Load each case
            for _, row in labels_df.iterrows():
                case = self._load_single_case(system, row)
                if case is not None:
                    cases.append(case)

        print(f"✅ Loaded {len(cases)} failure cases from {len(systems)} systems")
        return cases

    def _load_single_case(self, system: str, label_row: pd.Series) -> Optional[FailureCase]:
        """Load a single failure case with all modalities"""
        try:
            case_id = label_row['case_id']
            case_dir = self.data_dir / system / 'cases' / case_id

            if not case_dir.exists():
                return None

            # Load metrics
            metrics_file = case_dir / 'metrics.csv'
            metrics = pd.read_csv(metrics_file) if metrics_file.exists() else None

            # Load logs
            logs_file = case_dir / 'logs.csv'
            logs = pd.read_csv(logs_file) if logs_file.exists() else None

            # Load traces
            traces_file = case_dir / 'traces.csv'
            traces = pd.read_csv(traces_file) if traces_file.exists() else None

            # Create failure case object
            case = FailureCase(
                case_id=case_id,
                system=system,
                fault_type=label_row['fault_type'],
                root_cause_service=label_row['root_cause_service'],
                root_cause_indicator=label_row['root_cause_indicator'],
                metrics=metrics,
                logs=logs,
                traces=traces,
                timestamp=pd.to_datetime(label_row.get('timestamp')),
                duration_minutes=label_row.get('duration_minutes', 60)
            )

            return case

        except Exception as e:
            print(f"⚠️  Error loading case {label_row.get('case_id', 'unknown')}: {e}")
            return None

    def load_splits(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        random_seed: int = 42
    ) -> Tuple[List[FailureCase], List[FailureCase], List[FailureCase]]:
        """
        Load dataset with train/val/test splits

        Args:
            train_ratio: Proportion for training (default: 0.6)
            val_ratio: Proportion for validation (default: 0.2)
            random_seed: Random seed for reproducibility

        Returns:
            (train_cases, val_cases, test_cases)
            Default split: 162 train / 54 val / 54 test (from 270 total)
        """
        cases = self.load_all_cases()

        # Shuffle with fixed seed
        np.random.seed(random_seed)
        indices = np.random.permutation(len(cases))

        # Calculate split points
        n_train = int(len(cases) * train_ratio)
        n_val = int(len(cases) * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        train_cases = [cases[i] for i in train_idx]
        val_cases = [cases[i] for i in val_idx]
        test_cases = [cases[i] for i in test_idx]

        print(f"📊 Dataset splits:")
        print(f"   Train: {len(train_cases)} cases ({len(train_cases)/len(cases)*100:.1f}%)")
        print(f"   Val:   {len(val_cases)} cases ({len(val_cases)/len(cases)*100:.1f}%)")
        print(f"   Test:  {len(test_cases)} cases ({len(test_cases)/len(cases)*100:.1f}%)")

        return train_cases, val_cases, test_cases

    def get_fault_type_distribution(self, cases: List[FailureCase]) -> Dict[str, int]:
        """Get distribution of fault types in dataset"""
        from collections import Counter
        return dict(Counter(case.fault_type for case in cases))

    def get_system_distribution(self, cases: List[FailureCase]) -> Dict[str, int]:
        """Get distribution of systems in dataset"""
        from collections import Counter
        return dict(Counter(case.system for case in cases))


def load_rcaeval_dataset(
    data_dir: str = 'project/data/RCAEval',
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
    # Load dataset
    loader = RCAEvalDataLoader('project/data/RCAEval')

    # Get splits
    train, val, test = loader.load_splits()

    # Show statistics
    print("\n📊 Fault Type Distribution (Training Set):")
    for fault_type, count in loader.get_fault_type_distribution(train).items():
        print(f"   {fault_type}: {count} cases")

    print("\n📊 System Distribution (Training Set):")
    for system, count in loader.get_system_distribution(train).items():
        print(f"   {system}: {count} cases")

    # Show sample case
    if train:
        sample = train[0]
        print(f"\n📦 Sample Case: {sample.case_id}")
        print(f"   System: {sample.system}")
        print(f"   Fault Type: {sample.fault_type}")
        print(f"   Root Cause: {sample.root_cause_service}")
        if sample.metrics is not None:
            print(f"   Metrics: {len(sample.metrics.columns)} features, {len(sample.metrics)} timesteps")
        if sample.logs is not None:
            print(f"   Logs: {len(sample.logs)} entries")
        if sample.traces is not None:
            print(f"   Traces: {len(sample.traces)} spans")
