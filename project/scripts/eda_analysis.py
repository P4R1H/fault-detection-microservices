"""
Exploratory Data Analysis for RCAEval Dataset

Analyzes all three modalities:
- Metrics: Temporal patterns, correlations, anomaly characteristics
- Logs: Template distributions, error patterns, service coverage
- Traces: Service dependencies, call graphs, latency distributions

Usage:
    python scripts/eda_analysis.py --system TrainTicket --version RE2
    python scripts/eda_analysis.py --all  # Analyze all systems
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import Counter
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import RCAEvalDataLoader, FailureCase


class RCAEvalEDA:
    """Comprehensive EDA for RCAEval dataset"""

    def __init__(self, data_dir: str = 'project/data/RCAEval', output_dir: str = 'project/outputs/eda'):
        """
        Initialize EDA analyzer

        Args:
            data_dir: Path to RCAEval dataset
            output_dir: Directory to save analysis outputs
        """
        self.loader = RCAEvalDataLoader(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    def analyze_all(self, systems: List[str] = None):
        """Run complete EDA pipeline"""
        print("=" * 80)
        print("RCAEval Dataset - Exploratory Data Analysis")
        print("=" * 80)

        # Load dataset
        cases = self.loader.load_all_cases(systems=systems)

        if not cases:
            print("❌ No cases loaded. Please ensure dataset is extracted.")
            return

        print(f"\n📊 Loaded {len(cases)} failure cases\n")

        # 1. Dataset-level statistics
        self.analyze_dataset_statistics(cases)

        # 2. Metrics analysis
        self.analyze_metrics_modality(cases)

        # 3. Logs analysis
        self.analyze_logs_modality(cases)

        # 4. Traces analysis
        self.analyze_traces_modality(cases)

        # 5. Cross-modality analysis
        self.analyze_cross_modality_patterns(cases)

        # 6. Root cause patterns
        self.analyze_root_cause_patterns(cases)

        print(f"\n✅ EDA complete! Results saved to: {self.output_dir}")

    def analyze_dataset_statistics(self, cases: List[FailureCase]):
        """Analyze high-level dataset statistics"""
        print("=" * 80)
        print("1. Dataset-Level Statistics")
        print("=" * 80)

        # System distribution
        system_dist = Counter(case.system for case in cases)
        print(f"\n📦 System Distribution:")
        for system, count in sorted(system_dist.items()):
            print(f"   {system}: {count} cases ({count/len(cases)*100:.1f}%)")

        # Fault type distribution
        fault_dist = Counter(case.fault_type for case in cases)
        print(f"\n🔥 Fault Type Distribution:")
        for fault, count in sorted(fault_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"   {fault}: {count} cases ({count/len(cases)*100:.1f}%)")

        # Root cause service distribution
        rc_dist = Counter(case.root_cause_service for case in cases)
        print(f"\n🎯 Root Cause Service Distribution:")
        print(f"   Unique services: {len(rc_dist)}")
        print(f"   Top 10 services:")
        for service, count in rc_dist.most_common(10):
            print(f"      {service}: {count} cases")

        # Data modality availability
        metrics_available = sum(1 for c in cases if c.metrics is not None)
        logs_available = sum(1 for c in cases if c.logs is not None)
        traces_available = sum(1 for c in cases if c.traces is not None)

        print(f"\n📊 Data Modality Availability:")
        print(f"   Metrics: {metrics_available}/{len(cases)} cases ({metrics_available/len(cases)*100:.1f}%)")
        print(f"   Logs: {logs_available}/{len(cases)} cases ({logs_available/len(cases)*100:.1f}%)")
        print(f"   Traces: {traces_available}/{len(cases)} cases ({traces_available/len(cases)*100:.1f}%)")

        # Save distribution plots
        self._plot_distributions(system_dist, fault_dist, rc_dist)

    def analyze_metrics_modality(self, cases: List[FailureCase]):
        """Analyze metrics modality characteristics"""
        print("\n" + "=" * 80)
        print("2. Metrics Modality Analysis")
        print("=" * 80)

        metrics_cases = [c for c in cases if c.metrics is not None]

        if not metrics_cases:
            print("⚠️  No metrics data available")
            return

        print(f"\n📈 Analyzing {len(metrics_cases)} cases with metrics")

        # Dimensionality analysis
        feature_counts = [len(c.metrics.columns) for c in metrics_cases]
        timestep_counts = [len(c.metrics) for c in metrics_cases]

        print(f"\n🔢 Dimensionality:")
        print(f"   Features per case:")
        print(f"      Min: {min(feature_counts)}, Max: {max(feature_counts)}, Median: {np.median(feature_counts):.0f}")
        print(f"   Timesteps per case:")
        print(f"      Min: {min(timestep_counts)}, Max: {max(timestep_counts)}, Median: {np.median(timestep_counts):.0f}")

        # Sample case analysis
        sample_case = metrics_cases[0]
        if sample_case.metrics is not None:
            df = sample_case.metrics
            print(f"\n📊 Sample Case ({sample_case.case_id}):")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)[:5]}... (showing first 5)")

            # Basic statistics
            print(f"\n📉 Statistical Properties:")
            print(f"   Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / df.size * 100:.2f}%)")
            print(f"   Mean: {df.mean().mean():.4f}")
            print(f"   Std: {df.std().mean():.4f}")

        # Save metrics analysis
        self._save_metrics_statistics(metrics_cases)

    def analyze_logs_modality(self, cases: List[FailureCase]):
        """Analyze logs modality characteristics"""
        print("\n" + "=" * 80)
        print("3. Logs Modality Analysis")
        print("=" * 80)

        logs_cases = [c for c in cases if c.logs is not None]

        if not logs_cases:
            print("⚠️  No logs data available")
            return

        print(f"\n📝 Analyzing {len(logs_cases)} cases with logs")

        # Log volume analysis
        log_counts = [len(c.logs) for c in logs_cases]

        print(f"\n📊 Log Volume:")
        print(f"   Total logs: {sum(log_counts):,}")
        print(f"   Logs per case:")
        print(f"      Min: {min(log_counts):,}")
        print(f"      Max: {max(log_counts):,}")
        print(f"      Median: {np.median(log_counts):,.0f}")
        print(f"      Mean: {np.mean(log_counts):,.0f}")

        # Sample case analysis
        sample_case = logs_cases[0]
        if sample_case.logs is not None:
            df = sample_case.logs
            print(f"\n📄 Sample Case ({sample_case.case_id}):")
            print(f"   Log entries: {len(df):,}")
            print(f"   Columns: {list(df.columns)}")

            # Check for common log fields
            if 'level' in df.columns:
                level_dist = Counter(df['level'])
                print(f"\n🔍 Log Level Distribution:")
                for level, count in sorted(level_dist.items(), key=lambda x: x[1], reverse=True):
                    print(f"      {level}: {count:,} ({count/len(df)*100:.1f}%)")

    def analyze_traces_modality(self, cases: List[FailureCase]):
        """Analyze traces modality characteristics"""
        print("\n" + "=" * 80)
        print("4. Traces Modality Analysis")
        print("=" * 80)

        traces_cases = [c for c in cases if c.traces is not None]

        if not traces_cases:
            print("⚠️  No traces data available")
            return

        print(f"\n🔗 Analyzing {len(traces_cases)} cases with traces")

        # Trace volume analysis
        trace_counts = [len(c.traces) for c in traces_cases]

        print(f"\n📊 Trace Volume:")
        print(f"   Total spans: {sum(trace_counts):,}")
        print(f"   Spans per case:")
        print(f"      Min: {min(trace_counts):,}")
        print(f"      Max: {max(trace_counts):,}")
        print(f"      Median: {np.median(trace_counts):,.0f}")
        print(f"      Mean: {np.mean(trace_counts):,.0f}")

        # Sample case analysis
        sample_case = traces_cases[0]
        if sample_case.traces is not None:
            df = sample_case.traces
            print(f"\n🔍 Sample Case ({sample_case.case_id}):")
            print(f"   Trace spans: {len(df):,}")
            print(f"   Columns: {list(df.columns)}")

            # Service graph analysis
            if 'service' in df.columns:
                services = df['service'].unique()
                print(f"\n🏢 Services in trace:")
                print(f"      Unique services: {len(services)}")
                print(f"      Top services: {list(services)[:5]}")

    def analyze_cross_modality_patterns(self, cases: List[FailureCase]):
        """Analyze patterns across modalities"""
        print("\n" + "=" * 80)
        print("5. Cross-Modality Patterns")
        print("=" * 80)

        # Completeness analysis
        completeness = {
            'all_three': 0,
            'metrics_logs': 0,
            'metrics_traces': 0,
            'logs_traces': 0,
            'metrics_only': 0,
            'logs_only': 0,
            'traces_only': 0,
            'none': 0
        }

        for case in cases:
            has_metrics = case.metrics is not None
            has_logs = case.logs is not None
            has_traces = case.traces is not None

            if has_metrics and has_logs and has_traces:
                completeness['all_three'] += 1
            elif has_metrics and has_logs:
                completeness['metrics_logs'] += 1
            elif has_metrics and has_traces:
                completeness['metrics_traces'] += 1
            elif has_logs and has_traces:
                completeness['logs_traces'] += 1
            elif has_metrics:
                completeness['metrics_only'] += 1
            elif has_logs:
                completeness['logs_only'] += 1
            elif has_traces:
                completeness['traces_only'] += 1
            else:
                completeness['none'] += 1

        print(f"\n📊 Modality Completeness:")
        for pattern, count in sorted(completeness.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"   {pattern}: {count} cases ({count/len(cases)*100:.1f}%)")

    def analyze_root_cause_patterns(self, cases: List[FailureCase]):
        """Analyze root cause distribution patterns"""
        print("\n" + "=" * 80)
        print("6. Root Cause Patterns")
        print("=" * 80)

        # Fault type by system
        print(f"\n🔥 Fault Types by System:")
        for system in set(c.system for c in cases):
            system_cases = [c for c in cases if c.system == system]
            fault_dist = Counter(c.fault_type for c in system_cases)
            print(f"\n   {system}:")
            for fault, count in sorted(fault_dist.items()):
                print(f"      {fault}: {count} cases")

        # Root cause indicator patterns
        rc_indicators = Counter(c.root_cause_indicator for c in cases)
        print(f"\n🎯 Root Cause Indicators:")
        print(f"   Unique indicators: {len(rc_indicators)}")
        print(f"   Top 10 indicators:")
        for indicator, count in rc_indicators.most_common(10):
            print(f"      {indicator}: {count} cases")

    def _plot_distributions(self, system_dist: Dict, fault_dist: Dict, rc_dist: Dict):
        """Create distribution plots"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # System distribution
        axes[0].bar(system_dist.keys(), system_dist.values())
        axes[0].set_title('System Distribution')
        axes[0].set_ylabel('Number of Cases')
        axes[0].tick_params(axis='x', rotation=45)

        # Fault type distribution
        axes[1].bar(fault_dist.keys(), fault_dist.values())
        axes[1].set_title('Fault Type Distribution')
        axes[1].set_ylabel('Number of Cases')
        axes[1].tick_params(axis='x', rotation=45)

        # Root cause service distribution (top 10)
        top_rc = dict(sorted(rc_dist.items(), key=lambda x: x[1], reverse=True)[:10])
        axes[2].barh(list(top_rc.keys()), list(top_rc.values()))
        axes[2].set_title('Top 10 Root Cause Services')
        axes[2].set_xlabel('Number of Cases')
        axes[2].invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n💾 Saved distribution plots to: {self.output_dir / 'dataset_distributions.png'}")

    def _save_metrics_statistics(self, metrics_cases: List[FailureCase]):
        """Save detailed metrics statistics"""
        stats_file = self.output_dir / 'metrics_statistics.txt'

        with open(stats_file, 'w') as f:
            f.write("Metrics Modality Statistics\n")
            f.write("=" * 80 + "\n\n")

            for i, case in enumerate(metrics_cases[:5]):  # First 5 cases
                if case.metrics is not None:
                    df = case.metrics
                    f.write(f"Case {i+1}: {case.case_id}\n")
                    f.write(f"  Shape: {df.shape}\n")
                    f.write(f"  Columns: {list(df.columns)}\n")
                    f.write(f"  Summary statistics:\n")
                    f.write(f"{df.describe()}\n\n")

        print(f"💾 Saved metrics statistics to: {stats_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='RCAEval Dataset EDA')
    parser.add_argument('--all', action='store_true', help='Analyze all systems')
    parser.add_argument('--systems', nargs='+', choices=['TrainTicket', 'SockShop', 'OnlineBoutique'],
                        help='Specific systems to analyze')
    parser.add_argument('--data-dir', type=str, default='project/data/RCAEval',
                        help='Path to RCAEval dataset')
    parser.add_argument('--output-dir', type=str, default='project/outputs/eda',
                        help='Output directory for analysis results')

    args = parser.parse_args()

    # Determine systems to analyze
    systems = None
    if args.systems:
        systems = args.systems
    elif not args.all:
        systems = ['TrainTicket']  # Default to TrainTicket for quick analysis

    # Run EDA
    eda = RCAEvalEDA(data_dir=args.data_dir, output_dir=args.output_dir)
    eda.analyze_all(systems=systems)


if __name__ == '__main__':
    main()
