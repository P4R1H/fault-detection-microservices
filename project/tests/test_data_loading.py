"""
Test Script: RCAEval Data Loading

Verifies dataset extraction was successful and data loader works correctly.

Usage:
    cd project
    python tests/test_data_loading.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import RCAEvalDataLoader


def test_data_loading():
    """Test RCAEval data loader with extracted dataset"""

    print("=" * 80)
    print("TEST 1: Data Loader Initialization")
    print("=" * 80)

    try:
        loader = RCAEvalDataLoader('data/RCAEval')
        print("✅ Data loader initialized successfully\n")
    except FileNotFoundError as e:
        print(f"❌ Dataset not found: {e}")
        print("Please ensure dataset is extracted to data/RCAEval/")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    print("=" * 80)
    print("TEST 2: Load All Cases")
    print("=" * 80)

    try:
        cases = loader.load_all_cases()
        print(f"✅ Loaded {len(cases)} total failure cases")

        # Expected: 270 cases (90 per system) if all systems extracted
        if len(cases) > 0:
            print(f"   Systems found: {set(c.system for c in cases)}")
        else:
            print("⚠️  No cases loaded - check dataset structure")
            return False

    except Exception as e:
        print(f"❌ Error loading cases: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("TEST 3: Load TrainTicket System")
    print("=" * 80)

    try:
        tt_cases = loader.load_all_cases(systems=['TrainTicket'])
        print(f"✅ Loaded {len(tt_cases)} TrainTicket cases")

        if len(tt_cases) == 0:
            print("⚠️  No TrainTicket cases found")
            return False

    except Exception as e:
        print(f"❌ Error loading TrainTicket: {e}")
        return False

    print("\n" + "=" * 80)
    print("TEST 4: Inspect Sample Case")
    print("=" * 80)

    sample = tt_cases[0]
    print(f"\n📦 Sample Case: {sample.case_id}")
    print(f"   System: {sample.system}")
    print(f"   Fault Type: {sample.fault_type}")
    print(f"   Root Cause Service: {sample.root_cause_service}")
    print(f"   Root Cause Indicator: {sample.root_cause_indicator}")

    # Check modalities
    print(f"\n📊 Data Modalities:")
    if sample.metrics is not None:
        print(f"   ✅ Metrics: {sample.metrics.shape} (timesteps × features)")
        print(f"      Columns (first 5): {list(sample.metrics.columns)[:5]}")
    else:
        print(f"   ❌ Metrics: Not available")

    if sample.logs is not None:
        print(f"   ✅ Logs: {len(sample.logs)} entries")
        if len(sample.logs) > 0:
            print(f"      Columns: {list(sample.logs.columns)}")
    else:
        print(f"   ❌ Logs: Not available")

    if sample.traces is not None:
        print(f"   ✅ Traces: {len(sample.traces)} spans")
        if len(sample.traces) > 0:
            print(f"      Columns: {list(sample.traces.columns)}")
    else:
        print(f"   ❌ Traces: Not available")

    print("\n" + "=" * 80)
    print("TEST 5: Load Train/Val/Test Splits")
    print("=" * 80)

    try:
        train, val, test = loader.load_splits(train_ratio=0.6, val_ratio=0.2, random_seed=42)

        print(f"✅ Dataset splits created:")
        print(f"   Train: {len(train)} cases ({len(train)/len(cases)*100:.1f}%)")
        print(f"   Val:   {len(val)} cases ({len(val)/len(cases)*100:.1f}%)")
        print(f"   Test:  {len(test)} cases ({len(test)/len(cases)*100:.1f}%)")

        # Verify no overlap
        train_ids = set(c.case_id for c in train)
        val_ids = set(c.case_id for c in val)
        test_ids = set(c.case_id for c in test)

        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            print("❌ Data leakage detected: overlapping case IDs in splits!")
            return False
        else:
            print("✅ No data leakage: splits are disjoint")

    except Exception as e:
        print(f"❌ Error creating splits: {e}")
        return False

    print("\n" + "=" * 80)
    print("TEST 6: Fault Type Distribution")
    print("=" * 80)

    fault_dist = loader.get_fault_type_distribution(cases)
    print(f"\n🔥 Fault Types:")
    for fault, count in sorted(fault_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"   {fault}: {count} cases ({count/len(cases)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("TEST 7: System Distribution")
    print("=" * 80)

    system_dist = loader.get_system_distribution(cases)
    print(f"\n📦 Systems:")
    for system, count in sorted(system_dist.items()):
        print(f"   {system}: {count} cases ({count/len(cases)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nDataset Statistics Summary:")
    print(f"  Total cases: {len(cases)}")
    print(f"  Systems: {len(system_dist)}")
    print(f"  Fault types: {len(fault_dist)}")
    print(f"  Data completeness:")
    metrics_count = sum(1 for c in cases if c.metrics is not None)
    logs_count = sum(1 for c in cases if c.logs is not None)
    traces_count = sum(1 for c in cases if c.traces is not None)
    print(f"    Metrics: {metrics_count}/{len(cases)} ({metrics_count/len(cases)*100:.1f}%)")
    print(f"    Logs: {logs_count}/{len(cases)} ({logs_count/len(cases)*100:.1f}%)")
    print(f"    Traces: {traces_count}/{len(cases)} ({traces_count/len(cases)*100:.1f}%)")

    print("\n🎉 Dataset is ready for experiments!")
    return True


if __name__ == '__main__':
    success = test_data_loading()
    sys.exit(0 if success else 1)
