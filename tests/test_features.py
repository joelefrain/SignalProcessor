#!/usr/bin/env python
"""
Test script for new benchmark features:
- High resolution spectrum (200+ periods)
- Configurable beta and gamma parameters
- Significant duration calculation
- Enhanced plots with original motion
- Duration variation in summary
"""

import shutil
from pathlib import Path

from src.signalprocessor.benchmark import (
    run_seismomatch_benchmark,
    write_benchmark_report,
    plot_benchmark_spectra,
    write_scaled_benchmark_motions,
)
from src.signalprocessor.spectra import significant_duration, logspace_periods
from src.signalprocessor.io import read_motion_csv

ROOT = Path(__file__).resolve().parent

def test_significant_duration():
    """Test significant duration calculation"""
    print("\n" + "="*70)
    print("TEST 1: Significant Duration Calculation")
    print("="*70)
    
    # Read a sample motion
    motion_path = ROOT / "examples" / "data" / "motion" / "LIMAEW.csv"
    if not motion_path.exists():
        print(f"⚠️  Motion file not found: {motion_path}")
        return
    
    motion = read_motion_csv(motion_path, unit="g", name="LIMAEW")
    duration = significant_duration(motion)
    
    print(f"Motion: {motion.name}")
    print(f"Total duration: {motion.time[-1]:.2f}s")
    print(f"Significant duration (5%-95%): {duration:.2f}s")
    print(f"Ratio: {duration / motion.time[-1] * 100:.1f}%")


def test_high_resolution_periods():
    """Test high resolution period generation"""
    print("\n" + "="*70)
    print("TEST 2: High Resolution Period Generation")
    print("="*70)
    
    periods_200 = logspace_periods(n=200)
    periods_120 = logspace_periods(n=120)
    
    print(f"200 periods: min={periods_200[0]:.4f}s, max={periods_200[-1]:.2f}s, count={len(periods_200)}")
    print(f"120 periods: min={periods_120[0]:.4f}s, max={periods_120[-1]:.2f}s, count={len(periods_120)}")
    print(f"Resolution improvement: {len(periods_200)/len(periods_120):.2f}x")


def test_full_benchmark_with_new_features():
    """Run full benchmark with new features"""
    print("\n" + "="*70)
    print("TEST 3: Full Benchmark Run with New Features")
    print("="*70)
    
    output_dir = ROOT / "examples" / "output" / "test_new_features"
    
    # Clean previous output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning benchmark with new parameters:")
    print(f"  - num_periods: 200")
    print(f"  - beta: 0.25 (default)")
    print(f"  - gamma: 0.5 (default)")
    print(f"  - Output: {output_dir}\n")
    
    # Run benchmark with new parameters
    run = run_seismomatch_benchmark(
        ROOT,
        tolerance=0.3,
        period_range=(0.1, 2.0),
        comparison_metric="mean_relative_error",
        method="frequency_match",
        damping=0.05,
        beta=0.25,
        gamma=0.5,
        num_periods=200,
        iterations=3,
    )
    
    print(f"\nBenchmark Results:")
    print(f"  - Records processed: {len(run.records)}")
    print(f"  - All passed: {run.all_passed}")
    
    # Print duration information
    print(f"\nDuration Analysis:")
    print(f"{'Record':<12} {'Duration (s)':<45} {'Variation (%)':<15}")
    print("-" * 72)
    for record in run.records:
        dur_str = f"Raw={record.duration_raw_s:.2f} → SP={record.duration_ours_s:.2f} / SM={record.duration_seismomatch_s:.2f}"
        print(f"{record.name:<12} {dur_str:<45} {record.duration_variation_pct:>6.2f}")
    
    # Write outputs
    write_benchmark_report(run, output_dir / "reports")
    plot_benchmark_spectra(run, output_dir / "plots")
    write_scaled_benchmark_motions(run, output_dir / "motions", beta=0.25, gamma=0.5, num_periods=200)
    
    print(f"\n✓ Reports saved to: {output_dir / 'reports'}")
    print(f"✓ Plots saved to: {output_dir / 'plots'}")
    print(f"✓ Motions saved to: {output_dir / 'motions'}")
    
    return run


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█ " + " " * 66 + " █")
    print("█  Testing New SignalProcessor Benchmark Features " + " " * 18 + " █")
    print("█ " + " " * 66 + " █")
    print("█" * 70)
    
    try:
        test_significant_duration()
        test_high_resolution_periods()
        test_full_benchmark_with_new_features()
        
        print("\n" + "█" * 70)
        print("█ ✓ All tests completed successfully! " + " " * 28 + " █")
        print("█" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
