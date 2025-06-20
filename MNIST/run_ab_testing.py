#!/usr/bin/env python3
"""
Quick Start Script for MNIST A/B Testing

This script provides an easy way to run A/B testing with sensible defaults
and common configurations.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Quick Start MNIST A/B Testing")
    parser.add_argument("--mode", choices=["quick", "full", "analysis"], default="quick",
                       help="Mode to run: quick (3 runs, 20 epochs), full (5 runs, 30 epochs), or analysis only")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--output-dir", type=str, default="ab_test_results", help="Output directory")
    parser.add_argument("--results-file", type=str, help="Results file for analysis mode")
    
    args = parser.parse_args()
    
    # Set up parameters based on mode
    if args.mode == "quick":
        num_runs = 3
        max_epochs = 20
        print("Running QUICK mode: 3 runs per configuration, 20 epochs max")
    elif args.mode == "full":
        num_runs = 5
        max_epochs = 30
        print("Running FULL mode: 5 runs per configuration, 30 epochs max")
    else:  # analysis mode
        if not args.results_file:
            print("Error: --results-file is required for analysis mode")
            sys.exit(1)
        
        print("Running ANALYSIS mode")
        cmd = [
            sys.executable, "ab_testing_analyzer.py",
            "--results", args.results_file,
            "--output-dir", "analysis_results"
        ]
        
        if run_command(cmd, "Analysis"):
            print("\nAnalysis complete! Check the 'analysis_results' directory.")
        else:
            print("\nAnalysis failed!")
            sys.exit(1)
        return
    
    # Build command for A/B testing
    cmd = [
        sys.executable, "ab_testing_runner.py",
        "--num-runs", str(num_runs),
        "--max-epochs", str(max_epochs),
        "--output-dir", args.output_dir
    ]
    
    if args.gpu:
        cmd.append("--gpu")
    if args.mixed_precision:
        cmd.append("--mixed-precision")
    
    # Run A/B testing
    if run_command(cmd, "A/B Testing"):
        print(f"\nA/B testing complete! Results saved to: {args.output_dir}")
        
        # Automatically run analysis
        results_file = Path(args.output_dir) / "final_results.json"
        if results_file.exists():
            print("\nRunning automatic analysis...")
            analysis_cmd = [
                sys.executable, "ab_testing_analyzer.py",
                "--results", str(results_file),
                "--output-dir", "analysis_results"
            ]
            
            if run_command(analysis_cmd, "Analysis"):
                print("\n✓ Complete! Check both directories:")
                print(f"  - A/B Test Results: {args.output_dir}")
                print(f"  - Analysis Results: analysis_results")
            else:
                print("\n✓ A/B testing completed, but analysis failed.")
                print(f"  You can run analysis manually with:")
                print(f"  python ab_testing_analyzer.py --results {results_file}")
        else:
            print(f"\n✓ A/B testing completed, but results file not found: {results_file}")
    else:
        print("\nA/B testing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 