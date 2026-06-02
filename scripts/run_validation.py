#!/usr/bin/env python3
"""
Run the FDA-compliant validation pipeline.

Usage:
    python scripts/run_validation.py --training-run-id <run-id>
    python scripts/run_validation.py --training-run-id <run-id> --config configs/validation_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run FDA-compliant ML validation pipeline",
    )
    parser.add_argument(
        "--training-run-id",
        type=str,
        required=True,
        help="Metaflow run ID of the training flow to validate",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/validation_config.yaml",
        help="Path to validation configuration YAML",
    )
    parser.add_argument(
        "--previous-run-id",
        type=str,
        default=None,
        help="Previous model run ID for regression testing",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests",
    )
    parser.add_argument(
        "--ni-margin",
        type=float,
        default=0.05,
        help="Non-inferiority margin",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/validation",
        help="Directory for validation outputs",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=True,
        help="Generate regulatory validation report",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for validation pipeline execution."""
    args = parse_args()

    print("=" * 60)
    print("FDA-Compliant ML Validation Pipeline")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Training Run: {args.training_run_id}")
    print(f"Config: {args.config}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    flow_args = [
        f"--training-run-id={args.training_run_id}",
        f"--config={args.config}",
        f"--alpha={args.alpha}",
        f"--ni-margin={args.ni_margin}",
    ]
    if args.previous_run_id:
        flow_args.append(f"--previous-run-id={args.previous_run_id}")

    print(f"\n[INFO] Flow arguments: {' '.join(flow_args)}")
    print("[INFO] To run the Metaflow flow directly:")
    print(f"  python src/pipeline/validation_flow.py run {' '.join(flow_args)}")

    print("\n[INFO] Validation pipeline ready for execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
