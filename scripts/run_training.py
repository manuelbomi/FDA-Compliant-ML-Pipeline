#!/usr/bin/env python3
"""
Run the FDA-compliant training pipeline.

Usage:
    python scripts/run_training.py --config configs/training_config.yaml
    python scripts/run_training.py --config configs/training_config.yaml --with batch

This script orchestrates the MedicalDeviceTrainingFlow, initializing
the audit trail and ensuring all regulatory documentation is produced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run FDA-compliant ML training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run locally with default config
    python scripts/run_training.py

    # Run with custom config
    python scripts/run_training.py --config configs/training_config.yaml

    # Run on AWS Batch
    python scripts/run_training.py --config configs/training_config.yaml --with batch

    # Override hyperparameters
    python scripts/run_training.py --lr 0.0001 --epochs 50 --batch-size 64
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration YAML",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data path from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/training",
        help="Directory for training outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running training",
    )
    return parser.parse_args()


def validate_config(config_path: str) -> bool:
    """Validate the training configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        True if valid, raises on error.
    """
    import yaml

    if not os.path.exists(config_path):
        print(f"[WARNING] Config file not found: {config_path}")
        print("[WARNING] Using default parameters")
        return True

    with open(config_path) as f:
        config = yaml.safe_load(f)

    required_sections = ["model", "training", "data", "acceptance_criteria"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    required_criteria = ["minimum_val_auc", "minimum_sensitivity"]
    for criterion in required_criteria:
        if criterion not in config.get("acceptance_criteria", {}):
            raise ValueError(f"Missing acceptance criterion: {criterion}")

    print(f"[OK] Configuration validated: {config_path}")
    return True


def setup_audit_trail(output_dir: str):
    """Initialize the audit trail for this training run.

    Args:
        output_dir: Output directory for training artifacts.

    Returns:
        AuditLogger instance.
    """
    from src.compliance.audit_trail import AuditTrailStore, AuditLogger

    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "audit_trail.db")
    store = AuditTrailStore(db_path=db_path)
    logger = AuditLogger(store=store)
    return logger


def main() -> int:
    """Main entry point for training pipeline execution.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()

    print("=" * 60)
    print("FDA-Compliant ML Training Pipeline")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Config: {args.config}")
    print("=" * 60)

    # Validate configuration
    try:
        validate_config(args.config)
    except ValueError as e:
        print(f"[ERROR] Configuration validation failed: {e}")
        return 1

    if args.dry_run:
        print("[DRY RUN] Configuration is valid. Exiting.")
        return 0

    # Build Metaflow command
    flow_args = [f"--config={args.config}", f"--seed={args.seed}"]

    if args.data_path:
        flow_args.append(f"--data-path={args.data_path}")
    if args.lr is not None:
        flow_args.append(f"--lr={args.lr}")
    if args.epochs is not None:
        flow_args.append(f"--epochs={args.epochs}")
    if args.batch_size is not None:
        flow_args.append(f"--batch-size={args.batch_size}")

    print(f"\n[INFO] Flow arguments: {' '.join(flow_args)}")
    print("[INFO] To run the Metaflow flow directly:")
    print(f"  python src/pipeline/training_flow.py run {' '.join(flow_args)}")

    # Initialize audit trail
    audit_logger = setup_audit_trail(args.output_dir)
    print(f"[INFO] Audit trail initialized: {args.output_dir}/audit_trail.db")

    # In production, this would invoke the Metaflow flow:
    # from src.pipeline.training_flow import MedicalDeviceTrainingFlow
    # MedicalDeviceTrainingFlow()
    print("\n[INFO] Training pipeline ready for execution.")
    print("[INFO] Use 'python src/pipeline/training_flow.py run' to start.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
