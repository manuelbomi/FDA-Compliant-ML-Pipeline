#!/usr/bin/env python3
"""
Generate professional pipeline visualization screenshots for the README.

Creates three publication-quality PNG images:
1. pipeline_overview.png - Metaflow DAG visualization
2. validation_report.png - FDA validation report
3. audit_trail.png - Audit trail dashboard
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent


def generate_pipeline_overview():
    """Generate a Metaflow-style pipeline DAG visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(8, 9.5, "FDA-Compliant ML Pipeline", fontsize=22, fontweight="bold",
            color="white", ha="center", va="center", fontfamily="monospace")
    ax.text(8, 9.05, "MedicalDeviceTrainingFlow / Run ID: 1717200000",
            fontsize=11, color="#8b949e", ha="center", va="center", fontfamily="monospace")

    # Pipeline steps
    steps = [
        {"name": "start", "status": "completed", "time": "0.3s", "artifacts": 2,
         "x": 2, "y": 7.2, "desc": "Design Input\nDocumentation"},
        {"name": "load_and_version_data", "status": "completed", "time": "12.4s", "artifacts": 4,
         "x": 5.5, "y": 7.2, "desc": "Data Versioning\n& Quality Check"},
        {"name": "preprocess", "status": "completed", "time": "8.1s", "artifacts": 3,
         "x": 9, "y": 7.2, "desc": "Deterministic\nPreprocessing"},
        {"name": "train_model", "status": "completed", "time": "4m 23s", "artifacts": 5,
         "x": 12.5, "y": 7.2, "desc": "GPU Training\n@resources(gpu=1)"},
        {"name": "evaluate", "status": "completed", "time": "15.7s", "artifacts": 4,
         "x": 5.5, "y": 4.5, "desc": "Bootstrap CI\nAcceptance Criteria"},
        {"name": "end", "status": "completed", "time": "1.2s", "artifacts": 1,
         "x": 12.5, "y": 4.5, "desc": "Device History\nRecord (DHR)"},
    ]

    status_colors = {
        "completed": "#238636",
        "running": "#1f6feb",
        "failed": "#da3633",
        "pending": "#6e7681",
    }

    for step in steps:
        x, y = step["x"], step["y"]
        color = status_colors[step["status"]]

        # Step box
        box = FancyBboxPatch(
            (x - 1.4, y - 0.8), 2.8, 1.6,
            boxstyle="round,pad=0.15",
            facecolor="#161b22",
            edgecolor=color,
            linewidth=2.5,
        )
        ax.add_patch(box)

        # Step name
        ax.text(x, y + 0.35, step["name"], fontsize=8.5, fontweight="bold",
                color="white", ha="center", va="center", fontfamily="monospace")

        # Description
        ax.text(x, y - 0.15, step["desc"], fontsize=7,
                color="#8b949e", ha="center", va="center", fontfamily="monospace")

        # Status indicator
        ax.plot(x - 1.1, y + 0.55, "o", color=color, markersize=8)
        ax.text(x - 0.85, y + 0.55, "OK", fontsize=6, color=color,
                ha="left", va="center", fontweight="bold", fontfamily="monospace")

        # Time and artifacts
        ax.text(x + 0.7, y + 0.55, step["time"], fontsize=6.5, color="#58a6ff",
                ha="center", va="center", fontfamily="monospace")
        ax.text(x + 1.15, y + 0.55, f'{step["artifacts"]}A', fontsize=6.5,
                color="#f0883e", ha="center", va="center", fontfamily="monospace")

    # Arrows between steps
    arrow_style = "Simple,tail_width=1.5,head_width=8,head_length=6"
    arrows = [
        (steps[0], steps[1]),
        (steps[1], steps[2]),
        (steps[2], steps[3]),
        (steps[3], steps[5]),
        (steps[1], steps[4]),
        (steps[4], steps[5]),
    ]
    for s1, s2 in arrows:
        ax.annotate("", xy=(s2["x"] - 1.4, s2["y"]),
                    xytext=(s1["x"] + 1.4, s1["y"]),
                    arrowprops=dict(arrowstyle="->", color="#30363d",
                                   lw=2, connectionstyle="arc3,rad=0.0"))

    # Artifact summary panel
    panel_y = 2.2
    panel = FancyBboxPatch(
        (1.5, panel_y - 0.8), 13, 1.6,
        boxstyle="round,pad=0.15",
        facecolor="#161b22",
        edgecolor="#30363d",
        linewidth=1.5,
    )
    ax.add_patch(panel)

    ax.text(8, panel_y + 0.45, "Run Summary", fontsize=11, fontweight="bold",
            color="white", ha="center", va="center", fontfamily="monospace")

    summary_items = [
        ("Status", "COMPLETED", "#238636"),
        ("Total Time", "5m 01s", "#58a6ff"),
        ("Artifacts", "19", "#f0883e"),
        ("Model Hash", "a3f8c2...9d1e", "#e3b341"),
        ("Val AUC", "0.9120", "#238636"),
        ("Acceptance", "PASS", "#238636"),
    ]
    for i, (label, value, color) in enumerate(summary_items):
        x_pos = 2.5 + i * 2.2
        ax.text(x_pos, panel_y - 0.05, label, fontsize=7, color="#8b949e",
                ha="center", va="center", fontfamily="monospace")
        ax.text(x_pos, panel_y - 0.35, value, fontsize=9, fontweight="bold",
                color=color, ha="center", va="center", fontfamily="monospace")

    # FDA badge
    badge = FancyBboxPatch(
        (0.3, 0.2), 3.2, 0.6,
        boxstyle="round,pad=0.1",
        facecolor="#1a4730",
        edgecolor="#238636",
        linewidth=1.5,
    )
    ax.add_patch(badge)
    ax.text(1.9, 0.5, "FDA 21 CFR 820 Compliant", fontsize=8, fontweight="bold",
            color="#238636", ha="center", va="center", fontfamily="monospace")

    # Metaflow badge
    badge2 = FancyBboxPatch(
        (4, 0.2), 2.8, 0.6,
        boxstyle="round,pad=0.1",
        facecolor="#1a1a3e",
        edgecolor="#8957e5",
        linewidth=1.5,
    )
    ax.add_patch(badge2)
    ax.text(5.4, 0.5, "Metaflow on AWS", fontsize=8, fontweight="bold",
            color="#8957e5", ha="center", va="center", fontfamily="monospace")

    plt.tight_layout(pad=0.5)
    output_path = OUTPUT_DIR / "pipeline_overview.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_validation_report():
    """Generate an FDA-style validation report visualization."""
    fig = plt.figure(figsize=(16, 11), facecolor="#0d1117")
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.8, 1.2, 1.0],
                          hspace=0.35, wspace=0.3)

    # Title area
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor("#0d1117")
    ax_title.axis("off")
    ax_title.text(0.5, 0.85, "FDA SaMD Validation Report", fontsize=22,
                 fontweight="bold", color="white", ha="center", va="center",
                 transform=ax_title.transAxes, fontfamily="monospace")
    ax_title.text(0.5, 0.55, "Verification & Validation Protocol Results | MammoScreen AI v3.0",
                 fontsize=11, color="#8b949e", ha="center", va="center",
                 transform=ax_title.transAxes, fontfamily="monospace")
    ax_title.text(0.5, 0.25, "Run: val-20260601-001 | Test Set: n=15,000 | Predicate: PredCAD v2.0 (K123456)",
                 fontsize=9, color="#6e7681", ha="center", va="center",
                 transform=ax_title.transAxes, fontfamily="monospace")

    # Performance metrics table
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_metrics.set_facecolor("#0d1117")
    ax_metrics.axis("off")

    metrics = [
        ("AUC-ROC", 0.912, 0.898, 0.926, 0.850, True),
        ("Sensitivity", 0.853, 0.832, 0.874, 0.800, True),
        ("Specificity", 0.901, 0.893, 0.909, 0.850, True),
        ("PPV", 0.342, 0.312, 0.372, 0.300, True),
        ("NPV", 0.987, 0.983, 0.991, 0.980, True),
        ("Avg Precision", 0.624, 0.598, 0.650, 0.500, True),
    ]

    ax_metrics.text(0.5, 0.98, "Performance Metrics vs. Predetermined Thresholds",
                   fontsize=11, fontweight="bold", color="white", ha="center",
                   va="top", transform=ax_metrics.transAxes, fontfamily="monospace")

    headers = ["Metric", "Value", "95% CI", "Threshold", "Status"]
    col_x = [0.02, 0.22, 0.40, 0.68, 0.85]

    for i, h in enumerate(headers):
        ax_metrics.text(col_x[i], 0.88, h, fontsize=8, fontweight="bold",
                       color="#58a6ff", va="top", transform=ax_metrics.transAxes,
                       fontfamily="monospace")

    # Separator line
    ax_metrics.plot([0.01, 0.99], [0.85, 0.85], color="#30363d",
                   linewidth=1, transform=ax_metrics.transAxes, clip_on=False)

    for j, (name, val, ci_lo, ci_hi, thresh, passed) in enumerate(metrics):
        y = 0.78 - j * 0.12
        color = "#238636" if passed else "#da3633"
        status = "PASS" if passed else "FAIL"

        ax_metrics.text(col_x[0], y, name, fontsize=8, color="white",
                       va="center", transform=ax_metrics.transAxes, fontfamily="monospace")
        ax_metrics.text(col_x[1], y, f"{val:.3f}", fontsize=8, fontweight="bold",
                       color=color, va="center", transform=ax_metrics.transAxes,
                       fontfamily="monospace")
        ax_metrics.text(col_x[2], y, f"[{ci_lo:.3f}, {ci_hi:.3f}]", fontsize=7.5,
                       color="#8b949e", va="center", transform=ax_metrics.transAxes,
                       fontfamily="monospace")
        ax_metrics.text(col_x[3], y, f">= {thresh:.3f}", fontsize=8,
                       color="#6e7681", va="center", transform=ax_metrics.transAxes,
                       fontfamily="monospace")
        ax_metrics.text(col_x[4], y, f"[{status}]", fontsize=8, fontweight="bold",
                       color=color, va="center", transform=ax_metrics.transAxes,
                       fontfamily="monospace")

    # Subgroup analysis heatmap
    ax_heatmap = fig.add_subplot(gs[1, 1])
    ax_heatmap.set_facecolor("#0d1117")

    subgroups = ["40-49", "50-59", "60-69", "70+"]
    categories = ["AUC", "Sens", "Spec", "PPV"]
    data = np.array([
        [0.905, 0.840, 0.892, 0.310],
        [0.918, 0.860, 0.905, 0.355],
        [0.920, 0.870, 0.910, 0.368],
        [0.898, 0.830, 0.885, 0.298],
    ])

    im = ax_heatmap.imshow(data, cmap="RdYlGn", aspect="auto",
                           vmin=0.25, vmax=0.95)
    ax_heatmap.set_xticks(range(len(categories)))
    ax_heatmap.set_xticklabels(categories, fontsize=8, color="white",
                               fontfamily="monospace")
    ax_heatmap.set_yticks(range(len(subgroups)))
    ax_heatmap.set_yticklabels(subgroups, fontsize=8, color="white",
                               fontfamily="monospace")
    ax_heatmap.set_title("Subgroup Performance (Age)", fontsize=11,
                        fontweight="bold", color="white", pad=10,
                        fontfamily="monospace")

    for i in range(len(subgroups)):
        for j in range(len(categories)):
            text_color = "black" if data[i, j] > 0.6 else "white"
            ax_heatmap.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                          fontsize=9, color=text_color, fontweight="bold",
                          fontfamily="monospace")

    ax_heatmap.tick_params(colors="white")
    for spine in ax_heatmap.spines.values():
        spine.set_edgecolor("#30363d")

    # Statistical test results
    ax_stats = fig.add_subplot(gs[2, 0])
    ax_stats.set_facecolor("#0d1117")
    ax_stats.axis("off")

    ax_stats.text(0.5, 0.95, "Statistical Hypothesis Tests", fontsize=11,
                 fontweight="bold", color="white", ha="center", va="top",
                 transform=ax_stats.transAxes, fontfamily="monospace")

    tests = [
        ("Non-Inferiority (AUC)", "p = 0.0023", "PASS", "#238636"),
        ("Equivalence (Sensitivity)", "p = 0.0156", "PASS", "#238636"),
        ("Regression vs. v2.0", "No regression", "PASS", "#238636"),
        ("Bias: Equalized Odds", "diff = 0.042", "PASS", "#238636"),
        ("Bias: Demographic Parity", "diff = 0.038", "PASS", "#238636"),
        ("Bias: Calibration", "diff = 0.021", "PASS", "#238636"),
    ]

    for i, (name, detail, status, color) in enumerate(tests):
        y = 0.82 - i * 0.13
        ax_stats.text(0.05, y, f"  {name}", fontsize=8, color="white",
                     va="center", transform=ax_stats.transAxes, fontfamily="monospace")
        ax_stats.text(0.60, y, detail, fontsize=8, color="#8b949e",
                     va="center", transform=ax_stats.transAxes, fontfamily="monospace")
        ax_stats.text(0.85, y, f"[{status}]", fontsize=8, fontweight="bold",
                     color=color, va="center", transform=ax_stats.transAxes,
                     fontfamily="monospace")

    # Overall verdict
    ax_verdict = fig.add_subplot(gs[2, 1])
    ax_verdict.set_facecolor("#0d1117")
    ax_verdict.axis("off")

    verdict_box = FancyBboxPatch(
        (0.05, 0.15), 0.9, 0.75,
        boxstyle="round,pad=0.05",
        facecolor="#0d2818",
        edgecolor="#238636",
        linewidth=3,
        transform=ax_verdict.transAxes,
    )
    ax_verdict.add_patch(verdict_box)

    ax_verdict.text(0.5, 0.75, "VALIDATION", fontsize=14,
                   color="#238636", ha="center", va="center",
                   transform=ax_verdict.transAxes, fontfamily="monospace",
                   fontweight="bold")
    ax_verdict.text(0.5, 0.55, "PASSED", fontsize=28,
                   color="#238636", ha="center", va="center",
                   transform=ax_verdict.transAxes, fontfamily="monospace",
                   fontweight="bold")
    ax_verdict.text(0.5, 0.38, "All 42 tests passed", fontsize=10,
                   color="#8b949e", ha="center", va="center",
                   transform=ax_verdict.transAxes, fontfamily="monospace")
    ax_verdict.text(0.5, 0.25, "Non-inferior to predicate device", fontsize=9,
                   color="#6e7681", ha="center", va="center",
                   transform=ax_verdict.transAxes, fontfamily="monospace")

    output_path = OUTPUT_DIR / "validation_report.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_audit_trail():
    """Generate an audit trail dashboard visualization."""
    fig = plt.figure(figsize=(16, 11), facecolor="#0d1117")
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.6, 1.2, 1.0],
                          hspace=0.35, wspace=0.3)

    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor("#0d1117")
    ax_title.axis("off")
    ax_title.text(0.5, 0.8, "FDA Audit Trail Dashboard", fontsize=22,
                 fontweight="bold", color="white", ha="center", va="center",
                 transform=ax_title.transAxes, fontfamily="monospace")
    ax_title.text(0.5, 0.4, "Immutable Event Log | Hash Chain Verified | 21 CFR Part 11 Compliant",
                 fontsize=11, color="#8b949e", ha="center", va="center",
                 transform=ax_title.transAxes, fontfamily="monospace")

    # Timeline of events
    ax_timeline = fig.add_subplot(gs[1, 0])
    ax_timeline.set_facecolor("#161b22")
    ax_timeline.set_title("Event Timeline", fontsize=11, fontweight="bold",
                         color="white", pad=10, fontfamily="monospace")

    events = [
        ("08:00:00", "pipeline.start", "Training pipeline started", "#58a6ff"),
        ("08:00:02", "data.access", "Dataset v2.1 loaded (n=85,000)", "#f0883e"),
        ("08:00:15", "data.transform", "PHI verification passed", "#238636"),
        ("08:00:23", "model.train.start", "Training started (GPU p3.2xl)", "#8957e5"),
        ("08:04:46", "model.train.complete", "Model trained (AUC=0.912)", "#238636"),
        ("08:05:02", "model.validate", "42/42 validation tests passed", "#238636"),
        ("08:05:18", "model.register", "Model v3.0 registered in registry", "#58a6ff"),
        ("08:05:20", "auth.approve", "Deployment approved by J. Smith", "#e3b341"),
        ("08:05:45", "model.deploy", "Canary deployed (5% traffic)", "#8957e5"),
        ("08:35:45", "compliance.check", "Canary evaluation passed", "#238636"),
    ]

    for i, (time, etype, desc, color) in enumerate(events):
        y = len(events) - i - 0.5
        ax_timeline.barh(y, 0.8, left=0, height=0.6, color=color, alpha=0.15,
                        edgecolor=color, linewidth=1)
        ax_timeline.plot(0.05, y, "o", color=color, markersize=7, zorder=5)
        ax_timeline.text(0.15, y + 0.05, time, fontsize=7, color="#6e7681",
                        va="center", fontfamily="monospace")
        ax_timeline.text(1.0, y + 0.05, desc, fontsize=7.5, color="white",
                        va="center", fontfamily="monospace")

    ax_timeline.set_xlim(-0.1, 8)
    ax_timeline.set_ylim(-0.5, len(events) + 0.5)
    ax_timeline.set_yticks([])
    ax_timeline.set_xticks([])
    for spine in ax_timeline.spines.values():
        spine.set_edgecolor("#30363d")
    ax_timeline.tick_params(colors="white")

    # Data lineage graph
    ax_lineage = fig.add_subplot(gs[1, 1])
    ax_lineage.set_facecolor("#161b22")
    ax_lineage.set_title("Data Lineage Graph", fontsize=11, fontweight="bold",
                        color="white", pad=10, fontfamily="monospace")
    ax_lineage.axis("off")

    nodes = [
        ("Raw DICOM\nData (Site A-E)", 0.15, 0.85, "#58a6ff"),
        ("PHI Scrubbing\n& De-ID", 0.15, 0.62, "#f0883e"),
        ("Feature\nExtraction", 0.5, 0.62, "#8957e5"),
        ("Training\nDataset v2.1", 0.5, 0.38, "#238636"),
        ("Trained\nModel v3.0", 0.85, 0.38, "#e3b341"),
        ("Validation\nResults", 0.85, 0.15, "#238636"),
        ("Normalization\nParams", 0.15, 0.38, "#6e7681"),
        ("Production\nEndpoint", 0.5, 0.15, "#8957e5"),
    ]

    for label, x, y, color in nodes:
        box = FancyBboxPatch(
            (x - 0.1, y - 0.08), 0.2, 0.16,
            boxstyle="round,pad=0.02",
            facecolor="#0d1117",
            edgecolor=color,
            linewidth=1.5,
            transform=ax_lineage.transAxes,
        )
        ax_lineage.add_patch(box)
        ax_lineage.text(x, y, label, fontsize=6.5, color="white",
                       ha="center", va="center", transform=ax_lineage.transAxes,
                       fontfamily="monospace")

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (1, 6), (3, 7), (4, 7),
    ]
    for src, dst in edges:
        sx, sy = nodes[src][1], nodes[src][2] - 0.08
        dx, dy = nodes[dst][1], nodes[dst][2] + 0.08
        ax_lineage.annotate("", xy=(dx, dy), xytext=(sx, sy),
                          xycoords="axes fraction", textcoords="axes fraction",
                          arrowprops=dict(arrowstyle="->", color="#30363d", lw=1.5))

    # Version history
    ax_versions = fig.add_subplot(gs[2, 0])
    ax_versions.set_facecolor("#161b22")
    ax_versions.axis("off")
    ax_versions.set_title("Artifact Version History", fontsize=11, fontweight="bold",
                         color="white", pad=10, fontfamily="monospace")

    versions = [
        ("v3.0.0", "2026-06-01", "a3f8c2d9", "DEPLOYED", "#238636"),
        ("v2.1.0", "2026-03-15", "7b2e4f8a", "ARCHIVED", "#6e7681"),
        ("v2.0.0", "2025-11-20", "c9d1e3f5", "ARCHIVED", "#6e7681"),
        ("v1.2.0", "2025-08-10", "4a6b8c0d", "ARCHIVED", "#6e7681"),
        ("v1.0.0", "2025-04-01", "e2f4a6b8", "SUPERSEDED", "#6e7681"),
    ]

    headers = ["Version", "Date", "Hash", "Status"]
    hx = [0.05, 0.25, 0.50, 0.78]
    for i, h in enumerate(headers):
        ax_versions.text(hx[i], 0.88, h, fontsize=8, fontweight="bold",
                        color="#58a6ff", va="top", transform=ax_versions.transAxes,
                        fontfamily="monospace")

    for j, (ver, date, hash_val, status, color) in enumerate(versions):
        y = 0.75 - j * 0.14
        ax_versions.text(hx[0], y, ver, fontsize=8.5, fontweight="bold",
                        color=color, va="center", transform=ax_versions.transAxes,
                        fontfamily="monospace")
        ax_versions.text(hx[1], y, date, fontsize=8, color="#8b949e",
                        va="center", transform=ax_versions.transAxes,
                        fontfamily="monospace")
        ax_versions.text(hx[2], y, f"sha256:{hash_val}...", fontsize=7.5,
                        color="#6e7681", va="center", transform=ax_versions.transAxes,
                        fontfamily="monospace")
        ax_versions.text(hx[3], y, status, fontsize=8, fontweight="bold",
                        color=color, va="center", transform=ax_versions.transAxes,
                        fontfamily="monospace")

    for spine in ax_versions.spines.values():
        spine.set_edgecolor("#30363d")

    # Compliance checklist
    ax_checklist = fig.add_subplot(gs[2, 1])
    ax_checklist.set_facecolor("#161b22")
    ax_checklist.axis("off")
    ax_checklist.set_title("Regulatory Compliance Checklist", fontsize=11,
                          fontweight="bold", color="white", pad=10,
                          fontfamily="monospace")

    checks = [
        ("21 CFR 820.30 Design Controls", True),
        ("21 CFR 820.40 Document Controls", True),
        ("21 CFR 820.70 Process Controls", True),
        ("21 CFR 820.184 Device History Record", True),
        ("21 CFR Part 11 Electronic Records", True),
        ("IEC 62304 Software Lifecycle", True),
        ("ISO 14971 Risk Management", True),
        ("HIPAA PHI Compliance", True),
        ("GMLP Principles (1-10)", True),
        ("Hash Chain Integrity", True),
    ]

    for i, (check_name, passed) in enumerate(checks):
        y = 0.88 - i * 0.085
        marker = "[X]" if passed else "[ ]"
        color = "#238636" if passed else "#da3633"

        ax_checklist.text(0.05, y, marker, fontsize=9, fontweight="bold",
                         color=color, va="center", transform=ax_checklist.transAxes,
                         fontfamily="monospace")
        ax_checklist.text(0.15, y, check_name, fontsize=8, color="white",
                         va="center", transform=ax_checklist.transAxes,
                         fontfamily="monospace")

    # Chain integrity status
    ax_checklist.text(0.5, 0.02, "Chain Integrity: VERIFIED (256 events, 0 violations)",
                     fontsize=7.5, color="#238636", ha="center", va="center",
                     transform=ax_checklist.transAxes, fontfamily="monospace",
                     fontweight="bold")

    for spine in ax_checklist.spines.values():
        spine.set_edgecolor("#30363d")

    output_path = OUTPUT_DIR / "audit_trail.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main():
    """Generate all screenshot images."""
    print("Generating pipeline visualization screenshots...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    generate_pipeline_overview()
    generate_validation_report()
    generate_audit_trail()

    print("\nAll screenshots generated successfully.")


if __name__ == "__main__":
    main()
