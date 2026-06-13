#!/usr/bin/env python3
"""Plot training learning curves from CSVLogger metrics.csv."""

from __future__ import annotations

import csv
import os
from typing import Optional, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai2pot_cli.menu import print_section, print_kv, print_sep, print_error, print_warning

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

METRIC_LABELS = {"e": "Energy", "f": "Force", "v": "Virial"}
SET_STYLES = {
    "train": {"color": "#2166ac", "label": "Train", "lw": 1.2},
    "val": {"color": "#d73027", "label": "Val", "lw": 1.2},
}


def _parse_metrics(csv_path: str, x_axis: str) -> Dict:
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {}

    all_cols = list(rows[0].keys())
    data = {}

    for prefix in ("train", "val"):
        for metric in ("e_rmse", "f_rmse", "v_rmse"):
            col = f"{prefix}/{metric}_{x_axis}"
            if col not in all_cols:
                continue
            x_vals = []
            y_vals = []
            for row in rows:
                x_str = row.get(x_axis, "").strip()
                y_str = row.get(col, "").strip()
                if x_str and y_str:
                    try:
                        x_vals.append(float(x_str))
                        y_vals.append(float(y_str))
                    except ValueError:
                        continue

            if x_vals:
                short = metric[0]  # e, f, v
                key = f"{prefix}_{short}"
                data[key] = {"x": np.array(x_vals), "y": np.array(y_vals)}

    return data


def plot_trainlog(
    csv_path: str, x_axis: str = "epoch", output_path: Optional[str] = None
):
    if not os.path.exists(csv_path):
        print_error(f"File not found: {csv_path}")
        return

    data = _parse_metrics(csv_path, x_axis)

    if not data:
        print_warning("No valid metrics found in the CSV file.")
        return

    if output_path is None:
        out_dir = os.path.join(
            os.path.dirname(csv_path) or ".", "trainlog_analysis"
        )
    else:
        out_dir = output_path
    os.makedirs(out_dir, exist_ok=True)

    # --- Determine available quantities ---
    available = []
    for short in ("e", "f", "v"):
        for prefix in ("train", "val"):
            key = f"{prefix}_{short}"
            if key in data:
                available.append(short)
                break
    available = list(dict.fromkeys(available))  # deduplicate, keep order

    if not available:
        print_warning("No e_rmse / f_rmse / v_rmse metrics found in the CSV.")
        return

    # --- Plot ---
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for idx, short in enumerate(available):
        ax = axes[0, idx]
        for prefix, style in SET_STYLES.items():
            key = f"{prefix}_{short}"
            if key in data:
                d = data[key]
                ax.plot(
                    d["x"],
                    d["y"],
                    color=style["color"],
                    lw=style["lw"],
                    alpha=0.85,
                    label=style["label"],
                )
        ax.set_xlabel(x_axis.capitalize())
        ax.set_ylabel(f"{METRIC_LABELS[short]} RMSE")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    out_file = os.path.join(out_dir, f"learning_curve_{x_axis}.png")
    fig.savefig(out_file)
    plt.close(fig)

    # --- Print results ---
    print_section("Learning Curve Generated Successfully")
    print_kv("Output Dir", out_dir)
    print_sep()
    print()
