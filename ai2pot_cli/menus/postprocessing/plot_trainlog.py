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

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

METRIC_STYLES = {
    "e": {"color": "#2166ac", "label": "Energy", "lw": 1.5},
    "f": {"color": "#d73027", "label": "Force", "lw": 1.5},
    "v": {"color": "#4daf4a", "label": "Virial", "lw": 1.5},
}
SET_LINESTYLE = {"train": "-", "val": "--"}


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


def _make_plot(data: Dict, x_axis: str, available: list, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    for short in available:
        mstyle = METRIC_STYLES[short]
        for prefix, ls in SET_LINESTYLE.items():
            key = f"{prefix}_{short}"
            if key not in data:
                continue
            d = data[key]
            label = f"{mstyle['label']} ({prefix.capitalize()})"
            ax.plot(
                d["x"],
                d["y"],
                color=mstyle["color"],
                lw=mstyle["lw"],
                ls=ls,
                alpha=0.85,
                label=label,
            )

    ax.set_xlabel(x_axis.capitalize())
    ax.set_ylabel("RMSE")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_file = os.path.join(out_dir, f"learning_curve_{x_axis}.png")
    fig.savefig(out_file)
    plt.close(fig)


def plot_trainlog(csv_path: str, output_path: Optional[str] = None):
    if not os.path.exists(csv_path):
        print_error(f"File not found: {csv_path}")
        return

    if output_path is None:
        out_dir = os.path.join(os.getcwd(), "trainlog_analysis")
    else:
        out_dir = output_path
    os.makedirs(out_dir, exist_ok=True)

    generated = []

    for x_axis in ("epoch", "step"):
        data = _parse_metrics(csv_path, x_axis)
        if not data:
            continue

        available = []
        for short in ("e", "f", "v"):
            for prefix in ("train", "val"):
                if f"{prefix}_{short}" in data:
                    available.append(short)
                    break
        available = list(dict.fromkeys(available))

        if not available:
            continue

        _make_plot(data, x_axis, available, out_dir)
        generated.append(x_axis)

    if not generated:
        print_warning("No valid metrics found in the CSV file.")
        return

    print_section("Learning Curve Generated Successfully")
    print_kv("Output Dir", out_dir)
    print_sep()
    print()
