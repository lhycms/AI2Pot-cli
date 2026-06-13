#!/usr/bin/env python3
"""Plot training curves from CSVLogger metrics.csv."""

from __future__ import annotations

import csv
import os
from typing import Optional, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai2pot_cli.menu import print_section, print_kv, print_sep, print_error, print_warning

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
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

GRAD_STYLES = {
    "avg_norm":       {"color": "#2166ac", "label": "Avg Norm",       "lw": 1.5},
    "raw_norm":       {"color": "#d73027", "label": "Raw Norm",       "lw": 1.5},
    "clip_threshold": {"color": "#4daf4a", "label": "Clip Threshold", "lw": 1.5},
}


def _read_csv(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _extract_series(rows: List[Dict], x_axis: str, y_col: str):
    """Extract x, y arrays from CSV rows for a given column."""
    x_vals, y_vals = [], []
    for row in rows:
        x_str = row.get(x_axis, "").strip()
        y_str = row.get(y_col, "").strip()
        if x_str and y_str:
            try:
                xv = float(x_str)
                yv = float(y_str)
            except ValueError:
                continue
            x_vals.append(xv)
            y_vals.append(yv)
    return np.array(x_vals), np.array(y_vals)


def _make_rmse_plot(rows: List[Dict], x_axis: str, out_dir: str):
    """Energy / Force / Virial RMSE on a single log-scale plot."""
    available = []
    for short in ("e", "f", "v"):
        col = f"train/{short}_rmse_{x_axis}"
        found = False
        for row in rows:
            if row.get(col, "").strip():
                found = True
                break
        if found:
            available.append(short)

    if not available:
        return False

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    for short in available:
        mstyle = METRIC_STYLES[short]
        for prefix, ls in SET_LINESTYLE.items():
            col = f"{prefix}/{short}_rmse_{x_axis}"
            x, y = _extract_series(rows, x_axis, col)
            if len(x) == 0:
                continue
            label = f"{mstyle['label']} ({prefix.capitalize()})"
            ax.plot(x, y, color=mstyle["color"], lw=mstyle["lw"], ls=ls,
                    alpha=0.85, label=label)

    ax.set_xlabel(x_axis.capitalize())
    ax.set_ylabel("RMSE")
    if x_axis == "step":
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, f"learning_curve_{x_axis}.png"))
    plt.close(fig)
    return True


def _make_weight_plot(rows: List[Dict], x_axis: str, out_dir: str):
    """Energy / Force / Virial loss weights on a single plot."""
    available = []
    for short in ("e", "f", "v"):
        col = f"train/{short}_wgt"
        for row in rows:
            if row.get(col, "").strip():
                available.append(short)
                break

    if not available:
        return False

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    for short in available:
        mstyle = METRIC_STYLES[short]
        col = f"train/{short}_wgt"
        x, y = _extract_series(rows, x_axis, col)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=mstyle["color"], lw=mstyle["lw"],
                alpha=0.85, label=mstyle["label"])

    ax.set_xlabel(x_axis.capitalize())
    ax.set_ylabel("Loss Weight")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, f"weight_curve_{x_axis}.png"))
    plt.close(fig)
    return True


def _make_lr_plot(rows: List[Dict], x_axis: str, out_dir: str):
    """Learning rate on a log-scale plot."""
    col = "train/lr"
    x, y = _extract_series(rows, x_axis, col)
    if len(x) == 0:
        return False

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.plot(x, y, color="#333333", lw=1.5, alpha=0.85)
    ax.set_xlabel(x_axis.capitalize())
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, f"lr_curve_{x_axis}.png"))
    plt.close(fig)
    return True


def _make_grad_plot(rows: List[Dict], x_axis: str, out_dir: str):
    """Gradient avg_norm / raw_norm / clip_threshold on a single log-scale plot."""
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    any_line = False

    for short, gstyle in GRAD_STYLES.items():
        col = f"grad/{short}_{x_axis}"
        x, y = _extract_series(rows, x_axis, col)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=gstyle["color"], lw=gstyle["lw"],
                alpha=0.85, label=gstyle["label"])
        any_line = True

    if not any_line:
        plt.close(fig)
        return False

    ax.set_xlabel(x_axis.capitalize())
    ax.set_ylabel("Gradient Norm")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, f"grad_curve_{x_axis}.png"))
    plt.close(fig)
    return True


def plot_trainlog(csv_path: str, output_path: Optional[str] = None):
    if not os.path.exists(csv_path):
        print_error(f"File not found: {csv_path}")
        return

    rows = _read_csv(csv_path)
    if not rows:
        print_warning("CSV file is empty.")
        return

    if output_path is None:
        out_dir = os.path.join(os.getcwd(), "trainlog_analysis")
    else:
        out_dir = output_path
    os.makedirs(out_dir, exist_ok=True)

    all_generated = []

    for x_axis in ("epoch", "step"):
        if _make_rmse_plot(rows, x_axis, out_dir):
            all_generated.append(x_axis)

    if _make_weight_plot(rows, "step", out_dir):
        all_generated.append("weight")
    if _make_lr_plot(rows, "step", out_dir):
        all_generated.append("lr")
    if _make_grad_plot(rows, "step", out_dir):
        all_generated.append("grad")

    if not all_generated:
        print_warning("No valid metrics found in the CSV file.")
        return

    print_section("Training Curves Generated Successfully")
    print_kv("Output Dir", out_dir)
    print_sep()
    print()
