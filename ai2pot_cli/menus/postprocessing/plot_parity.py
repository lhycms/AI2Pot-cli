"""Parity plot: prediction vs. reference for E, F, V."""
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ai2pot_cli.menu import print_section, print_kv, print_sep, print_error


def _detect_model_type(checkpoint_path: str) -> str:
    """Detect MTP vs NEP from checkpoint hyperparameters."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    if "mtp_level" in hp:
        return "mtp"
    if "n_radial_basis" in hp:
        return "nep"
    raise ValueError("Cannot detect model type from checkpoint. Expected 'mtp_level' or 'n_radial_basis' in hyper_parameters.")


def _make_parity_plot(e_dft, e_ml, f_dft, f_ml, v_dft=None, v_ml=None, output_path="parity_plot.png"):
    """Generate and save the E/F/V parity scatter plots."""
    has_virial = v_dft is not None and v_ml is not None
    ncols = 3 if has_virial else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.5), squeeze=False)
    axes = axes[0]

    # --- Energy ---
    ax = axes[0]
    ax.scatter(e_dft, e_ml, s=8, alpha=0.6, c="#2c3e50", edgecolors="none")
    lim_e = [min(e_dft.min(), e_ml.min()), max(e_dft.max(), e_ml.max())]
    ax.plot(lim_e, lim_e, "r--", linewidth=1)
    ax.set_xlabel("DFT Energy (eV/atom)")
    ax.set_ylabel("ML Energy (eV/atom)")
    ax.set_title("Energy Parity")
    e_rmse = np.sqrt(np.mean((e_ml - e_dft) ** 2))
    ax.text(0.05, 0.92, f"RMSE = {e_rmse * 1000:.2f} meV/atom", transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # --- Force ---
    ax = axes[1]
    ax.scatter(f_dft, f_ml, s=3, alpha=0.4, c="#2c3e50", edgecolors="none")
    lim_f = [min(f_dft.min(), f_ml.min()), max(f_dft.max(), f_ml.max())]
    ax.plot(lim_f, lim_f, "r--", linewidth=1)
    ax.set_xlabel("DFT Force (eV/A)")
    ax.set_ylabel("ML Force (eV/A)")
    ax.set_title("Force Parity")
    f_rmse = np.sqrt(np.mean((f_ml - f_dft) ** 2))
    ax.text(0.05, 0.92, f"RMSE = {f_rmse * 1000:.2f} meV/A", transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # --- Virial (optional) ---
    if has_virial:
        ax = axes[2]
        ax.scatter(v_dft, v_ml, s=8, alpha=0.6, c="#2c3e50", edgecolors="none")
        lim_v = [min(v_dft.min(), v_ml.min()), max(v_dft.max(), v_ml.max())]
        ax.plot(lim_v, lim_v, "r--", linewidth=1)
        ax.set_xlabel("DFT Virial (eV/atom)")
        ax.set_ylabel("ML Virial (eV/atom)")
        ax.set_title("Virial Parity")
        v_rmse = np.sqrt(np.mean((v_ml - v_dft) ** 2))
        ax.text(0.05, 0.92, f"RMSE = {v_rmse * 1000:.2f} meV/atom", transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return e_rmse, f_rmse, (v_rmse if has_virial else None)


def plot_parity(checkpoint_path: str, testset_path: str, output_path: Optional[str] = None):
    """Run parity calculation and generate E/F/V scatter plots.

    Args:
        checkpoint_path: Path to the .ckpt file.
        testset_path: Path to the extxyz test set.
        output_path: Path for the output PNG (default: parity_plot.png).
    """
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "parity_plot.png")
    abs_output = os.path.abspath(output_path)

    # --- Device selection ---
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"  Running on: {device.upper()}")

    model_type = _detect_model_type(checkpoint_path)

    if model_type == "mtp":
        from ai2pot.models.mtp.linear_mtp_utils import LinearMtp4Extxyz
        model = LinearMtp4Extxyz(checkpoint_path=checkpoint_path, testset_path=testset_path, map_location=device)
    else:
        from ai2pot.models.nep.nep_utils import Nep4Extxyz
        model = Nep4Extxyz(checkpoint_path=checkpoint_path, testset_path=testset_path, map_location=device)

    has_virial = model.fit_virial

    if has_virial:
        e_dft, f_dft, v_dft, e_ml, f_ml, v_ml = model.calculate_parity()
    else:
        e_dft, f_dft, e_ml, f_ml = model.calculate_parity()
        v_dft, v_ml = None, None

    e_rmse, f_rmse, v_rmse = _make_parity_plot(
        e_dft, e_ml, f_dft, f_ml, v_dft, v_ml, abs_output,
    )

    # --- Save raw data as .npy ---
    out_dir = os.path.dirname(abs_output)
    np.save(os.path.join(out_dir, "energy_dft.npy"), e_dft)
    np.save(os.path.join(out_dir, "energy_pred.npy"), e_ml)
    np.save(os.path.join(out_dir, "force_dft.npy"), f_dft)
    np.save(os.path.join(out_dir, "force_pred.npy"), f_ml)
    if has_virial:
        np.save(os.path.join(out_dir, "virial_dft.npy"), v_dft)
        np.save(os.path.join(out_dir, "virial_pred.npy"), v_ml)

    # --- Print results ---
    print_section("Parity Plot Generated Successfully")
    print_kv("Output Plot", abs_output)
    print_kv("Output Data", os.path.join(out_dir, "energy_dft.npy, energy_pred.npy, ..."))
    print()
    print_kv("RMSE (Energy)", f"{e_rmse * 1000:.2f} meV/atom")
    print_kv("RMSE (Force)", f"{f_rmse * 1000:.2f} meV/A")
    if v_rmse is not None:
        print_kv("RMSE (Virial)", f"{v_rmse * 1000:.2f} meV/atom")
    print_sep()
    print()
