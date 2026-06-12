"""Parity plot: prediction vs. reference for E, F, V."""
import os
from typing import Optional, List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ai2pot_cli.menu import print_section, print_kv, print_sep, print_error, print_success

# Paper-ready style
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

DATASET_STYLES = {
    "Train": {"color": "#2166ac", "marker": "o", "alpha": 0.5, "s": 12},
    "Test": {"color": "#d73027", "marker": "s", "alpha": 0.6, "s": 14},
}


def _detect_model_type(checkpoint_path: str) -> str:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    if "mtp_level" in hp:
        return "mtp"
    if "n_radial_basis" in hp:
        return "nep"
    raise ValueError(
        "Cannot detect model type from checkpoint. "
        "Expected 'mtp_level' or 'n_radial_basis' in hyper_parameters."
    )


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_model(checkpoint_path: str, dataset_path: str, device: str):
    model_type = _detect_model_type(checkpoint_path)
    if model_type == "mtp":
        from ai2pot.models.mtp.linear_mtp_utils import LinearMtp4Extxyz
        return LinearMtp4Extxyz(checkpoint_path=checkpoint_path, testset_path=dataset_path, map_location=device)
    else:
        from ai2pot.models.nep.nep_utils import Nep4Extxyz
        return Nep4Extxyz(checkpoint_path=checkpoint_path, testset_path=dataset_path, map_location=device)


def _compute_parity(model) -> Dict:
    has_virial = model.fit_virial
    if has_virial:
        e_dft, f_dft, v_dft, e_ml, f_ml, v_ml = model.calculate_parity()
        return {"e_dft": e_dft, "e_ml": e_ml, "f_dft": f_dft, "f_ml": f_ml, "v_dft": v_dft, "v_ml": v_ml}
    else:
        e_dft, f_dft, e_ml, f_ml = model.calculate_parity()
        return {"e_dft": e_dft, "e_ml": e_ml, "f_dft": f_dft, "f_ml": f_ml}


def _expand_lim(lo, hi, pct=0.20):
    span = hi - lo
    if span == 0:
        span = abs(hi) if hi != 0 else 1.0
    return lo - span * pct, hi + span * pct


def _make_parity_plot(datasets: List[Dict], output_path: str):
    """Generate paper-ready E/F/V parity scatter plots.

    datasets: list of {"label": str, "e_dft": ..., "e_ml": ..., "f_dft": ..., "f_ml": ..., ("v_dft": ..., "v_ml": ...)}
    """
    has_virial = any("v_dft" in d for d in datasets)
    ncols = 3 if has_virial else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5.8), squeeze=False)
    axes = axes[0]

    # --- Energy ---
    ax = axes[0]
    for ds in datasets:
        style = DATASET_STYLES.get(ds["label"], DATASET_STYLES["Test"])
        ax.scatter(ds["e_dft"], ds["e_ml"], s=style["s"], alpha=style["alpha"],
                   c=style["color"], marker=style["marker"], edgecolors="none", label=ds["label"])
    all_e = np.concatenate([np.concatenate([d["e_dft"], d["e_ml"]]) for d in datasets])
    elo, ehi = _expand_lim(all_e.min(), all_e.max())
    ax.plot([elo, ehi], [elo, ehi], "k--", linewidth=1.2)
    ax.set_xlim(elo, ehi)
    ax.set_ylim(elo, ehi)
    ax.set_xlabel("DFT Energy (eV/atom)")
    ax.set_ylabel("ML Energy (eV/atom)")
    ax.set_title("Energy Parity")
    ax.legend(loc="lower right", framealpha=0.8)
    # RMSE annotation per dataset
    rmse_lines = []
    for ds in datasets:
        e_rmse = np.sqrt(np.mean((ds["e_ml"] - ds["e_dft"]) ** 2))
        rmse_lines.append(f"{ds['label']} RMSE = {e_rmse * 1000:.2f} meV/atom")
    ax.text(0.05, 0.92, "\n".join(rmse_lines), transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    # --- Force ---
    ax = axes[1]
    for ds in datasets:
        style = DATASET_STYLES.get(ds["label"], DATASET_STYLES["Test"])
        ax.scatter(ds["f_dft"], ds["f_ml"], s=style["s"] * 0.3, alpha=style["alpha"] * 0.7,
                   c=style["color"], marker=style["marker"], edgecolors="none", label=ds["label"])
    all_f = np.concatenate([np.concatenate([d["f_dft"], d["f_ml"]]) for d in datasets])
    flo, fhi = _expand_lim(all_f.min(), all_f.max())
    ax.plot([flo, fhi], [flo, fhi], "k--", linewidth=1.2)
    ax.set_xlim(flo, fhi)
    ax.set_ylim(flo, fhi)
    ax.set_xlabel("DFT Force (eV/A)")
    ax.set_ylabel("ML Force (eV/A)")
    ax.set_title("Force Parity")
    ax.legend(loc="lower right", framealpha=0.8)
    rmse_lines = []
    for ds in datasets:
        f_rmse = np.sqrt(np.mean((ds["f_ml"] - ds["f_dft"]) ** 2))
        rmse_lines.append(f"{ds['label']} RMSE = {f_rmse * 1000:.2f} meV/A")
    ax.text(0.05, 0.92, "\n".join(rmse_lines), transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    # --- Virial (optional) ---
    if has_virial:
        ax = axes[2]
        for ds in datasets:
            if "v_dft" not in ds:
                continue
            style = DATASET_STYLES.get(ds["label"], DATASET_STYLES["Test"])
            ax.scatter(ds["v_dft"], ds["v_ml"], s=style["s"], alpha=style["alpha"],
                       c=style["color"], marker=style["marker"], edgecolors="none", label=ds["label"])
        all_v = np.concatenate([np.concatenate([d["v_dft"], d["v_ml"]]) for d in datasets if "v_dft" in d])
        vlo, vhi = _expand_lim(all_v.min(), all_v.max())
        ax.plot([vlo, vhi], [vlo, vhi], "k--", linewidth=1.2)
        ax.set_xlim(vlo, vhi)
        ax.set_ylim(vlo, vhi)
        ax.set_xlabel("DFT Virial (eV/atom)")
        ax.set_ylabel("ML Virial (eV/atom)")
        ax.set_title("Virial Parity")
        ax.legend(loc="lower right", framealpha=0.8)
        rmse_lines = []
        for ds in datasets:
            if "v_dft" not in ds:
                continue
            v_rmse = np.sqrt(np.mean((ds["v_ml"] - ds["v_dft"]) ** 2))
            rmse_lines.append(f"{ds['label']} RMSE = {v_rmse * 1000:.2f} meV/atom")
        ax.text(0.05, 0.92, "\n".join(rmse_lines), transform=ax.transAxes, fontsize=12,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_npy(data: Dict, prefix: str, out_dir: str):
    np.save(os.path.join(out_dir, f"{prefix}energy_dft.npy"), data["e_dft"])
    np.save(os.path.join(out_dir, f"{prefix}energy_pred.npy"), data["e_ml"])
    np.save(os.path.join(out_dir, f"{prefix}force_dft.npy"), data["f_dft"])
    np.save(os.path.join(out_dir, f"{prefix}force_pred.npy"), data["f_ml"])
    if "v_dft" in data:
        np.save(os.path.join(out_dir, f"{prefix}virial_dft.npy"), data["v_dft"])
        np.save(os.path.join(out_dir, f"{prefix}virial_pred.npy"), data["v_ml"])


def plot_parity(
    checkpoint_path: str,
    trainset_path: Optional[str] = None,
    testset_path: Optional[str] = None,
    output_path: Optional[str] = None,
):
    if not trainset_path and not testset_path:
        raise ValueError("At least one of trainset_path or testset_path must be provided.")

    if output_path is None:
        output_path = os.path.join(os.getcwd(), "parity_plot.png")
    abs_output = os.path.abspath(output_path)
    out_dir = os.path.dirname(abs_output)

    # --- Device ---
    device = _get_device()
    print_section(f"Running on {device.upper()}")

    # --- Compute parity for each dataset ---
    datasets: List[Dict] = []
    npy_files: List[str] = []

    if trainset_path:
        print_success(f"Loading trainset: {trainset_path}")
        model = _build_model(checkpoint_path, trainset_path, device)
        data = _compute_parity(model)
        data["label"] = "Train"
        datasets.append(data)
        _save_npy(data, "train_", out_dir)
        npy_files.extend([
            os.path.join(out_dir, "train_energy_dft.npy"),
            os.path.join(out_dir, "train_energy_pred.npy"),
            os.path.join(out_dir, "train_force_dft.npy"),
            os.path.join(out_dir, "train_force_pred.npy"),
        ])
        if "v_dft" in data:
            npy_files.extend([
                os.path.join(out_dir, "train_virial_dft.npy"),
                os.path.join(out_dir, "train_virial_pred.npy"),
            ])

    if testset_path:
        print_success(f"Loading testset: {testset_path}")
        model = _build_model(checkpoint_path, testset_path, device)
        data = _compute_parity(model)
        data["label"] = "Test"
        datasets.append(data)
        _save_npy(data, "test_", out_dir)
        npy_files.extend([
            os.path.join(out_dir, "test_energy_dft.npy"),
            os.path.join(out_dir, "test_energy_pred.npy"),
            os.path.join(out_dir, "test_force_dft.npy"),
            os.path.join(out_dir, "test_force_pred.npy"),
        ])
        if "v_dft" in data:
            npy_files.extend([
                os.path.join(out_dir, "test_virial_dft.npy"),
                os.path.join(out_dir, "test_virial_pred.npy"),
            ])

    # --- Plot ---
    _make_parity_plot(datasets, abs_output)

    # --- Print results ---
    print_section("Parity Plot Generated Successfully")
    print_kv("Output Plot", abs_output)
    print()
    for f in npy_files:
        print_kv("Output Data", f)
    print()
    for ds in datasets:
        e_rmse = np.sqrt(np.mean((ds["e_ml"] - ds["e_dft"]) ** 2))
        f_rmse = np.sqrt(np.mean((ds["f_ml"] - ds["f_dft"]) ** 2))
        label = ds["label"]
        print_kv(f"RMSE ({label} Energy)", f"{e_rmse * 1000:.2f} meV/atom")
        print_kv(f"RMSE ({label} Force)", f"{f_rmse * 1000:.2f} meV/A")
        if "v_dft" in ds:
            v_rmse = np.sqrt(np.mean((ds["v_ml"] - ds["v_dft"]) ** 2))
            print_kv(f"RMSE ({label} Virial)", f"{v_rmse * 1000:.2f} meV/atom")
    print_sep()
    print()
