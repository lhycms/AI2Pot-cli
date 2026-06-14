"""Descriptor projection plot: PCA of atomic descriptors for train/test sets."""

import os
from typing import Optional, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from ai2pot_cli.menu import print_section, print_kv, print_sep, print_error, print_success

# Paper-ready style (consistent with plot_parity.py)
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _get_symbol(z: int) -> str:
    """Atomic number → symbol via ase."""
    from ase.data import chemical_symbols
    if 0 < z < len(chemical_symbols):
        return chemical_symbols[z]
    return f"Z{z}"


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
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(checkpoint_path: str, dataset_path: str, device: str):
    model_type = _detect_model_type(checkpoint_path)
    if model_type == "mtp":
        from ai2pot.models.mtp.linear_mtp_utils import LinearMtp4Extxyz
        return LinearMtp4Extxyz(checkpoint_path=checkpoint_path, testset_path=dataset_path, map_location=device)
    else:
        from ai2pot.models.nep.nep_utils import Nep4Extxyz
        return Nep4Extxyz(checkpoint_path=checkpoint_path, testset_path=dataset_path, map_location=device)


def _compute_descriptors(model) -> Tuple[np.ndarray, np.ndarray]:
    """Return (descriptors_NxD, atomic_numbers_N) from a model."""
    return model.calculate_descriptors()


def _make_projection_plot(
    train_desc: Optional[np.ndarray],
    train_z: Optional[np.ndarray],
    test_desc: Optional[np.ndarray],
    test_z: Optional[np.ndarray],
    output_path: str,
):
    """PCA projection of descriptors.

    Colour encodes the element, marker shape encodes train (circle) vs test (square).
    PCA is fitted on the training set (if available) and applied to both sets.
    """
    # --- gather unique elements ---
    all_z: List[int] = []
    if train_z is not None:
        all_z.extend(np.unique(train_z).tolist())
    if test_z is not None:
        all_z.extend(np.unique(test_z).tolist())
    unique_z = sorted(set(all_z))

    # --- high-distinction palette ---
    _DISTINCT_COLORS = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#000000",  # black
        "#999999",  # grey
        "#882255",
        "#44AA99",
        "#AA4499",
    ]
    z_to_color = {
        z: _DISTINCT_COLORS[i % len(_DISTINCT_COLORS)]
        for i, z in enumerate(unique_z)
    }

    # --- PCA: fit on train only if available, else fit on test ---
    pca = PCA(n_components=2)
    if train_desc is not None:
        pca.fit(train_desc)
    elif test_desc is not None:
        pca.fit(test_desc)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 6.5))

    marker_size = 8

    for z in unique_z:
        base = z_to_color[z]
        symbol = _get_symbol(z)

        # Train
        if train_desc is not None and train_z is not None:
            mask = train_z == z
            if mask.any():
                proj = pca.transform(train_desc[mask])
                ax.scatter(proj[:, 0], proj[:, 1], s=marker_size, alpha=0.65,
                           c=base, marker="o", edgecolors="none",
                           label=f"{symbol} (Train)")

        # Test
        if test_desc is not None and test_z is not None:
            mask = test_z == z
            if mask.any():
                proj = pca.transform(test_desc[mask])
                ax.scatter(proj[:, 0], proj[:, 1], s=marker_size * 1.5, alpha=0.85,
                           c=base, marker="s", edgecolors="black", linewidths=0.3,
                           label=f"{symbol} (Test)")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    n_elements = len(unique_z)
    ncol = 2 if n_elements > 5 else 1
    ax.legend(loc="best", framealpha=0.35, fontsize=13, ncol=ncol,
              handletextpad=0.3, columnspacing=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_descriptor_projection(
    checkpoint_path: str,
    trainset_path: Optional[str] = None,
    testset_path: Optional[str] = None,
    output_path: Optional[str] = None,
):
    if not trainset_path and not testset_path:
        raise ValueError("At least one of trainset_path or testset_path must be provided.")

    if output_path is None:
        out_dir = os.path.join(os.getcwd(), "descriptor_analysis")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "descriptor_projection.png")
    else:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
    abs_output = os.path.abspath(output_path)

    device = _get_device()
    print_section(f"Running on {device.upper()}")

    # --- Compute descriptors ---
    train_desc: Optional[np.ndarray] = None
    train_z: Optional[np.ndarray] = None
    test_desc: Optional[np.ndarray] = None
    test_z: Optional[np.ndarray] = None

    if trainset_path:
        print_success(f"Loading trainset: {trainset_path}")
        model = _build_model(checkpoint_path, trainset_path, device)
        train_desc, train_z = _compute_descriptors(model)
        print_kv("Train atoms", f"{train_desc.shape[0]:,}")
        unique_elements = sorted(set(np.unique(train_z).tolist()))
        element_str = ", ".join(_get_symbol(z) for z in unique_elements)
        print_kv("Train elements", element_str)

    if testset_path:
        print_success(f"Loading testset: {testset_path}")
        model = _build_model(checkpoint_path, testset_path, device)
        test_desc, test_z = _compute_descriptors(model)
        print_kv("Test atoms", f"{test_desc.shape[0]:,}")
        unique_elements = sorted(set(np.unique(test_z).tolist()))
        element_str = ", ".join(_get_symbol(z) for z in unique_elements)
        print_kv("Test elements", element_str)

    # --- Plot ---
    _make_projection_plot(train_desc, train_z, test_desc, test_z, abs_output)

    # --- Print results ---
    print_section("Descriptor Projection Generated Successfully")
    print_kv("Output Plot", abs_output)
    print_sep()
    print()
