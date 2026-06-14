"""Descriptor projection plot: PCA of atomic descriptors for train/test sets."""

import os
from typing import Optional, List, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# Distinguishable base colours for elements (colourblind-friendly palette, extended)
_ELEMENT_BASE = {
    1:  "#1f77b4",   # H  - blue
    3:  "#ff7f0e",   # Li - orange
    5:  "#2ca02c",   # B  - green
    6:  "#d62728",   # C  - red
    7:  "#9467bd",   # N  - purple
    8:  "#8c564b",   # O  - brown
    9:  "#e377c2",   # F  - pink
    11: "#7f7f7f",   # Na - grey
    13: "#bcbd22",   # Al - olive
    14: "#17becf",   # Si - cyan
    15: "#aec7e8",   # P  - lt blue
    16: "#ffbb78",   # S  - lt orange
    17: "#98df8a",   # Cl - lt green
    19: "#ff9896",   # K  - lt red
    20: "#c5b0d5",   # Ca - lt purple
    22: "#c49c94",   # Ti - lt brown
    25: "#f7b6d2",   # Mn - lt pink
    26: "#dbdb8d",   # Fe - lt olive
    28: "#9edae5",   # Ni - lt cyan
    29: "#393b79",   # Cu - dk blue
    30: "#637939",   # Zn - dk green
    40: "#8c6d31",   # Zr - dk brown
    56: "#843c39",   # Ba - dk red
    74: "#7b4173",   # W  - dk purple
}

_FALLBACK = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#393b79",
]


def _get_symbol(z: int) -> str:
    """Atomic number → symbol via ase."""
    from ase.data import chemical_symbols
    if 0 < z < len(chemical_symbols):
        return chemical_symbols[z]
    return f"Z{z}"


def _lighten(hex_color: str, factor: float = 0.35) -> str:
    """Blend hex colour toward white."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return mcolors.to_hex(rgb + (1.0 - rgb) * factor)


def _darken(hex_color: str, factor: float = 0.35) -> str:
    """Blend hex colour toward black."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return mcolors.to_hex(rgb * (1.0 - factor))


def _get_element_color(z: int) -> str:
    """Return a stable base colour for an atomic number."""
    if z in _ELEMENT_BASE:
        return _ELEMENT_BASE[z]
    idx = hash(z) % len(_FALLBACK)
    return _FALLBACK[idx]


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
    """PCA projection of descriptors, coloured by element and dataset.

    Train markers use lighter shades; Test markers use darker shades of the
    same per-element colour.  PCA is fitted on the union of both sets.
    """
    # --- gather unique elements ---
    all_z: List[int] = []
    if train_z is not None:
        all_z.extend(np.unique(train_z).tolist())
    if test_z is not None:
        all_z.extend(np.unique(test_z).tolist())
    unique_z = sorted(set(all_z))

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
        base = _get_element_color(z)
        light = _lighten(base)
        dark = _darken(base)
        symbol = _get_symbol(z)

        # Train
        if train_desc is not None and train_z is not None:
            mask = train_z == z
            if mask.any():
                proj = pca.transform(train_desc[mask])
                ax.scatter(proj[:, 0], proj[:, 1], s=marker_size, alpha=0.7,
                           c=light, marker="o", edgecolors="none",
                           label=f"{symbol} (Train)")

        # Test
        if test_desc is not None and test_z is not None:
            mask = test_z == z
            if mask.any():
                proj = pca.transform(test_desc[mask])
                ax.scatter(proj[:, 0], proj[:, 1], s=marker_size * 1.5, alpha=0.85,
                           c=dark, marker="s", edgecolors="none",
                           label=f"{symbol} (Test)")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    # Legend outside
    n_elements = len(unique_z)
    ncol = 2 if n_elements > 5 else 1
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              framealpha=0.8, fontsize=13, ncol=ncol,
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
