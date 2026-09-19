"""ZBL pair curves: energy/force vs. distance for every element pair (potential + ZBL)."""

import os
import re
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols

from ai2pot_cli.menu import (print_section, print_kv, print_sep,
                             print_error, print_success, print_warning)

# Paper-ready style (consistent with plot_parity.py / plot_descriptors.py)
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

OUT_DIR_NAME = "zbl_analysis"
PLOT_NAME = "zbl_plot.png"
ENERGY_NPY_NAME = "zbl_energy-distance.npy"
FORCE_NPY_NAME = "zbl_force-distance.npy"

DEFAULT_RMIN = 0.5
DEFAULT_RMAX = 6.0
DEFAULT_N_POINTS = 300
MIN_BOX_LENGTH = 20.0

PAIR_COLORS = [
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


def _get_symbol(z: int) -> str:
    """Atomic number → symbol via ase."""
    if 0 < z < len(chemical_symbols):
        return chemical_symbols[z]
    return f"Z{z}"


def _detect_model_type(checkpoint_path: str) -> str:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    if "n_radial_basis" in hp:
        return "nep"
    if "mtp_level" in hp and "num_neurons" in hp:
        return "nnmtp"
    if "mtp_level" in hp:
        return "mtp"
    raise ValueError(
        "Cannot detect model type from checkpoint. "
        "Expected 'mtp_level'/'num_neurons' or 'n_radial_basis' in hyper_parameters."
    )


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_calculator(checkpoint_path: str, device: str):
    """ASE calculator of the trained potential; its predict_ef already contains the ZBL term."""
    model_type = _detect_model_type(checkpoint_path)
    if model_type == "mtp":
        from ai2pot.models.mtp.linear_mtp_utils import LinearMtpCalculator
        calculator = LinearMtpCalculator(checkpoint_path=checkpoint_path, map_location=device)
    elif model_type == "nnmtp":
        from ai2pot.models.mtp.nn_mtp_utils import NNMtpCalculator
        calculator = NNMtpCalculator(checkpoint_path=checkpoint_path, map_location=device)
    else:
        from ai2pot.models.nep.nep_utils import NepCalculator
        calculator = NepCalculator(checkpoint_path=checkpoint_path, map_location=device)
    # load_from_checkpoint(map_location=...) only maps the checkpoint storages; the module
    # parameters themselves stay on CPU, while mlff_input tensors follow map_location ->
    # move the module explicitly so predict_ef runs entirely on `device`.
    calculator.lit_module.to(torch.device(device))
    return calculator


def _parse_distance_range(distance_range: Union[None, str, Sequence[float]]) -> Tuple[float, float]:
    """Return (rmin, rmax) from None | (rmin, rmax) | '0.5-6.0' | '0.5 6.0' | '6.0'."""
    if distance_range is None:
        return DEFAULT_RMIN, DEFAULT_RMAX
    if isinstance(distance_range, (tuple, list)):
        values = [float(v) for v in distance_range]
    else:
        tokens = re.split(r"[-\s,~]+", str(distance_range).strip())
        values = [float(t) for t in tokens if t]
    if not values:
        return DEFAULT_RMIN, DEFAULT_RMAX
    if len(values) == 1:
        values = [DEFAULT_RMIN, values[0]]
    if len(values) != 2:
        raise ValueError(
            f"Invalid ZBL distance range '{distance_range}'. "
            "Expected a range like 0.5-6.0 or a single upper bound like 6.0."
        )
    rmin, rmax = values
    if rmin <= 0.0 or rmax <= rmin:
        raise ValueError(
            f"Invalid ZBL distance range '{distance_range}'. "
            "Expected 0 < min < max."
        )
    return rmin, rmax


def _eval_dimer(calculator,
                symbol_i: str,
                symbol_j: str,
                distance: float,
                box_length: float) -> Tuple[float, np.ndarray]:
    """Energy and forces of an isolated i-j dimer separated by ``distance``."""
    atoms = Atoms(symbols=[symbol_i, symbol_j],
                  positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]],
                  cell=[box_length, box_length, box_length],
                  pbc=True)
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return float(energy), forces


def _scan_element_pairs(calculator,
                        distances: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Dimer scan over all unique element pairs.

    Returns labels plus (n_pairs, n_distances) arrays of the pair interaction energy
    (referenced to isolated atoms) and the radial force -dE/dr (> 0 repulsive).
    """
    model = calculator.model
    type_map = [int(z) for z in model.type_map_tensor.detach().cpu().numpy().tolist()]
    rcut = float(model.rmax)
    zbl_rmax = float(model.zbl_rmax)

    # periodic images of the dimer must stay outside all cutoffs
    box_length = max(MIN_BOX_LENGTH,
                     2.0 * (float(distances.max()) + rcut + zbl_rmax) + 4.0)
    r_ref = box_length / 2.0

    pairs: List[Tuple[str, str, str]] = []
    for i in range(len(type_map)):
        for j in range(i, len(type_map)):
            symbol_i = _get_symbol(type_map[i])
            symbol_j = _get_symbol(type_map[j])
            pairs.append((symbol_i, symbol_j, f"{symbol_i}-{symbol_j}"))

    energy = np.empty((len(pairs), distances.size))
    force = np.empty((len(pairs), distances.size))

    for p, (symbol_i, symbol_j, _label) in enumerate(pairs):
        print_success(f"Scanning pair {p + 1}/{len(pairs)}: {symbol_i}-{symbol_j} ...")
        e_ref, _ = _eval_dimer(calculator, symbol_i, symbol_j, r_ref, box_length)
        for k, tmp_distance in enumerate(distances):
            tmp_energy, tmp_forces = _eval_dimer(calculator,
                                                 symbol_i, symbol_j,
                                                 float(tmp_distance), box_length)
            energy[p, k] = tmp_energy - e_ref
            # -dE/dr from the equal-and-opposite z-components of the two atomic forces
            force[p, k] = 0.5 * (tmp_forces[1][2] - tmp_forces[0][2])

    labels = [label for _, _, label in pairs]
    return labels, energy, force


def _make_zbl_plot(labels: List[str],
                   distances: np.ndarray,
                   energy: np.ndarray,
                   force: np.ndarray,
                   zbl_rmax: float,
                   output_path: str):
    """Two panels: pair energy and radial force vs. distance, one curve per element pair."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5 * 2, 5.8), squeeze=False)
    axes = axes[0]

    colors = [PAIR_COLORS[i % len(PAIR_COLORS)] for i in range(len(labels))]
    legend_cols = 2 if len(labels) > 6 else 1

    for ax, curves, ylabel in ((axes[0], energy, "Pair Energy (eV)"),
                               (axes[1], force, "Radial Force (eV/A)")):
        for color, label, curve in zip(colors, labels, curves):
            ax.plot(distances, curve, color=color, linewidth=2.2, label=label)
        if zbl_rmax > 0.0:
            ax.axvline(zbl_rmax, color="grey", linestyle="--", linewidth=1.2,
                       label="ZBL Rmax")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Distance (A)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", framealpha=0.8, fontsize=12, ncol=legend_cols)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_zbl(
    checkpoint_path: str,
    distance_range: Union[None, str, Sequence[float]] = None,
    n_points: int = DEFAULT_N_POINTS,
    output_path: Optional[str] = None,
):
    """Plot energy/force vs. distance for every element pair of a trained potential.

    The checkpoint's potential (MTP / NEP / NNMTP) is evaluated together with its ZBL
    correction. Energies are pair interaction energies referenced to isolated atoms;
    forces are the radial force -dE/dr (positive = repulsive).

    Args:
        checkpoint_path: Path to the .ckpt checkpoint file.
        distance_range: None (default 0.5-6.0 A), (rmin, rmax), or a string such as
            "0.5-6.0" / "0.5 6.0" / "6.0" (single value as upper bound).
        n_points: Number of distances scanned.
        output_path: Path of the plot; defaults to ./zbl_analysis/zbl_plot.png.
    """
    rmin, rmax = _parse_distance_range(distance_range)

    if output_path is None:
        out_dir = os.path.join(os.getcwd(), OUT_DIR_NAME)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, PLOT_NAME)
    else:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
    abs_output = os.path.abspath(output_path)

    # --- Device ---
    device = _get_device()
    print_section(f"Running on {device.upper()}")

    # --- Model ---
    model_type = _detect_model_type(checkpoint_path)
    print_success(f"Loading checkpoint: {checkpoint_path}")
    calculator = _build_calculator(checkpoint_path, device)

    model = calculator.model
    type_map = [int(z) for z in model.type_map_tensor.detach().cpu().numpy().tolist()]
    zbl_rmax = float(model.zbl_rmax)
    zbl_typewise_factor = float(model.zbl_typewise_factor)

    print_kv("Model Type", model_type.upper())
    print_kv("Type Map", ", ".join(_get_symbol(z) for z in type_map))
    print_kv("ZBL Rmax (A)", f"{zbl_rmax:.2f}")
    print_kv("ZBL Typewise Factor", f"{zbl_typewise_factor:.2f}")
    print_kv("Distance Range (A)", f"{rmin:.2f} - {rmax:.2f}")
    print_kv("Distance Points", f"{n_points}")
    if zbl_rmax <= 0.0:
        print_warning("ZBL is disabled in this checkpoint (zbl_rmax = 0); "
                      "curves show the bare pair interaction.")

    # --- Compute ---
    distances = np.linspace(rmin, rmax, n_points)
    labels, energy, force = _scan_element_pairs(calculator, distances)
    print_success(f"Scanned {len(labels)} element pair(s): {', '.join(labels)}")

    # --- Save data (row 0 = distance, rows 1..N = pairs in the printed order) ---
    energy_path = os.path.join(out_dir, ENERGY_NPY_NAME)
    force_path = os.path.join(out_dir, FORCE_NPY_NAME)
    np.save(energy_path, np.vstack([distances[None, :], energy]))
    np.save(force_path, np.vstack([distances[None, :], force]))

    # --- Plot ---
    _make_zbl_plot(labels, distances, energy, force, zbl_rmax, abs_output)

    # --- Print results ---
    print_section("ZBL Plot Generated Successfully")
    print_kv("Output Dir", out_dir)
    print_kv("Output Plot", abs_output)
    print_kv("Energy Data", energy_path)
    print_kv("Force Data", force_path)
    print()

    for label, curve_e, curve_f in zip(labels, energy, force):
        idx = int(np.argmin(curve_e))
        print(f"  {'Pair ' + label:<18}: "
              f"E min = {curve_e[idx]:>9.3f} eV | "
              f"r min = {distances[idx]:>5.2f} A | "
              f"F max = {curve_f.max():>10.2f} eV/A")

    print_kv("Data Layout", "row 0 = distance (A), rows 1..N = pairs (in the order above)")
    print_sep()
    print()
