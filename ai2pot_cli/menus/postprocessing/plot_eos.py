"""EOS plot: energy-vs-volume curve of a trained potential for a scaled reference structure."""

import os
import re
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read
from scipy.optimize import curve_fit

from ai2pot_cli.menu import (print_section, print_kv, print_sep,
                             print_success, print_warning)

# Paper-ready style (consistent with plot_parity.py / plot_zbl.py)
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

OUT_DIR_NAME = "eos_analysis"
PLOT_NAME = "eos_plot.png"
DATA_NAME = "eos_data.dat"

# Lattice scale factors s (each lattice vector x s); s = 1.0 is the original structure.
DEFAULT_SCALE_MIN = 0.90
DEFAULT_SCALE_MAX = 1.10
DEFAULT_N_POINTS = 21
MIN_N_POINTS = 5

# Birch-Murnaghan B0 is fitted in eV/A^3; 1 eV/A^3 = 160.2176634 GPa
EV_A3_TO_GPA = 160.2176634

POINT_COLOR = "#4477AA"
ORIGINAL_COLOR = "#d73027"


def _get_symbol(z: int) -> str:
    """Atomic number → symbol via ase."""
    from ase.data import chemical_symbols
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
    """ASE calculator of the trained potential (MTP / NNMTP / NEP)."""
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


def _parse_scale_range(scale_range: Union[None, str, Sequence[float]]) -> Tuple[float, float]:
    """Return (s_min, s_max) from None | (s_min, s_max) | '0.90-1.10' | '0.9 1.1' | '0.10' (symmetric +/- 0.10)."""
    if scale_range is None:
        return DEFAULT_SCALE_MIN, DEFAULT_SCALE_MAX
    if isinstance(scale_range, (tuple, list)):
        values = [float(v) for v in scale_range]
    else:
        values = []
        for token in re.split(r"[-\s,~]+", str(scale_range).strip()):
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                raise ValueError(
                    f"Invalid lattice scale range '{scale_range}'. "
                    "Expected a range like 0.90-1.10 or a single deviation like 0.10."
                )
    if not values:
        return DEFAULT_SCALE_MIN, DEFAULT_SCALE_MAX
    if len(values) == 1:
        deviation = values[0]
        values = [1.0 - deviation, 1.0 + deviation]
    if len(values) != 2:
        raise ValueError(
            f"Invalid lattice scale range '{scale_range}'. "
            "Expected a range like 0.90-1.10 or a single deviation like 0.10."
        )
    smin, smax = values
    if smin <= 0.0 or smax <= smin:
        raise ValueError(
            f"Invalid lattice scale range '{scale_range}'. Expected 0 < min < max."
        )
    if not (smin <= 1.0 <= smax):
        raise ValueError(
            f"Invalid lattice scale range '{scale_range}'. "
            "The range must include s = 1.0 (the original structure)."
        )
    return smin, smax


def _build_scales(smin: float, smax: float, n_points: int) -> np.ndarray:
    """Uniform scale grid on [smin, smax] that always contains s = 1.0."""
    if n_points < MIN_N_POINTS:
        raise ValueError(
            f"Invalid number of points {n_points}; at least {MIN_N_POINTS} are required for the EOS fit."
        )
    if np.isclose(smin + smax, 2.0) and n_points % 2 == 0:
        print_warning(f"An even number of points cannot include the original structure (s = 1.0) "
                      f"on a symmetric scale range; using {n_points + 1} points.")
        n_points += 1
    scales = np.linspace(smin, smax, n_points)
    index_original = int(np.argmin(np.abs(scales - 1.0)))
    if not np.isclose(scales[index_original], 1.0):
        raise ValueError(
            f"The scale grid {smin:g} .. {smax:g} with {n_points} points does not contain s = 1.0. "
            "Adjust the range or the number of points so that the original structure is included "
            "(e.g. a symmetric range with an odd number of points)."
        )
    scales[index_original] = 1.0  # snap the original point exactly onto s = 1.0
    return scales


def _birch_murnaghan_3rd(v, e0, v0, b0, b0p):
    """Birch-Murnaghan 3rd-order equation of state E(V)."""
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((eta - 1.0) ** 3 * b0p + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta))


def _fit_eos(volumes: np.ndarray, energies: np.ndarray) -> Dict[str, float]:
    """Fit Birch-Murnaghan 3rd order; returns dict(e0, v0, b0 (eV/A^3), b0p) with per-atom units."""
    # parabolic pre-estimate: a V^2 + b V + c -> V0 = -b/(2a), B0 = 2a V0
    a, b, c = np.polyfit(volumes, energies, 2)
    if a > 0.0:
        v0_guess = -b / (2.0 * a)
        e0_guess = a * v0_guess ** 2 + b * v0_guess + c
        b0_guess = 2.0 * a * v0_guess
    else:
        index_min = int(np.argmin(energies))
        v0_guess, e0_guess, b0_guess = volumes[index_min], energies[index_min], 0.3
    try:
        popt, _ = curve_fit(
            _birch_murnaghan_3rd, volumes, energies,
            p0=[e0_guess, v0_guess, b0_guess, 4.0],
            bounds=([-np.inf, 0.5 * volumes.min(), 1e-8, -10.0],
                    [np.inf, 2.0 * volumes.max(), np.inf, 20.0]),
            maxfev=10000)
    except RuntimeError as exc:
        raise ValueError(f"Birch-Murnaghan fit did not converge: {exc}")
    e0, v0, b0, b0p = (float(p) for p in popt)
    return {"e0": e0, "v0": v0, "b0": b0, "b0p": b0p}


def _make_eos_plot(volumes: np.ndarray,
                   energies: np.ndarray,
                   index_original: int,
                   fit: Dict[str, float],
                   model_label: str,
                   output_path: str):
    """E-V points plus the fitted Birch-Murnaghan curve, annotated with the fit parameters."""
    fig, ax = plt.subplots(figsize=(6.5, 5.8))

    fit_volumes = np.linspace(float(volumes.min()), float(volumes.max()), 300)
    ax.plot(fit_volumes,
            _birch_murnaghan_3rd(fit_volumes, fit["e0"], fit["v0"], fit["b0"], fit["b0p"]),
            color="black", linewidth=2.0, label="Birch-Murnaghan fit", zorder=2)
    ax.scatter(volumes, energies, s=60, alpha=0.9, c=POINT_COLOR,
               marker="o", edgecolors="none", label=model_label, zorder=3)
    ax.scatter([volumes[index_original]], [energies[index_original]],
               s=90, facecolors="none", edgecolors=ORIGINAL_COLOR, marker="D",
               linewidths=1.6, label="Original structure", zorder=4)
    ax.axvline(fit["v0"], color="grey", linestyle="--", linewidth=1.2, zorder=1)

    ax.set_xlabel("Volume (A^3/atom)")
    ax.set_ylabel("Energy (eV/atom)")
    ax.legend(loc="lower right", framealpha=0.8)

    text = (f"V0 = {fit['v0']:.4f} A^3/atom\n"
            f"E0 = {fit['e0']:.4f} eV/atom\n"
            f"B0 = {fit['b0'] * EV_A3_TO_GPA:.2f} GPa\n"
            f"B0' = {fit['b0p']:.2f}")
    ax.text(0.05, 0.92, text, transform=ax.transAxes, fontsize=16,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_eos_dat(dat_path: str,
                   structure_path: str,
                   checkpoint_path: str,
                   model_type: str,
                   formula: str,
                   n_atoms: int,
                   scales: np.ndarray,
                   volumes_total: np.ndarray,
                   energies_total: np.ndarray,
                   fit: Dict[str, float]):
    """Human-readable .dat table: scale, volume and energy (total and per atom) plus fit parameters."""
    lines = [
        "# EOS data (energy vs. volume) -- generated by ai2pot-cli",
        f"# Structure  : {structure_path}",
        f"# Checkpoint : {checkpoint_path}  (model type: {model_type.upper()})",
        f"# Formula    : {formula} ({n_atoms} atoms)",
        "# Columns    : scale  V_total(A^3)  V_per_atom(A^3/atom)  E_total(eV)  E_per_atom(eV/atom)",
        "# Fit (Birch-Murnaghan 3rd order, per atom):",
        f"#   V0 = {fit['v0']:.6f} A^3/atom",
        f"#   E0 = {fit['e0']:.8f} eV/atom",
        f"#   B0 = {fit['b0'] * EV_A3_TO_GPA:.4f} GPa",
        f"#   B0'= {fit['b0p']:.6f}",
    ]
    for s, v_total, v_atom, e_total, e_atom in zip(scales, volumes_total,
                                                   volumes_total / n_atoms,
                                                   energies_total,
                                                   energies_total / n_atoms):
        lines.append(f"  {s:8.4f}  {v_total:16.6f}  {v_atom:16.6f}  {e_total:18.8f}  {e_atom:18.8f}")
    with open(dat_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_eos(
    checkpoint_path: str,
    structure_path: str,
    scale_range: Union[None, str, Sequence[float]] = None,
    n_points: Optional[int] = None,
    output_path: Optional[str] = None,
):
    """Plot the energy-vs-volume (EOS) curve of a trained potential.

    The reference structure is scaled uniformly (every lattice vector x s, fractional
    coordinates unchanged); the grid spans ``scale_range`` and always contains the
    original structure (s = 1.0). Every scaled structure is evaluated with the
    checkpoint's potential and a Birch-Murnaghan 3rd-order EOS is fitted per atom.

    Args:
        checkpoint_path: Path to the .ckpt checkpoint file (MTP / NNMTP / NEP).
        structure_path: Reference structure (POSCAR / extxyz); the first frame is used.
        scale_range: None (default 0.90-1.10), (s_min, s_max), or a string such as
            "0.90-1.10" / "0.9 1.1" / "0.10" (single value = symmetric deviation).
        n_points: Number of scale points including the original; default 21.
        output_path: Path of the plot; defaults to ./eos_analysis/eos_plot.png.
    """
    smin, smax = _parse_scale_range(scale_range)
    if n_points is None:
        n_points = DEFAULT_N_POINTS
    scales = _build_scales(smin, smax, n_points)

    if output_path is None:
        out_dir = os.path.join(os.getcwd(), OUT_DIR_NAME)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, PLOT_NAME)
    else:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
    abs_output = os.path.abspath(output_path)

    # --- Reference structure ---
    if not os.path.isfile(structure_path):
        raise ValueError(f"Structure file not found: {structure_path}")
    atoms = ase_read(structure_path, index=0)
    if not np.all(atoms.pbc):
        raise ValueError("The reference structure must be periodic in all three directions (pbc).")
    if atoms.cell.rank < 3 or float(np.linalg.det(atoms.cell.array)) <= 0.0:
        raise ValueError("The reference structure must have a complete 3D cell (non-zero volume).")
    n_atoms = len(atoms)
    base_positions = atoms.get_positions()
    base_cell = np.array(atoms.cell.array)

    # --- Device / model ---
    device = _get_device()
    print_section(f"Running on {device.upper()}")
    model_type = _detect_model_type(checkpoint_path)
    print_success(f"Loading checkpoint: {checkpoint_path}")
    calculator = _build_calculator(checkpoint_path, device)

    model = calculator.model
    type_map = [int(z) for z in model.type_map_tensor.detach().cpu().numpy().tolist()]
    missing = sorted(set(int(z) for z in atoms.get_atomic_numbers()) - set(type_map))
    if missing:
        raise ValueError(
            "Reference structure contains element(s) not in the checkpoint type map: "
            + ", ".join(_get_symbol(z) for z in missing)
            + f". Type map = {', '.join(_get_symbol(z) for z in type_map)}."
        )
    print_success(f"Loading reference structure: {structure_path}")

    print_kv("Model Type", model_type.upper())
    print_kv("Type Map", ", ".join(_get_symbol(z) for z in type_map))
    print_kv("Formula", f"{atoms.get_chemical_formula()} ({n_atoms} atoms)")
    print_kv("Scale Range", f"{smin:.2f} - {smax:.2f} (V/V0 = {smin ** 3:.3f} - {smax ** 3:.3f})")
    print_kv("Points", f"{len(scales)} (incl. s = 1.00)")

    # --- Evaluate EOS points ---
    volumes_total = np.empty(len(scales))
    energies_total = np.empty(len(scales))
    for k, s in enumerate(scales):
        print_success(f"Evaluating scale s = {s:.2f} ({k + 1}/{len(scales)}) ...")
        # scale the lattice vectors and keep the fractional coordinates: the cartesian
        # positions scale by s as well (uniform dilation of the whole configuration)
        scaled = Atoms(numbers=atoms.get_atomic_numbers(),
                       positions=base_positions * s,
                       cell=base_cell * s,
                       pbc=True)
        scaled.calc = calculator
        energies_total[k] = float(scaled.get_potential_energy())
        volumes_total[k] = float(np.linalg.det(scaled.cell.array))
    print_success(f"Evaluated {len(scales)} EOS point(s)")

    volumes = volumes_total / n_atoms
    energies = energies_total / n_atoms

    # --- Fit ---
    fit = _fit_eos(volumes, energies)
    print_success(f"Fitted Birch-Murnaghan 3rd order: V0 = {fit['v0']:.4f} A^3/atom, "
                  f"E0 = {fit['e0']:.4f} eV/atom, B0 = {fit['b0'] * EV_A3_TO_GPA:.2f} GPa, "
                  f"B0' = {fit['b0p']:.2f}")

    # --- Save data ---
    data_path = os.path.join(out_dir, DATA_NAME)
    _write_eos_dat(data_path, os.path.abspath(structure_path), os.path.abspath(checkpoint_path),
                   model_type, atoms.get_chemical_formula(), n_atoms, scales,
                   volumes_total, energies_total, fit)

    # --- Plot ---
    index_original = int(np.argmin(np.abs(scales - 1.0)))
    _make_eos_plot(volumes, energies, index_original, fit, model_type.upper(), abs_output)

    # --- Print results ---
    print_section("EOS Plot Generated Successfully")
    print_kv("Output Dir", out_dir)
    print_kv("Output Plot", abs_output)
    print_kv("Output Data", data_path)
    print()
    print("  EOS Fit (Birch-Murnaghan 3rd order):")
    print_kv("V0 (A^3/atom)", f"{fit['v0']:.4f}")
    print_kv("E0 (eV/atom)", f"{fit['e0']:.4f}")
    print_kv("B0 (GPa)", f"{fit['b0'] * EV_A3_TO_GPA:.2f}")
    print_kv("B0'", f"{fit['b0p']:.2f}")
    print_sep()
    print()
