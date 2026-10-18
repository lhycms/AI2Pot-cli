"""MTP Active Learning -- build the MaxVol basis and analyze extrapolation levels."""

import os
import sys

from ai2pot_cli.menu import (
    show_numbered_menu,
    get_choice,
    print_section,
    print_success,
    print_warning,
    print_error,
    print_kv,
    print_sep,
)
from ai2pot_cli.commands import register

DEFAULT_CHECKPOINT: str = "./lightning_logs/version_0/checkpoints/last.ckpt"
DEFAULT_OUTPUT: str = "./maxvol_matrix.pt"
DEFAULT_ACTIVE_SET: str = "./active_set.xyz"
DEFAULT_MAXVOL: str = "./maxvol_matrix.pt"
DEFAULT_EXPLORE: str = "./explore.xyz"
DEFAULT_GAMMA_DIR: str = "./gamma_analysis"
DEFAULT_MAXITER: int = 20000
DEFAULT_TOL: float = 1e-4
DEFAULT_RANK_TOL: float = 1e-12

_MENU_ITEMS = [
    (141, "Initialize Maxvol Matrix", "Build the initial D-optimal basis from a labeled candidate set"),
    (142, "Analyze Extrapolation Level (Gamma)", "Compute gamma for a new candidate set with the MaxVol matrix"),
]

register(141, "ai2pot_cli.menus.preprocessing.mtp_active_learning", "initialize_maxvol_matrix")
register(142, "ai2pot_cli.menus.preprocessing.mtp_active_learning", "analyze_gamma")


def mtp_active_learning_menu():
    """Sub-menu for MTP Active Learning (choice 14)."""
    while True:
        show_numbered_menu("MTP Active Learning", _MENU_ITEMS)
        choice = get_choice()

        from ai2pot_cli.commands import dispatch
        if dispatch(choice):
            continue

        if choice == 141:
            initialize_maxvol_matrix()
        elif choice == 9:
            return
        elif choice == 0:
            print_success("Bye.")
            sys.exit(0)
        else:
            print_warning(f"Invalid option: {choice}")


def _ask_checkpoint_path() -> str:
    checkpoint_path = input(f" Checkpoint path (.ckpt) [default: {DEFAULT_CHECKPOINT}]: ").strip() or DEFAULT_CHECKPOINT
    if not os.path.isfile(checkpoint_path):
        print_error(f"File not found: {checkpoint_path}")
        return ""
    return checkpoint_path


def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _detect_model_type(checkpoint_path: str) -> str:
    import torch

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


def initialize_maxvol_matrix():
    """Generate the MaxVol matrix for a candidate set with the trained MTP checkpoint."""
    print_section("MTP Active Learning -- Initialize Maxvol Matrix")

    checkpoint_path = _ask_checkpoint_path()
    if not checkpoint_path:
        return

    candidate_set_path = input(" Candidate set path (.xyz): ").strip()
    if not candidate_set_path:
        print_warning("No candidate set path provided.")
        return
    if not os.path.isfile(candidate_set_path):
        print_error(f"File not found: {candidate_set_path}")
        return

    pbc_xyz = [True, True, True]

    output_path = input(f" Output path [default: {DEFAULT_OUTPUT}]: ").strip() or DEFAULT_OUTPUT
    if not output_path.lower().endswith(".pt"):
        output_path += ".pt"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    active_set_path = input(
        f" Active set path (.xyz) [default: {DEFAULT_ACTIVE_SET}]: "
    ).strip() or DEFAULT_ACTIVE_SET
    if not active_set_path.lower().endswith(".xyz"):
        active_set_path += ".xyz"
    os.makedirs(os.path.dirname(os.path.abspath(active_set_path)), exist_ok=True)

    device = _get_device()
    print_section(f"Running on {device.upper()}")
    print_kv("Checkpoint", checkpoint_path)
    print_kv("Candidate Set", candidate_set_path)
    print_sep()

    try:
        model_type = _detect_model_type(checkpoint_path)
    except Exception as exc:
        print_error(str(exc))
        return
    if model_type != "mtp":
        print_error(f"Checkpoint is a {model_type.upper()} model; this tool requires an MTP checkpoint.")
        return

    try:
        import torch

        from ai2pot.models.mtp.linear_mtp_al_utils import LinearMtpMaxvolGenerator

        generator = LinearMtpMaxvolGenerator(
            checkpoint_path=checkpoint_path,
            candidate_set_path=candidate_set_path,
            map_location=device,
            pbc_xyz=pbc_xyz,
        )
        # load_from_checkpoint(map_location=...) only maps the checkpoint storages;
        # move the module explicitly so the Jacobian runs on `device`.
        generator.lit_module.to(torch.device(device))
        if not hasattr(generator, "generate_maxvol_matrix"):
            print_error(
                "LinearMtpMaxvolGenerator.generate_maxvol_matrix() is not available; "
                "please update AI2Pot to a version that provides it."
            )
            return

        print_success("Initializing MaxVol matrix ...")
        maxvol_structure_id, maxvol_atom_id, maxvol_matrix, maxvol_matrix_inv = \
            generator.generate_maxvol_matrix(
                maxiter=DEFAULT_MAXITER,
                tol=DEFAULT_TOL,
                rank_tol=DEFAULT_RANK_TOL,
            )
    except Exception as exc:
        print_error(str(exc))
        return

    selected_structure_ids = [int(i) for i in torch.unique(maxvol_structure_id).cpu().tolist()]
    try:
        from ase.io import iread, write

        wanted_ids = set(selected_structure_ids)
        structures_by_id = {}
        for frame_index, atoms in enumerate(iread(candidate_set_path)):
            if frame_index in wanted_ids:
                structures_by_id[frame_index] = atoms
                if len(structures_by_id) == len(wanted_ids):
                    break
        missing_ids = [i for i in selected_structure_ids if i not in structures_by_id]
        if missing_ids:
            raise ValueError(
                f"Selected structure id {missing_ids[0]} is out of range "
                f"for the candidate set."
            )
        active_structures = [structures_by_id[i] for i in selected_structure_ids]
    except Exception as exc:
        print_error(f"Failed to build active set: {exc}")
        return

    torch.save(
        {
            "maxvol_structure_id": maxvol_structure_id.cpu(),
            "maxvol_atom_id": maxvol_atom_id.cpu(),
            "maxvol_matrix": maxvol_matrix.cpu(),
            "maxvol_matrix_inv": maxvol_matrix_inv.cpu(),
            "checkpoint_path": checkpoint_path,
            "candidate_set_path": candidate_set_path,
            "pbc_xyz": pbc_xyz,
            "maxiter": DEFAULT_MAXITER,
            "tol": DEFAULT_TOL,
            "rank_tol": DEFAULT_RANK_TOL,
        },
        output_path,
    )
    maxvol_matrix_inv_path = os.path.join(
        os.path.dirname(os.path.abspath(output_path)), "maxvol_matrix_inv.pt"
    )
    torch.save(maxvol_matrix_inv.cpu(), maxvol_matrix_inv_path)
    try:
        write(active_set_path, active_structures, format="extxyz")
    except Exception as exc:
        print_error(f"Failed to write active set: {exc}")
        return

    num_parameters = int(maxvol_matrix.size(0))

    print_section("Maxvol Matrix Initialized Successfully")
    print_kv("Output File", os.path.abspath(output_path))
    print_kv("Inverse Matrix", os.path.abspath(maxvol_matrix_inv_path))
    print_kv("Active Set", os.path.abspath(active_set_path))
    print_kv("Matrix Shape", f"{num_parameters} x {num_parameters}")
    print_kv("Selected Structures", str(len(selected_structure_ids)))
    print_kv("Selected Atoms", str(num_parameters))
    print_kv("Tolerance", f"{DEFAULT_TOL:g}")
    print_sep()
    print()

    sys.exit(0)


def analyze_gamma():
    """Compute extrapolation levels (gamma) for a new candidate set."""
    print_section("MTP Active Learning -- Analyze Extrapolation Level (Gamma)")

    maxvol_path = input(f" Maxvol matrix path (.pt) [default: {DEFAULT_MAXVOL}]: ").strip() or DEFAULT_MAXVOL
    if not os.path.isfile(maxvol_path):
        print_error(f"File not found: {maxvol_path}")
        return

    explore_path = input(f" Explore set path (.xyz) [default: {DEFAULT_EXPLORE}]: ").strip() or DEFAULT_EXPLORE
    if not os.path.isfile(explore_path):
        print_error(f"File not found: {explore_path}")
        return

    output_dir = input(f" Output directory [default: {DEFAULT_GAMMA_DIR}]: ").strip() or DEFAULT_GAMMA_DIR
    os.makedirs(output_dir, exist_ok=True)

    device = _get_device()
    print_section(f"Running on {device.upper()}")
    print_kv("Maxvol Matrix", maxvol_path)
    print_kv("Explore Set", explore_path)
    print_sep()

    try:
        import torch

        maxvol_data = torch.load(maxvol_path, map_location="cpu", weights_only=False)
        checkpoint_path = maxvol_data["checkpoint_path"]
        maxvol_matrix_inv = maxvol_data["maxvol_matrix_inv"].to(torch.device(device))
        pbc_xyz = [bool(v) for v in maxvol_data.get("pbc_xyz", [True, True, True])]

        model_type = _detect_model_type(checkpoint_path)
        if model_type != "mtp":
            print_error(f"Checkpoint is a {model_type.upper()} model; this tool requires an MTP checkpoint.")
            return

        from ai2pot.models.mtp.linear_mtp_al_utils import LinearMtpMaxvolGenerator

        generator = LinearMtpMaxvolGenerator(
            checkpoint_path=checkpoint_path,
            candidate_set_path=explore_path,
            map_location=device,
            pbc_xyz=pbc_xyz,
        )
        # load_from_checkpoint(map_location=...) only maps the checkpoint storages;
        # move the module explicitly so the Jacobian runs on `device`.
        generator.lit_module.to(torch.device(device))
        generator.lit_module.eval()
        if not hasattr(generator, "get_gamma"):
            print_error(
                "LinearMtpMaxvolGenerator.get_gamma() is not available; "
                "please update AI2Pot to a version that provides it."
            )
            return

        print_success("Computing gamma ...")
        gamma_structures, gamma_atoms_padded = generator.get_gamma(
            maxvol_matrix_inv=maxvol_matrix_inv,
            explore_path=explore_path,
        )
        gamma_structures = gamma_structures.cpu()

        from ase.io import iread

        num_atoms_list = [len(atoms) for atoms in iread(explore_path)]
        gamma_atoms_padded = gamma_atoms_padded.cpu()
        if gamma_atoms_padded.size(0) != len(num_atoms_list):
            raise ValueError(
                f"Gamma structure count ({gamma_atoms_padded.size(0)}) does not match "
                f"the explore set ({len(num_atoms_list)})."
            )
        gamma_atoms = torch.cat(
            [
                gamma_atoms_padded[structure_index, :num_atoms]
                for structure_index, num_atoms in enumerate(num_atoms_list)
            ]
        )
        num_atoms_per_structure = torch.tensor(num_atoms_list, dtype=torch.long)
        structure_ids = torch.repeat_interleave(
            torch.arange(num_atoms_per_structure.numel(), dtype=torch.long),
            num_atoms_per_structure,
        )
    except Exception as exc:
        print_error(str(exc))
        return

    gamma_structure_path = os.path.join(output_dir, "gamma_structure.pt")
    gamma_atoms_path = os.path.join(output_dir, "gamma_atoms.pt")
    histogram_path = os.path.join(output_dir, "gamma_histogram.png")

    metadata = {
        "checkpoint_path": checkpoint_path,
        "maxvol_path": maxvol_path,
        "explore_path": explore_path,
        "pbc_xyz": pbc_xyz,
    }
    torch.save(
        {
            "gamma_structures": gamma_structures,
            "num_atoms_per_structure": num_atoms_per_structure,
            **metadata,
        },
        gamma_structure_path,
    )
    torch.save(
        {
            "gamma_atoms": gamma_atoms,
            "structure_ids": structure_ids,
            "num_atoms_per_structure": num_atoms_per_structure,
            **metadata,
        },
        gamma_atoms_path,
    )

    try:
        _plot_gamma_histogram(gamma_structures, gamma_atoms, histogram_path)
    except Exception as exc:
        print_error(f"Failed to plot gamma histogram: {exc}")
        return

    print_section("Gamma Analysis Generated Successfully")
    print_kv("Output Dir", os.path.abspath(output_dir))
    print_kv("Structure Gamma", os.path.abspath(gamma_structure_path))
    print_kv("Atom Gamma", os.path.abspath(gamma_atoms_path))
    print_kv("Histogram", os.path.abspath(histogram_path))
    print()
    print(f"  {'Structures':<22}: {gamma_structures.numel()}")
    print(f"  {'Atoms':<22}: {gamma_atoms.numel()}")
    print(f"  {'Max Gamma (Structure)':<22}: {gamma_structures.max().item():.4f}")
    print(f"  {'Mean Gamma (Structure)':<22}: {gamma_structures.mean().item():.4f}")
    print(f"  {'Max Gamma (Atom)':<22}: {gamma_atoms.max().item():.4f}")
    print(f"  {'Mean Gamma (Atom)':<22}: {gamma_atoms.mean().item():.4f}")
    print_sep()
    print()

    sys.exit(0)


def _plot_gamma_histogram(gamma_structures, gamma_atoms, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Paper-ready style
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

    structure_values = gamma_structures.detach().cpu().numpy()
    atom_values = gamma_atoms.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    panels = (
        (axes[0], structure_values, "Structure Gamma", "#4477AA"),
        (axes[1], atom_values, "Atom Gamma", "#d73027"),
    )
    for ax, values, title, color in panels:
        weights = np.ones_like(values) / values.size
        ax.hist(values, bins=50, weights=weights, color=color, alpha=0.85, edgecolor="none")
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Probability")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
