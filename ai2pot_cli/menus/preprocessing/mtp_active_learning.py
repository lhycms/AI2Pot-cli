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
DEFAULT_MAXVOL_INV: str = "./maxvol_matrix_inv.pt"
DEFAULT_EXPLORE: str = "./explore.xyz"
DEFAULT_GAMMA_DIR: str = "./gamma_analysis"
DEFAULT_MAXITER: int = 20000
DEFAULT_TOL: float = 1e-4
DEFAULT_RANK_TOL: float = 1e-12

_MODE_LABELS = {
    "local": "Atom-local environment",
    "global": "Whole structure",
}

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


def _ask_mode() -> str:
    print(" Select analysis mode:")
    print("   1)  Atom-local environment (default)")
    print("   2)  Whole structure")
    while True:
        choice = input(" Mode [1]: ").strip().lower()
        if choice in ("", "1", "local"):
            return "local"
        if choice in ("2", "global"):
            return "global"
        print_warning("Invalid mode; please enter 1 or 2.")


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

    mode = _ask_mode()

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

    output_path = DEFAULT_OUTPUT
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    active_set_path = DEFAULT_ACTIVE_SET
    os.makedirs(os.path.dirname(os.path.abspath(active_set_path)), exist_ok=True)

    device = _get_device()
    print_section(f"Running on {device.upper()}")
    print_kv("Mode", _MODE_LABELS[mode])
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
        if mode == "local":
            if not hasattr(generator, "generate_maxvol_matrix_local"):
                print_error(
                    "LinearMtpMaxvolGenerator.generate_maxvol_matrix_local() is not available; "
                    "please update AI2Pot to a version that provides it."
                )
                return
            print_success("Initializing MaxVol matrix ...")
            maxvol_structure_id, maxvol_atom_id, maxvol_matrix, maxvol_matrix_inv = \
                generator.generate_maxvol_matrix_local(
                    maxiter=DEFAULT_MAXITER,
                    tol=DEFAULT_TOL,
                    rank_tol=DEFAULT_RANK_TOL,
                )
        else:
            if not hasattr(generator, "generate_maxvol_matrix_global"):
                print_error(
                    "LinearMtpMaxvolGenerator.generate_maxvol_matrix_global() is not available; "
                    "please update AI2Pot to a version that provides it."
                )
                return
            print_success("Initializing MaxVol matrix ...")
            maxvol_atom_id = None
            maxvol_structure_id, maxvol_matrix, maxvol_matrix_inv = \
                generator.generate_maxvol_matrix_global(
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
            "mode": mode,
            "maxvol_structure_id": maxvol_structure_id.cpu(),
            "maxvol_atom_id": maxvol_atom_id.cpu() if maxvol_atom_id is not None else None,
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
    maxvol_matrix_inv_path = DEFAULT_MAXVOL_INV
    torch.save(maxvol_matrix_inv.cpu(), maxvol_matrix_inv_path)
    try:
        write(active_set_path, active_structures, format="extxyz")
    except Exception as exc:
        print_error(f"Failed to write active set: {exc}")
        return

    num_parameters = int(maxvol_matrix.size(0))

    print_section("Maxvol Matrix Initialized Successfully")
    print_kv("Mode", _MODE_LABELS[mode])
    print_kv("Output File", os.path.abspath(output_path))
    print_kv("Inverse Matrix", os.path.abspath(maxvol_matrix_inv_path))
    print_kv("Active Set", os.path.abspath(active_set_path))
    print_kv("Matrix Shape", f"{num_parameters} x {num_parameters}")
    print_kv("Selected Structures", str(len(selected_structure_ids)))
    if mode == "local":
        print_kv("Selected Atoms", str(num_parameters))
    print_kv("Tolerance", f"{DEFAULT_TOL:g}")
    print_sep()
    print()

    sys.exit(0)


def analyze_gamma():
    """Compute extrapolation levels (gamma) for a new candidate set."""
    print_section("MTP Active Learning -- Analyze Extrapolation Level (Gamma)")

    mode = _ask_mode()

    checkpoint_path = _ask_checkpoint_path()
    if not checkpoint_path:
        return

    maxvol_inv_path = input(
        f" Maxvol matrix inv path (.pt) [default: {DEFAULT_MAXVOL_INV}]: "
    ).strip() or DEFAULT_MAXVOL_INV
    if not os.path.isfile(maxvol_inv_path):
        print_error(f"File not found: {maxvol_inv_path}")
        return

    explore_path = input(f" Explore set path (.xyz) [default: {DEFAULT_EXPLORE}]: ").strip() or DEFAULT_EXPLORE
    if not os.path.isfile(explore_path):
        print_error(f"File not found: {explore_path}")
        return

    output_dir = DEFAULT_GAMMA_DIR
    os.makedirs(output_dir, exist_ok=True)

    device = _get_device()
    print_section(f"Running on {device.upper()}")
    print_kv("Mode", _MODE_LABELS[mode])
    print_kv("Checkpoint", checkpoint_path)
    print_kv("Maxvol Matrix Inv", maxvol_inv_path)
    print_kv("Explore Set", explore_path)
    print_sep()

    try:
        import torch

        maxvol_matrix_inv = torch.load(
            maxvol_inv_path, map_location="cpu", weights_only=False
        ).to(torch.device(device))
        pbc_xyz = [True, True, True]

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
        if mode == "local":
            if not hasattr(generator, "get_gamma_local"):
                print_error(
                    "LinearMtpMaxvolGenerator.get_gamma_local() is not available; "
                    "please update AI2Pot to a version that provides it."
                )
                return
            print_success("Computing gamma ...")
            structure_ids, atom_ids, gamma_structures, gamma_atoms = generator.get_gamma_local(
                maxvol_matrix_inv=maxvol_matrix_inv,
                explore_path=explore_path,
            )
            structure_ids = structure_ids.cpu()
            atom_ids = atom_ids.cpu()
            gamma_structures = gamma_structures.cpu()
            gamma_atoms = gamma_atoms.cpu()
            num_atoms_per_structure = torch.bincount(
                structure_ids, minlength=gamma_structures.numel()
            )
        else:
            if not hasattr(generator, "get_gamma_global"):
                print_error(
                    "LinearMtpMaxvolGenerator.get_gamma_global() is not available; "
                    "please update AI2Pot to a version that provides it."
                )
                return
            print_success("Computing gamma ...")
            structure_ids, gamma_structures = generator.get_gamma_global(
                maxvol_matrix_inv=maxvol_matrix_inv,
                explore_path=explore_path,
            )
            structure_ids = structure_ids.cpu()
            gamma_structures = gamma_structures.cpu()
            atom_ids = None
            gamma_atoms = None
            num_atoms_per_structure = None
    except Exception as exc:
        print_error(str(exc))
        return

    difference_data = None
    difference_skip_reason = None
    try:
        difference_data = _compute_difference_data(
            checkpoint_path,
            explore_path,
            device,
            mode,
            structure_ids=structure_ids,
            gamma_structures=gamma_structures,
            atom_ids=atom_ids,
            gamma_atoms=gamma_atoms,
        )
        if difference_data is None:
            difference_skip_reason = "skipped (no labels in explore set)"
    except Exception as exc:
        print_error(f"Failed to analyze prediction differences: {exc}")
        difference_skip_reason = "skipped (analysis failed)"

    if difference_data is not None:
        try:
            _save_ef_npy(difference_data, output_dir)
        except Exception as exc:
            print_error(f"Failed to save E/F predictions: {exc}")

    gamma_structure_path = os.path.join(output_dir, "gamma_structure.pt")
    histogram_path = os.path.join(output_dir, "gamma_histogram.png")

    metadata = {
        "mode": mode,
        "checkpoint_path": checkpoint_path,
        "maxvol_inv_path": maxvol_inv_path,
        "explore_path": explore_path,
        "pbc_xyz": pbc_xyz,
    }
    gamma_structure_payload = {
        "gamma_structures": gamma_structures,
        **metadata,
    }
    if mode == "local":
        gamma_structure_payload["num_atoms_per_structure"] = num_atoms_per_structure
    torch.save(gamma_structure_payload, gamma_structure_path)

    gamma_atoms_path = None
    if mode == "local":
        gamma_atoms_path = os.path.join(output_dir, "gamma_atoms.pt")
        torch.save(
            {
                "gamma_atoms": gamma_atoms,
                "structure_ids": structure_ids,
                "atom_ids": atom_ids,
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

    try:
        if difference_data is not None:
            panels = []
            if mode == "local" and "force_gamma" in difference_data:
                panels.append({
                    "gamma": difference_data["force_gamma"],
                    "difference": difference_data["force_diff"],
                    "xlabel": "Atom Gamma",
                    "ylabel": "|F_ML - F_DFT| (eV/A)",
                    "title": "Atom Gamma vs. Force Difference",
                })
            if "energy_gamma" in difference_data:
                panels.append({
                    "gamma": difference_data["energy_gamma"],
                    "difference": difference_data["energy_diff"],
                    "xlabel": "Structure Gamma",
                    "ylabel": "|E_ML - E_DFT| (meV/atom)",
                    "title": "Structure Gamma vs. Energy Difference",
                })
            if panels:
                _plot_gamma_scatter(panels, os.path.join(output_dir, "gamma_scatter.png"))
    except Exception as exc:
        print_error(f"Failed to plot gamma difference figures: {exc}")
        return

    print_section("Gamma Analysis Generated Successfully")
    print_kv("Mode", _MODE_LABELS[mode])
    print_kv("Output Dir", os.path.abspath(output_dir))
    print_kv("Structure Gamma", os.path.abspath(gamma_structure_path))
    if gamma_atoms_path is not None:
        print_kv("Atom Gamma", os.path.abspath(gamma_atoms_path))
    print_kv("Histogram", os.path.abspath(histogram_path))
    if difference_data is None:
        print_kv("Error Plots", difference_skip_reason or "skipped")
    print()
    print(f"  {'Structures':<22}: {gamma_structures.numel()}")
    if gamma_atoms is not None:
        print(f"  {'Atoms':<22}: {gamma_atoms.numel()}")
    print(f"  {'Max Gamma (Structure)':<22}: {gamma_structures.max().item():.4f}")
    print(f"  {'Mean Gamma (Structure)':<22}: {gamma_structures.mean().item():.4f}")
    if gamma_atoms is not None:
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

    panels = [(gamma_structures.detach().cpu().numpy(), "Structure Gamma", "#4477AA")]
    if gamma_atoms is not None:
        panels.append((gamma_atoms.detach().cpu().numpy(), "Atom Gamma", "#d73027"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.8), squeeze=False)
    for ax, (values, title, color) in zip(axes[0], panels):
        counts, _, _ = ax.hist(values, bins=50, color=color, alpha=0.85, edgecolor="none")
        # Rare high-gamma bins are only a few counts per bin; a log y-axis keeps
        # them visible instead of flattening them to invisible slivers.
        ax.set_yscale("log")
        max_count = max(int(counts.max()), 1)
        top = max_count * 3.0
        ax.set_yticks([10.0 ** exponent for exponent in range(int(np.log10(top)) + 1)])
        ax.set_ylim(bottom=0.1, top=top)
        vmin, vmax = float(values.min()), float(values.max())
        pad = 0.02 * (vmax - vmin if vmax > vmin else 1.0)
        xleft = max(0.0, vmin - pad)
        if vmax > 1.0:
            ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2)
            xleft = min(xleft, 1.0)
        ax.set_xlim(xleft, vmax + pad)
        num_above = int(np.count_nonzero(values > 1.0))
        fraction_above = 100.0 * num_above / values.size
        stats = (
            f"$\\gamma$ > 1: {num_above:,} / {values.size:,} ({fraction_above:.3g}%)\n"
            f"max $\\gamma$ = {values.max():.3f}"
        )
        ax.text(
            0.97, 0.95, stats,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=14,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.9),
        )
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _collect_reference_labels(atoms_list):
    """Read per-structure energies and per-atom forces from the explore ASE atoms."""
    import numpy as np

    num_structures = len(atoms_list)
    energies = np.full(num_structures, np.nan)
    has_energy = np.zeros(num_structures, dtype=bool)
    force_chunks = []
    has_force_chunks = []

    for index, atoms in enumerate(atoms_list):
        results = atoms.calc.results if atoms.calc is not None else {}
        if "energy" in results:
            energies[index] = float(results["energy"])
            has_energy[index] = True
        elif "free_energy" in results:
            energies[index] = float(results["free_energy"])
            has_energy[index] = True

        num_atoms = len(atoms)
        if "forces" in results:
            force_chunks.append(np.asarray(results["forces"], dtype=float).reshape(num_atoms, 3))
            has_force_chunks.append(np.ones(num_atoms, dtype=bool))
        else:
            force_chunks.append(np.zeros((num_atoms, 3), dtype=float))
            has_force_chunks.append(np.zeros(num_atoms, dtype=bool))

    forces = np.concatenate(force_chunks, axis=0) if force_chunks else np.zeros((0, 3))
    has_force = np.concatenate(has_force_chunks) if has_force_chunks else np.zeros(0, dtype=bool)
    return energies, has_energy, forces, has_force


def _build_ef_model(checkpoint_path: str, dataset_path: str, device: str):
    model_type = _detect_model_type(checkpoint_path)
    if model_type == "mtp":
        from ai2pot.models.mtp.linear_mtp_utils import LinearMtp4Extxyz
        return LinearMtp4Extxyz(checkpoint_path=checkpoint_path, testset_path=dataset_path, map_location=device)
    if model_type == "nnmtp":
        from ai2pot.models.mtp.nn_mtp_utils import NNMtp4Extxyz
        return NNMtp4Extxyz(checkpoint_path=checkpoint_path, testset_path=dataset_path, map_location=device)
    from ai2pot.models.nep.nep_utils import Nep4Extxyz
    return Nep4Extxyz(checkpoint_path=checkpoint_path, testset_path=dataset_path, map_location=device)


def _predict_ef_per_atom(ef_model):
    """Predict per-atom energy (eV/atom) and per-atom forces (eV/A) for the explore set."""
    import numpy as np
    import torch

    device = ef_model.lit_module.device
    model = ef_model.model
    energy_list = []
    force_list = []

    with torch.no_grad():
        for batch_data in ef_model.test_dataloader:
            binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                brcs_tensor, btypes_tensor, bnghost_tensor = \
                [t.to(device) for t in batch_data[:7]]

            if ef_model.fit_virial:
                betot_ml_tensor, bforce_ml_tensor, _ = model.predict_efv(
                    binum_tensor,
                    bilist_tensor,
                    bnumneigh_tensor,
                    bfirstneigh_tensor,
                    brcs_tensor,
                    btypes_tensor,
                    bnghost_tensor,
                )
            else:
                betot_ml_tensor, bforce_ml_tensor = model.predict_ef(
                    binum_tensor,
                    bilist_tensor,
                    bnumneigh_tensor,
                    bfirstneigh_tensor,
                    brcs_tensor,
                    btypes_tensor,
                    bnghost_tensor,
                )

            mask_for_force = torch.arange(
                bilist_tensor.size(1), device=device
            )[None, :] < binum_tensor[:, None]
            energy_list.append((betot_ml_tensor / binum_tensor).detach().cpu().numpy())
            force_list.append(bforce_ml_tensor[mask_for_force].detach().cpu().numpy())

    energies = np.concatenate(energy_list, axis=0) if energy_list else np.zeros(0)
    forces = np.concatenate(force_list, axis=0) if force_list else np.zeros((0, 3))
    return energies, forces


def _compute_difference_data(
    checkpoint_path: str,
    explore_path: str,
    device: str,
    mode: str,
    structure_ids,
    gamma_structures,
    atom_ids,
    gamma_atoms,
):
    """Build gamma-vs-error series; returns None when the explore set has no labels."""
    import numpy as np
    from ase.io import read as ase_read

    first_atoms = ase_read(explore_path, index=0)
    first_results = first_atoms.calc.results if first_atoms.calc is not None else {}
    has_labels = any(key in first_results for key in ("energy", "free_energy", "forces"))
    if not has_labels:
        return None

    ef_model = _build_ef_model(checkpoint_path, explore_path, device)
    ef_model.lit_module.eval()

    atoms_list = ef_model.test_dataloader.dataset.atoms_list
    num_structures = len(atoms_list)
    num_atoms = np.array([len(atoms) for atoms in atoms_list], dtype=int)
    e_dft, has_energy, f_dft, has_force = _collect_reference_labels(atoms_list)
    needs_force = mode == "local" and bool(has_force.any())
    if not bool(has_energy.any()) and not needs_force:
        return None

    e_ml, f_ml = _predict_ef_per_atom(ef_model)
    if e_ml.size != num_structures:
        raise ValueError(
            f"Predicted {e_ml.size} energies for {num_structures} structures."
        )
    e_dft_per_atom = e_dft / num_atoms

    # get_gamma_* returns gamma_structures in candidate-set order (the dataloader
    # does not shuffle); in local mode structure_ids is per atom, not per structure.
    gamma_structure_array = gamma_structures.detach().cpu().numpy()
    if gamma_structure_array.size != num_structures:
        raise ValueError(
            f"Got {gamma_structure_array.size} structure gammas "
            f"for {num_structures} structures."
        )

    data = {}
    if has_energy.any():
        data["energy_pred"] = e_ml
        data["energy_dft"] = e_dft_per_atom
        energy_mask = has_energy & np.isfinite(gamma_structure_array)
        if energy_mask.any():
            data["energy_gamma"] = gamma_structure_array[energy_mask]
            data["energy_diff"] = np.abs(e_ml - e_dft_per_atom)[energy_mask] * 1000.0

    if has_force.any():
        data["force_pred"] = f_ml
        data["force_dft"] = f_dft

    if needs_force:
        starts = np.r_[0, np.cumsum(num_atoms)[:-1]].astype(int)
        total_atoms = int(num_atoms.sum())
        if f_ml.shape[0] != total_atoms or f_dft.shape[0] != total_atoms:
            raise ValueError(
                f"Force count mismatch: predicted {f_ml.shape[0]}, "
                f"reference {f_dft.shape[0]}, atoms {total_atoms}."
            )
        structure_id_array = structure_ids.detach().cpu().numpy().astype(int)
        atom_id_array = atom_ids.detach().cpu().numpy().astype(int)
        gamma_atom_array = np.full(total_atoms, np.nan)
        gamma_atom_array[starts[structure_id_array] + atom_id_array] = \
            gamma_atoms.detach().cpu().numpy()
        force_mask = has_force & np.isfinite(gamma_atom_array)
        if force_mask.any():
            data["force_gamma"] = gamma_atom_array[force_mask]
            data["force_diff"] = np.linalg.norm(f_ml - f_dft, axis=1)[force_mask]

    return data or None


def _save_ef_npy(difference_data, output_dir: str):
    """Save per-atom E/F predictions and labels in the parity-plot naming scheme."""
    import numpy as np

    prefix = "explore_"
    for key in ("energy_dft", "energy_pred", "force_dft", "force_pred"):
        if key in difference_data:
            np.save(os.path.join(output_dir, f"{prefix}{key}.npy"), difference_data[key])


_GRADE_STYLES = [
    ("Interpolation", 0.0, 1.0, "#0072B2"),
    ("Low", 1.0, 2.0, "#E69F00"),
    ("High", 2.0, 10.0, "#D55E00"),
    ("Severe", 10.0, float("inf"), "#CC79A7"),
]


def _grade_indices(gamma):
    import numpy as np

    indices = np.zeros(gamma.shape, dtype=int)
    for index, (_, lower, upper, _) in enumerate(_GRADE_STYLES):
        indices[(gamma >= lower) & (gamma < upper)] = index
    return indices


def _draw_gamma_vs_difference(ax, panel):
    import numpy as np
    from matplotlib.lines import Line2D

    gamma = np.asarray(panel["gamma"], dtype=float)
    difference = np.asarray(panel["difference"], dtype=float)
    grade_indices = _grade_indices(gamma)

    xright = max(float(gamma.max()) * 1.05, 1.0)

    for index, (_, lower, upper, color) in enumerate(_GRADE_STYLES):
        left = lower
        right = min(upper, xright)
        if right > left:
            ax.axvspan(left, right, color=color, alpha=0.07, zorder=0)
    for bound in (1.0, 2.0, 10.0):
        if bound < xright:
            ax.axvline(bound, color="0.25", linestyle="--", linewidth=1.2, zorder=1)

    scatter_styles = [
        dict(s=8, alpha=0.35),
        dict(s=14, alpha=0.65),
        dict(s=40, alpha=0.95),
        dict(s=40, alpha=0.95),
    ]
    for index, (_, _, _, color) in enumerate(_GRADE_STYLES):
        mask = grade_indices == index
        if mask.any():
            ax.scatter(gamma[mask], difference[mask], c=color,
                       edgecolors="none", rasterized=True, zorder=2,
                       **scatter_styles[index])

    ax.set_xlim(0.0, xright)
    ax.set_xlabel(panel["xlabel"])
    ax.set_ylabel(panel["ylabel"])
    ax.set_title(panel["title"])

    use_log = difference.size > 0 and float(difference.min()) > 0.0
    if use_log:
        positive = difference[difference > 0.0]
        lower = float(positive.min())
        upper = float(positive.max())
        if lower == upper:
            lower *= 0.5
            upper *= 2.0
        ax.set_yscale("log")
        ax.set_ylim(lower * 0.5, upper * 2.0)
    else:
        lower = float(difference.min())
        upper = float(difference.max())
        pad = 0.05 * (upper - lower if upper > lower else 1.0)
        ax.set_ylim(max(0.0, lower - pad), upper + pad)
    ax.grid(axis="both", which="major", linestyle="--", linewidth=0.8,
            color="gray", alpha=0.6, zorder=0)

    present_grades = [i for i in range(len(_GRADE_STYLES))
                      if np.any(grade_indices == i)]
    handles = [
        Line2D([], [], linestyle="none", marker="o",
               markerfacecolor=_GRADE_STYLES[i][3], markeredgecolor="black",
               markeredgewidth=0.5, markersize=9)
        for i in present_grades
    ]
    labels = [_GRADE_STYLES[i][0] for i in present_grades]
    ax.legend(handles, labels, loc="lower right", framealpha=0.85)


def _plot_gamma_scatter(panels, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.5 * len(panels), 5.8), squeeze=False
    )
    for ax, panel in zip(axes[0], panels):
        _draw_gamma_vs_difference(ax, panel)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
