"""MTP Active Learning -- build the initial MaxVol basis from a candidate set."""

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
DEFAULT_MAXITER: int = 20000
DEFAULT_TOL: float = 1e-4
DEFAULT_RANK_TOL: float = 1e-12

_MENU_ITEMS = [
    (141, "Initialize Maxvol Matrix", "Build the initial D-optimal basis from a labeled candidate set"),
]

register(141, "ai2pot_cli.menus.preprocessing.mtp_active_learning", "initialize_maxvol_matrix")


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

    device = _get_device()
    print_section(f"Running on {device.upper()}")

    try:
        model_type = _detect_model_type(checkpoint_path)
    except Exception as exc:
        print_error(str(exc))
        return
    if model_type != "mtp":
        print_error(f"Checkpoint is a {model_type.upper()} model; this tool requires an MTP checkpoint.")
        return

    print_kv("Checkpoint", checkpoint_path)
    print_kv("Candidate Set", candidate_set_path)
    print_kv("PBC", "T T T")

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
    active_set_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "active_set.xyz")
    try:
        from ase.io import read, write

        active_structures = read(candidate_set_path, index=selected_structure_ids)
        if len(active_structures) != len(selected_structure_ids):
            raise ValueError(
                f"Read {len(active_structures)} structures from the candidate set, "
                f"expected {len(selected_structure_ids)}."
            )
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
    try:
        write(active_set_path, active_structures, format="extxyz")
    except Exception as exc:
        print_error(f"Failed to write active set: {exc}")
        return

    num_parameters = int(maxvol_matrix.size(0))

    print_section("Maxvol Matrix Initialized Successfully")
    print_kv("Output File", os.path.abspath(output_path))
    print_kv("Active Set", os.path.abspath(active_set_path))
    print_kv("Matrix Shape", f"{num_parameters} x {num_parameters}")
    print_kv("Selected Structures", str(len(selected_structure_ids)))
    print_kv("Selected Atoms", str(num_parameters))
    print_kv("Tolerance", f"{DEFAULT_TOL:g}")
    print_sep()
    print()

    sys.exit(0)
