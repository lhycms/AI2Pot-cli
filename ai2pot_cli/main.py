"""AI2Pot CLI -- Main entry point."""

import argparse
import os
import sys
from typing import List, Optional, Tuple

from ai2pot_cli.menu import (
    show_banner,
    show_main_menu,
    get_choice,
    print_warning,
    print_success,
    print_error,
)

VERSION: str = "1.2.2"

# Training runs write their checkpoints here (relative to the run's working directory)
DEFAULT_CHECKPOINT: str = "./lightning_logs/version_0/checkpoints/last.ckpt"

MAIN_SECTIONS = [
    ("Installation", [
        (1,  "Install AI2Pot"),
        (2,  "Install LAMMPS with AI2Pot"),
    ]),
    ("Preprocessing", [
        (11, "Convert Dataset"),
        (12, "Standardize ExtXYZ"),
        (13, "Analyse Dataset"),
        (14, "MTP Active Learning"),
        (15, "NEP Active Learning"),
        (16, "NNMTP Active Learning"),
    ]),
    ("Potential Training Input", [
        (21, "MTP Training Input"),
        (22, "NEP Training Input"),
        (23, "NNMTP Training Input"),
        (24, "ZBL Input"),
    ]),
    ("Postprocessing", [
        (31, "Plot E/F/V Parity"),
        (32, "Plot Learning Curve"),
        (33, "Plot Descriptor Projection"),
        (34, "Export TorchScript Model"),
        (35, "Plot ZBL"),
        (36, "Plot EOS"),
    ]),
    ("MD Utilities", [
        (91, "Doctor"),
        (92, "Show Examples"),
        (93, "Print Version"),
        (94, "LAMMPS Runtime Fix"),
    ]),
]

MAIN_FOOTER: List[Tuple[int, str]] = [(0, "Quit")]


def _ask_checkpoint_path() -> Optional[str]:
    """Prompt for a checkpoint path; empty input uses DEFAULT_CHECKPOINT. Returns None if the file is missing."""
    checkpoint_path = input(f" Checkpoint path (.ckpt) [default: {DEFAULT_CHECKPOINT}]: ").strip() or DEFAULT_CHECKPOINT
    if not os.path.isfile(checkpoint_path):
        print_error(f"File not found: {checkpoint_path}")
        return None
    return checkpoint_path


def _interactive_loop():
    """Run the VASPKIT-style interactive menu loop."""
    show_banner(version=VERSION)
    while True:
        show_main_menu(MAIN_SECTIONS, MAIN_FOOTER)
        choice = get_choice()

        from ai2pot_cli.commands import dispatch
        if dispatch(choice):
            continue

        if choice == 0:
            print_success("Bye.")
            sys.exit(0)
        
        ### Sections
        # --- Installation ---
        elif choice == 1:
            from ai2pot_cli.menus.install.router import install_ai2pot_menu
            install_ai2pot_menu()
        elif choice == 2:
            from ai2pot_cli.menus.install.router import install_lammps_menu
            install_lammps_menu()

        # --- Preprocessing ---
        elif choice == 11:
            from ai2pot_cli.menus.preprocessing.convert_dataset import convert_dataset_menu
            convert_dataset_menu()
        elif choice == 12:
            extxyz_path = input(" ExtXYZ file path: ").strip()
            if not extxyz_path:
                print_warning("No file path provided.")
                continue
            out_path = input(" Output path [default: overwrite]: ").strip()
            from ai2pot_cli.menus.preprocessing.standardize_extxyz import standardize_extxyz
            standardize_extxyz(extxyz_path, None if not out_path else out_path)
            sys.exit(0)
        elif choice == 13:
            extxyz_path = input(" ExtXYZ file path: ").strip()
            if not extxyz_path:
                print_warning("No file path provided.")
                continue
            rcut_str = input(" Cutoff radius (A) [default: 6.0]: ").strip()
            rcut = float(rcut_str) if rcut_str else 6.0
            from ai2pot_cli.menus.preprocessing.analyse_nblist import analyse_dataset
            analyse_dataset(extxyz_path, rcut)
            sys.exit(0)
        elif choice == 14:
            from ai2pot_cli.menus.preprocessing.mtp_active_learning import mtp_active_learning_menu
            mtp_active_learning_menu()
        elif choice == 15:
            print(" -> NEP Active Learning (not yet implemented)\n")
        elif choice == 16:
            print(" -> NNMTP Active Learning (not yet implemented)\n")

        # --- Potential Training Input ---
        elif choice == 21:
            from ai2pot_cli.menus.potential_train.mtp_train_input import generate_mtp_input
            generate_mtp_input()
            sys.exit(0)
        elif choice == 22:
            from ai2pot_cli.menus.potential_train.nep_train_input import generate_nep_input
            generate_nep_input()
            sys.exit(0)
        elif choice == 23:
            from ai2pot_cli.menus.potential_train.nnmtp_train_input import generate_nnmtp_input
            generate_nnmtp_input()
            sys.exit(0)
        elif choice == 24:
            trainset_path = input(" Trainset path (.xyz): ").strip()
            if not trainset_path:
                print_warning("No file path provided.")
                continue
            if not os.path.isfile(trainset_path):
                print_error(f"File not found: {trainset_path}")
                continue
            from ai2pot_cli.menus.potential_train.zbl_input import generate_zbl_input
            generate_zbl_input(trainset_path)
            sys.exit(0)

        # --- Postprocessing ---
        elif choice == 31:
            checkpoint_path = _ask_checkpoint_path()
            if checkpoint_path is None:
                continue
            trainset_path = input(" Trainset path (.xyz) [optional]: ").strip() or None
            testset_path = input(" Testset path (.xyz) [optional]: ").strip() or None
            if not trainset_path and not testset_path:
                print_warning("At least one of trainset or testset must be provided.")
                continue
            from ai2pot_cli.menus.postprocessing.plot_parity import plot_parity
            plot_parity(checkpoint_path, trainset_path=trainset_path, testset_path=testset_path)
            sys.exit(0)
        elif choice == 32:
            csv_path = input(" Metrics CSV path [metrics.csv]: ").strip() or "metrics.csv"
            from ai2pot_cli.menus.postprocessing.plot_trainlog import plot_trainlog
            plot_trainlog(csv_path)
            sys.exit(0)
        elif choice == 33:
            checkpoint_path = _ask_checkpoint_path()
            if checkpoint_path is None:
                continue
            trainset_path = input(" Trainset path (.xyz) [optional]: ").strip() or None
            testset_path = input(" Testset path (.xyz) [optional]: ").strip() or None
            if not trainset_path and not testset_path:
                print_warning("At least one of trainset or testset must be provided.")
                continue
            from ai2pot_cli.menus.postprocessing.plot_descriptors import plot_descriptor_projection
            plot_descriptor_projection(checkpoint_path, trainset_path=trainset_path, testset_path=testset_path)
            sys.exit(0)
        elif choice == 34:
            checkpoint_path = _ask_checkpoint_path()
            if checkpoint_path is None:
                continue
            output_path = input(" Output path [default: ./ai2pot_libtorch.pt]: ").strip() or "./ai2pot_libtorch.pt"
            from ai2pot_cli.menus.postprocessing.serialize_model import serialize_model
            serialize_model(checkpoint_path, output_path=output_path)
            sys.exit(0)
        elif choice == 35:
            checkpoint_path = _ask_checkpoint_path()
            if checkpoint_path is None:
                continue
            distance_range = input(" ZBL distance range (A) [default: 0.5-4.0]: ").strip()
            from ai2pot_cli.menus.postprocessing.plot_zbl import plot_zbl
            try:
                plot_zbl(checkpoint_path, distance_range=distance_range or None)
            except ValueError as exc:
                print_error(str(exc))
                continue
            sys.exit(0)
        elif choice == 36:
            checkpoint_path = _ask_checkpoint_path()
            if checkpoint_path is None:
                continue
            structure_path = input(" Reference structure path (POSCAR/xyz): ").strip()
            if not structure_path:
                print_warning("No structure path provided.")
                continue
            if not os.path.isfile(structure_path):
                print_error(f"File not found: {structure_path}")
                continue
            scale_range = input(" Lattice scale range [default: 0.90-1.10]: ").strip()
            n_points_str = input(" Number of points incl. original [default: 21]: ").strip()
            try:
                n_points = int(n_points_str) if n_points_str else None
            except ValueError:
                print_error(f"Invalid number of points: {n_points_str}")
                continue
            from ai2pot_cli.menus.postprocessing.plot_eos import plot_eos
            try:
                plot_eos(checkpoint_path, structure_path,
                         scale_range=scale_range or None, n_points=n_points)
            except ValueError as exc:
                print_error(str(exc))
                continue
            sys.exit(0)

        # --- MD Utilities ---
        elif choice == 91:
            print(" -> Doctor (not yet implemented)\n")
        elif choice == 92:
            print(" -> Show Examples (not yet implemented)\n")
        elif choice == 93:
            print(f" AI2Pot-CLI  Version: {VERSION}\n")
        elif choice == 94:
            from ai2pot_cli.menus.md_utilities.lammps_runtime_fix import lammps_runtime_fix
            lammps_runtime_fix()

        else:
            print_warning(f"Invalid option: {choice}")


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="ai2pot-cli",
        description="Official Command Line Interface for AI2Pot",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"AI2Pot-cli {VERSION}",
    )
    subparsers: argparse._SubParsersAction = parser.add_subparsers(dest="command")

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Run potential training")
    train_parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="CONFIG.jsonc",
        help="Path to training config (JSON or JSONC, e.g. nep_train.jsonc)",
    )

    # --- plot subcommand ---
    test_parser = subparsers.add_parser("test", help="Evaluate trained potential")

    # --- plot subcommand ---
    plot_parser = subparsers.add_parser("plot", help="Generate plots")

    # --- main ---
    args = parser.parse_args()
    if args.command == "train":
        from ai2pot_cli.train import run_train
        run_train(args.input)
    else:
        _interactive_loop()
