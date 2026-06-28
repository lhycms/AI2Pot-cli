"""AI2Pot CLI -- Main entry point."""

import argparse
import sys
from typing import List, Tuple

from ai2pot_cli.menu import (
    show_banner,
    show_main_menu,
    get_choice,
    print_warning,
    print_success,
)

VERSION: str = "0.1.0"

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
    ]),
    ("Potential Training Input", [
        (21, "MTP Training Input"),
        (22, "NEP Training Input"),
    ]),
    ("Postprocessing", [
        (31, "Plot E/F/V Parity"),
        (32, "Plot Learning Curve"),
        (33, "Plot Descriptor Projection"),
        (34, "Export TorchScript Model"),
    ]),
    ("MD Utilities", [
        (91, "Doctor"),
        (92, "Show Examples"),
        (93, "Print Version"),
    ]),
]

MAIN_FOOTER: List[Tuple[int, str]] = [(0, "Quit")]


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
            print(" -> Convert Dataset (not yet implemented)\n")
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
            print(" -> MTP Active Learning (not yet implemented)\n")
        elif choice == 15:
            print(" -> NEP Active Learning (not yet implemented)\n")

        # --- Potential Training Input ---
        elif choice == 21:
            from ai2pot_cli.menus.potential_train.mtp_train_input import generate_mtp_input
            generate_mtp_input()
            sys.exit(0)
        elif choice == 22:
            from ai2pot_cli.menus.potential_train.nep_train_input import generate_nep_input
            generate_nep_input()
            sys.exit(0)

        # --- Postprocessing ---
        elif choice == 31:
            checkpoint_path = input(" Checkpoint path (.ckpt): ").strip()
            if not checkpoint_path:
                print_warning("No checkpoint path provided.")
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
            checkpoint_path = input(" Checkpoint path (.ckpt): ").strip()
            if not checkpoint_path:
                print_warning("No checkpoint path provided.")
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
            checkpoint_path = input(" Checkpoint path (.ckpt): ").strip()
            if not checkpoint_path:
                print_warning("No checkpoint path provided.")
                continue
            output_path = input(" Output path [default: ./ai2pot_libtorch.pt]: ").strip() or "./ai2pot_libtorch.pt"
            from ai2pot_cli.menus.postprocessing.serialize_model import serialize_model
            serialize_model(checkpoint_path, output_path=output_path)
            sys.exit(0)

        # --- MD Utilities ---
        elif choice == 91:
            print(" -> Doctor (not yet implemented)\n")
        elif choice == 92:
            print(" -> Show Examples (not yet implemented)\n")
        elif choice == 93:
            print(f" AI2Pot-CLI  Version: {VERSION}\n")

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
