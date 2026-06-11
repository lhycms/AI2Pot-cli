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
    ("Preprocessing", [
        (1,  "Convert Dataset"),
        (2,  "Standardize ExtXYZ"),
        (3,  "Analyse Dataset"),
        (4,  "MTP Active Learning"),
        (5,  "NEP Active Learning"),
    ]),
    ("Potential Training Input", [
        (11, "MTP Training Input"),
        (12, "NEP Training Input"),
    ]),
    ("Postprocessing", [
        (21, "Evaluate Potential"),
        (22, "Plot E/F/V Parity"),
        (23, "Plot Learning Curve"),
        (24, "Export Predictions"),
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

        if choice == 0:
            print_success("Bye.")
            sys.exit(0)
        
        ### Sections
        # --- Preprocessing ---
        elif choice == 1:
            print(" -> Convert Dataset (not yet implemented)\n")
        elif choice == 2:
            extxyz_path = input(" ExtXYZ file path: ").strip()
            if not extxyz_path:
                print_warning("No file path provided.")
                continue
            out_path = input(" Output path [default: overwrite]: ").strip()
            from ai2pot_cli.menus.preprocessing.standardize_extxyz import standardize_extxyz
            standardize_extxyz(extxyz_path, None if not out_path else out_path)
            sys.exit(0)
        elif choice == 3:
            extxyz_path = input(" ExtXYZ file path: ").strip()
            if not extxyz_path:
                print_warning("No file path provided.")
                continue
            rcut_str = input(" Cutoff radius (A) [default: 6.0]: ").strip()
            rcut = float(rcut_str) if rcut_str else 6.0
            from ai2pot_cli.menus.preprocessing.analyse_nblist import analyse_dataset
            analyse_dataset(extxyz_path, rcut)
            sys.exit(0)
        elif choice == 4:
            print(" -> MTP Active Learning (not yet implemented)\n")
        elif choice == 5:
            print(" -> NEP Active Learning (not yet implemented)\n")

        # --- Potential Training Input ---
        elif choice == 11:
            print(" -> MTP Training Input (not yet implemented)\n")
        elif choice == 12:
            from ai2pot_cli.menus.potential_train.nep_train_input import generate_nep_input
            generate_nep_input()
            sys.exit(0)

        # --- Postprocessing ---
        elif choice == 21:
            print(" -> Evaluate Potential (not yet implemented)\n")
        elif choice == 22:
            print(" -> Plot E/F/V Parity (not yet implemented)\n")
        elif choice == 23:
            print(" -> Plot Learning Curve (not yet implemented)\n")
        elif choice == 24:
            print(" -> Export Predictions (not yet implemented)\n")

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
