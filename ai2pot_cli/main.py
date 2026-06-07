"""AI2Pot CLI -- Main entry point."""

import sys
from typing import (List, Optional, Tuple)

from ai2pot_cli.menu import (
    show_banner,
    show_main_menu,
    get_choice)

VERSION: str = "0.1.0"

MAIN_SECTIONS = [
    ("Preprocessing", [
        (1,  "Convert Dataset"),
        (2,  "MTP Active Learning"),
        (3,  "NEP Active Learning"),
    ]),
    ("Potential Training Input", [
        (11, "MTP Training Input"),
        (12, "NEP Training Input")
    ]),
    ("Postprocessing", [
        (21, "Evaluate Potential"),
        (22, "Plot E/F/V Parity"),
        (23, "Plot Learning Curve"),
        (24, "Export Predictions")
    ]),
    ("MD Utilities", [
        (91, "Doctor"),
        (92, "Show Examples"),
        (93, "Print Version"),
    ]),
]

MAIN_FOOTER: List[Tuple[int, str]] = [(0, "Quit")]


def main():
    show_banner(version=VERSION)
    while True:
        show_main_menu(MAIN_SECTIONS, MAIN_FOOTER)
        choice = get_choice()

        if choice == 0:
            print(" Bye.")
            sys.exit(0)

        # --- Preprocessing ---
        elif choice == 1:
            print(" -> Convert Dataset (not yet implemented)\n")
        elif choice == 2:
            print(" -> Split Dataset (not yet implemented)\n")
        elif choice == 3:
            print(" -> Prepare MTP Active Learning (not yet implemented)\n")
        elif choice == 4:
            print(" -> Prepare NEP Active Learning (not yet implemented)\n")

        # --- Potential Training Input ---
        elif choice == 11:
            print(" -> Train Linear MTP (not yet implemented)\n")
        elif choice == 12:
            print(" -> Train Neural MTP (not yet implemented)\n")
        elif choice == 13:
            print(" -> Train NEP/GNEP (not yet implemented)\n")
        elif choice == 14:
            print(" -> Resume Training (not yet implemented)\n")
        elif choice == 15:
            print(" -> Generate Train Config (not yet implemented)\n")

        # --- Postprocessing ---
        elif choice == 21:
            print(" -> Evaluate Potential (not yet implemented)\n")
        elif choice == 22:
            print(" -> Plot E/F/V Parity (not yet implemented)\n")
        elif choice == 23:
            print(" -> Plot Learning Curve (not yet implemented)\n")
        elif choice == 24:
            print(" -> Export Predictions (not yet implemented)\n")
        elif choice == 25:
            print(" -> Analyze MD Trajectory (not yet implemented)\n")

        # --- MD Utilities ---
        elif choice == 91:
            print(" -> Doctor (not yet implemented)\n")
        elif choice == 92:
            print(" -> Show Examples (not yet implemented)\n")
        elif choice == 93:
            print(f" AI2Pot-CLI  Version: {VERSION}\n")

        else:
            print(f" Invalid option: {choice}\n")
