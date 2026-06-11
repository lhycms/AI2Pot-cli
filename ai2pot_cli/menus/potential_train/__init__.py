"""Potential Training Input -- submenu for generating training config files."""

import sys

from ai2pot_cli.menu import show_menu, get_choice, print_success, print_warning

TRAIN_INPUT_ITEMS = [
    ("MTP Training Input",  "Generate MTP training JSON config"),
    ("NEP Training Input",  "Generate NEP training JSON config"),
]


def potential_train_menu():
    while True:
        show_menu("Potential Training Input", TRAIN_INPUT_ITEMS)
        choice = get_choice()
        if choice == 0:
            print_success("Bye.")
            sys.exit(0)
        elif choice == 9:
            return
        elif choice == 1:
            print(" -> MTP Training Input (not yet implemented)\n")
        elif choice == 2:
            from ai2pot_cli.menus.potential_train.nep_train_input import generate_nep_input
            generate_nep_input()
        else:
            print_warning(f"Invalid option: {choice}")
