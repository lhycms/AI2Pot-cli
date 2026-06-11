import sys

from ai2pot_cli.menu import show_menu, get_choice, print_success, print_warning

EVAL_ITEMS = [
    ("Evaluate Model",  "Run evaluation on a trained potential"),
    ("Compare Models",  "Compare multiple trained potentials"),
]


def eval_menu():
    while True:
        show_menu("Evaluate Potential", EVAL_ITEMS)
        choice = get_choice()
        if choice == 0:
            print_success("Bye.")
            sys.exit(0)
        elif choice == 9:
            return
        elif choice == 1:
            print(" -> Running evaluation ...\n")
        elif choice == 2:
            print(" -> Comparing models ...\n")
        else:
            print_warning(f"Invalid option: {choice}")
