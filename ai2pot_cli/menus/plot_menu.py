import sys

from ai2pot_cli.menu import show_menu, get_choice, print_success, print_warning

PLOT_ITEMS = [
    ("Parity Plot",       "Prediction vs. reference parity plot"),
    ("Learning Curve",    "Training/validation loss curves"),
    ("Energy Landscape",  "Energy along a reaction path"),
]


def plot_menu():
    while True:
        show_menu("Plotting Tools", PLOT_ITEMS)
        choice = get_choice()
        if choice == 0:
            print_success("Bye.")
            sys.exit(0)
        elif choice == 9:
            return
        elif choice == 1:
            print(" -> Generating parity plot ...\n")
        elif choice == 2:
            print(" -> Generating learning curve ...\n")
        elif choice == 3:
            print(" -> Generating energy landscape ...\n")
        else:
            print_warning(f"Invalid option: {choice}")
