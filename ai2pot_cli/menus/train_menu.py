import sys

from ai2pot_cli.menu import show_menu, get_choice

TRAIN_ITEMS = [
    ("Linear MTP",  "Train a linear Moment Tensor Potential"),
    ("NN MTP",      "Train a neural-network MTP"),
    ("NEP",         "Train a Neuroevolution Potential"),
]


def train_menu():
    while True:
        show_menu("Train Potential", TRAIN_ITEMS)
        choice = get_choice()
        if choice == 0:
            print(" Bye.")
            sys.exit(0)
        elif choice == 9:
            return
        elif choice == 1:
            print(" -> Starting Linear MTP training ...\n")
        elif choice == 2:
            print(" -> Starting NN MTP training ...\n")
        elif choice == 3:
            print(" -> Starting NEP training ...\n")
        else:
            print(f" Invalid option: {choice}\n")
