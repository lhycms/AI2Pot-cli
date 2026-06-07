import sys

from ai2pot_cli.menu import show_menu, get_choice

DOCTOR_ITEMS = [
    ("Check input.json",   "Validate the input configuration file"),
    ("Check Dependencies", "Verify required packages are installed"),
    ("Check Templates",    "Verify template files are present"),
]


def doctor_menu():
    while True:
        show_menu("Doctor", DOCTOR_ITEMS)
        choice = get_choice()
        if choice == 0:
            print(" Bye.")
            sys.exit(0)
        elif choice == 9:
            return
        elif choice == 1:
            print(" -> Checking input.json ...\n")
        elif choice == 2:
            print(" -> Checking dependencies ...\n")
        elif choice == 3:
            print(" -> Checking templates ...\n")
        else:
            print(f" Invalid option: {choice}\n")
