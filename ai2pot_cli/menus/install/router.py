"""Top-level sub-menu routing for Installation."""

import sys
from ai2pot_cli.menu import show_numbered_menu, get_choice, print_warning, print_success

from ai2pot_cli.commands import register

register(1,  "ai2pot_cli.menus.install.router",         "install_ai2pot_menu")
register(101, "ai2pot_cli.menus.install.install_pypi",   "pypi_install")
register(102, "ai2pot_cli.menus.install.install_source", "source_install_menu")
register(2,   "ai2pot_cli.menus.install.router",         "install_lammps_menu")


_MENU_ITEMS = [
    (101, "Install from PyPI",    "pip install ai2pot (from PyPI)"),
    (102, "Install from Source",  "Build from local AI2Pot source step-by-step"),
]


def install_ai2pot_menu():
    """Sub-menu for AI2Pot installation (choice 1)."""
    while True:
        show_numbered_menu("Install AI2Pot", _MENU_ITEMS)
        choice = get_choice()
        from ai2pot_cli.commands import dispatch
        if dispatch(choice):
            continue
        if choice == 101:
            from ai2pot_cli.menus.install.install_pypi import pypi_install
            pypi_install()
        elif choice == 102:
            from ai2pot_cli.menus.install.install_source import source_install_menu
            source_install_menu()
        elif choice == 9:
            return
        elif choice == 0:
            print_success("Bye.")
            sys.exit(0)
        else:
            print_warning(f"Invalid option: {choice}")


def install_lammps_menu():
    """Step menu for LAMMPS with AI2Pot installation (choice 2)."""
    from ai2pot_cli.menus.install.install_lammps import lammps_step_menu
    lammps_step_menu()
