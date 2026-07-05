"""Convert Dataset sub-menu — DFT software output → extxyz conversion tools."""

import sys

from ai2pot_cli.menu import show_numbered_menu, get_choice, print_warning, print_success
from ai2pot_cli.commands import register

register(111, "ai2pot_cli.menus.preprocessing.converters.vasp2xyz",   "run_vasp2xyz")
register(112, "ai2pot_cli.menus.preprocessing.converters.cp2k2xyz",   "run_cp2k2xyz")
register(113, "ai2pot_cli.menus.preprocessing.converters.abacus2xyz", "run_abacus2xyz")
register(114, "ai2pot_cli.menus.preprocessing.converters.siesta2xyz", "run_siesta2xyz")
register(115, "ai2pot_cli.menus.preprocessing.converters.orca2xyz",   "run_orca2xyz")
register(116, "ai2pot_cli.menus.preprocessing.converters.dp2xyz",     "run_dp2xyz")
register(117, "ai2pot_cli.menus.preprocessing.converters.castep2exyz","run_castep2exyz")

_MENU_ITEMS = [
    (111, "VASP → extxyz",      "vasprun.xml → extxyz (ASE-based, recursive scan)"),
    (112, "CP2K → extxyz",      ".inp + .log → extxyz (energy/forces/stress/virial)"),
    (113, "ABACUS → extxyz",    "running_scf.log / abacus.json → extxyz"),
    (114, "SIESTA → extxyz",    "BASIS_ENTHALPY + .FA + STRUCT_OUT → extxyz"),
    (115, "ORCA → extxyz",      "Multi-frame .xyz + template.inp → run ORCA → extxyz"),
    (116, "DeepMD → extxyz",    "DeepMD-kit (type.raw + set.*/npy) → extxyz"),
    (117, "CASTEP → extxyz",    ".castep → extxyz (lattice/energy/stress/forces)"),
]


def convert_dataset_menu():
    """Sub-menu for Convert Dataset (choice 11)."""
    while True:
        show_numbered_menu("Convert Dataset -- DFT Output → extxyz", _MENU_ITEMS)
        choice = get_choice()

        from ai2pot_cli.commands import dispatch
        if dispatch(choice):
            continue

        if choice == 111:
            from ai2pot_cli.menus.preprocessing.converters.vasp2xyz import run_vasp2xyz
            run_vasp2xyz()
        elif choice == 112:
            from ai2pot_cli.menus.preprocessing.converters.cp2k2xyz import run_cp2k2xyz
            run_cp2k2xyz()
        elif choice == 113:
            from ai2pot_cli.menus.preprocessing.converters.abacus2xyz import run_abacus2xyz
            run_abacus2xyz()
        elif choice == 114:
            from ai2pot_cli.menus.preprocessing.converters.siesta2xyz import run_siesta2xyz
            run_siesta2xyz()
        elif choice == 115:
            from ai2pot_cli.menus.preprocessing.converters.orca2xyz import run_orca2xyz
            run_orca2xyz()
        elif choice == 116:
            from ai2pot_cli.menus.preprocessing.converters.dp2xyz import run_dp2xyz
            run_dp2xyz()
        elif choice == 117:
            from ai2pot_cli.menus.preprocessing.converters.castep2exyz import run_castep2exyz
            run_castep2exyz()
        elif choice == 9:
            return
        elif choice == 0:
            print_success("Bye.")
            sys.exit(0)
        else:
            print_warning(f"Invalid option: {choice}")
