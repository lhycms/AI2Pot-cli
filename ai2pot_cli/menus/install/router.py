"""Top-level sub-menu routing for Installation."""

from ai2pot_cli.commands import register

register(1,   "ai2pot_cli.menus.install.build_wheel",    "build_wheel_menu")
register(101, "ai2pot_cli.menus.install.install_pypi",   "pypi_install")
register(102, "ai2pot_cli.menus.install.install_source", "source_install_menu")
register(2,   "ai2pot_cli.menus.install.router",         "install_lammps_menu")


def install_lammps_menu():
    """Step menu for LAMMPS with AI2Pot installation (choice 2)."""
    from ai2pot_cli.menus.install.install_lammps import lammps_step_menu
    lammps_step_menu()
