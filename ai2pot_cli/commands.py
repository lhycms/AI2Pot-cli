"""Global command registry — enables dispatching any command number from any menu level."""

import importlib

_COMMANDS = {}  # {num: (module_path, func_name)}


def register(num: int, module_path: str, func_name: str):
    """Register a command number with its lazy-loaded handler."""
    _COMMANDS[num] = (module_path, func_name)


def dispatch(num: int) -> bool:
    """Try to dispatch a command number. Returns True if dispatched."""
    if num not in _COMMANDS:
        return False
    mod_path, func_name = _COMMANDS[num]
    mod = importlib.import_module(mod_path)
    func = getattr(mod, func_name)
    func()
    return True


# ── eagerly import all command modules to trigger their register() calls ──
from ai2pot_cli.menus.install import router       # noqa: E402
from ai2pot_cli.menus.install import install_source  # noqa: E402
from ai2pot_cli.menus.install import install_lammps  # noqa: E402
from ai2pot_cli.menus.preprocessing import convert_dataset  # noqa: E402
