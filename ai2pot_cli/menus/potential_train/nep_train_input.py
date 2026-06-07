"""Generate nep_train.json from the built-in template."""

import os
import shutil

from ai2pot_cli.menu import show_generation_success

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "templates", "nep_train.json"
)


def generate_nep_input(output_path: str = "nep_train.json"):
    """Copy the NEP training template to *output_path*."""
    shutil.copy(_TEMPLATE_PATH, output_path)
    abs_path = os.path.abspath(output_path)
    show_generation_success(
        title="NEP Training Input Generated Successfully",
        output_path=abs_path,
        next_command=f"ai2pot-cli train --input {abs_path}",
    )
