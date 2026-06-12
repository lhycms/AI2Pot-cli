"""Generate mtp_train.jsonc from the built-in template."""
import os
import shutil

from ai2pot_cli.menu import show_generation_success

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "templates", "mtp_train.jsonc"
)


def generate_mtp_input(output_path: str = "mtp_train.jsonc"):
    """Copy the MTP training template (with comments) to *output_path*."""
    shutil.copy(_TEMPLATE_PATH, output_path)
    abs_path = os.path.abspath(output_path)
    show_generation_success(
        title="MTP Training Input Generated Successfully",
        output_path=abs_path,
        next_command=f"ai2pot-cli train --input {abs_path}",
    )
