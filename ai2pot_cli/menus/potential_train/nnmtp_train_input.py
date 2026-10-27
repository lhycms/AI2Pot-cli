"""Generate nnmtp_train.jsonc from the built-in template."""
import os

from ai2pot_cli.menus.potential_train.train_input_common import generate_training_input

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "templates", "nnmtp_train.jsonc"
)


def generate_nnmtp_input(output_path: str = "nnmtp_train.jsonc"):
    """Write the NNMTP training template to *output_path* with the chosen duration mode."""
    generate_training_input(_TEMPLATE_PATH, output_path, "NNMTP")
