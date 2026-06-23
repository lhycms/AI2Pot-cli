"""Serialize trained model to TorchScript for LAMMPS simulation."""
import os

import torch

from ai2pot_cli.menu import print_section, print_kv, print_sep, print_error, print_success


def _detect_model_type(checkpoint_path: str) -> str:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    if "mtp_level" in hp:
        return "mtp"
    if "n_radial_basis" in hp:
        return "nep"
    raise ValueError(
        "Cannot detect model type from checkpoint. "
        "Expected 'mtp_level' or 'n_radial_basis' in hyper_parameters."
    )


def serialize_model(checkpoint_path: str, output_path: str = "./ai2pot_libtorch.pt"):
    """Serialize a trained model checkpoint to TorchScript format for LAMMPS.

    Args:
        checkpoint_path: Path to the .ckpt checkpoint file.
        output_path: Path for the output TorchScript .pt file.
    """
    model_type = _detect_model_type(checkpoint_path)

    print_section(f"Serializing {model_type.upper()} Model to TorchScript")
    print_kv("Checkpoint", checkpoint_path)
    print_kv("Output", os.path.abspath(output_path))

    if model_type == "mtp":
        from ai2pot.models.mtp.linear_mtp_utils import LinearMtpSerializer
        LinearMtpSerializer.serialize(ckpt_path=checkpoint_path, pt_path=output_path)
    else:
        from ai2pot.models.nep.nep_utils import NepSerializer
        NepSerializer.serialize(ckpt_path=checkpoint_path, pt_path=output_path)

    scripted_model = torch.jit.load(output_path, map_location="cpu")
    scripted_model.eval()
    methods = scripted_model._c._method_names()

    print_section("Serialization Completed Successfully")
    print_kv("Output File", os.path.abspath(output_path))
    print_kv("Model Type", model_type.upper())
    print_kv("Methods", ", ".join(methods))
    print_sep()
    print()
