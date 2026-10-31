"""Predict E/F/V for an xyz dataset and save them as .npy files."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ai2pot_cli.menu import print_section, print_kv, print_sep, print_success

OUTPUT_DIR_NAME = "predict_results"


def _detect_model_type(checkpoint_path: str) -> str:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    if "n_radial_basis" in hp:
        return "nep"
    if "mtp_level" in hp and "num_neurons" in hp:
        return "nnmtp"
    if "mtp_level" in hp:
        return "mtp"
    raise ValueError(
        "Cannot detect model type from checkpoint. "
        "Expected 'mtp_level'/'num_neurons' or 'n_radial_basis' in hyper_parameters."
    )


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(checkpoint_path: str, dataset_path: str, device: str) -> Tuple[str, object]:
    model_type = _detect_model_type(checkpoint_path)
    if model_type == "mtp":
        from ai2pot.models.mtp.linear_mtp_utils import LinearMtp4Extxyz
        model = LinearMtp4Extxyz(checkpoint_path=checkpoint_path, testset_path=dataset_path, map_location=device)
    elif model_type == "nnmtp":
        from ai2pot.models.mtp.nn_mtp_utils import NNMtp4Extxyz
        model = NNMtp4Extxyz(checkpoint_path=checkpoint_path, testset_path=dataset_path, map_location=device)
    else:
        from ai2pot.models.nep.nep_utils import Nep4Extxyz
        model = Nep4Extxyz(checkpoint_path=checkpoint_path, testset_path=dataset_path, map_location=device)
    return model_type, model


@torch.no_grad()
def _predict_efv(model) -> Dict[str, np.ndarray]:
    """Batched prediction; energy/virial are per atom, forces per atom. DFT labels are not needed."""
    device = model.lit_module.device
    e_list: List[torch.Tensor] = []
    f_list: List[torch.Tensor] = []
    v_list: List[torch.Tensor] = []

    for batch_data in model.test_dataloader:
        binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
            brcs_tensor, btypes_tensor, bnghost_tensor = \
            [t.to(device) for t in batch_data[:7]]

        if model.fit_virial:
            betot_ml_tensor, bforce_ml_tensor, bvirial_ml_tensor = model.model.predict_efv(
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                bnghost_tensor)
        else:
            betot_ml_tensor, bforce_ml_tensor = model.model.predict_ef(
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                bnghost_tensor)

        mask_for_force: torch.Tensor = torch.arange(bilist_tensor.size(1),
                                                    device=device)[None, :] < binum_tensor[:, None]
        e_list.append((betot_ml_tensor / binum_tensor).detach().cpu().numpy())
        f_list.append(bforce_ml_tensor[mask_for_force].detach().cpu().numpy())
        if model.fit_virial:
            v_list.append((bvirial_ml_tensor / binum_tensor[:, None]).detach().cpu().numpy())

    data: Dict[str, np.ndarray] = {
        "energy": np.concatenate(e_list, axis=0),
        "force": np.concatenate(f_list, axis=0),
    }
    if model.fit_virial:
        data["virial"] = np.concatenate(v_list, axis=0)
    return data


def predict_efv(checkpoint_path: str, xyz_path: str, output_dir: Optional[str] = None):
    """Predict energy, forces and (if available) virial with an MTP/NEP/NNMTP checkpoint.

    Saves <xyz_stem>_energy.npy / _force.npy / _virial.npy into predict_results/.
    Energy and virial are per atom; forces are per atom (eV/atom, eV/A, eV/atom).
    """
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"File not found: {checkpoint_path}")
    if not os.path.isfile(xyz_path):
        raise ValueError(f"File not found: {xyz_path}")

    out_dir = output_dir or os.path.join(os.getcwd(), OUTPUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(xyz_path))[0]

    device = _get_device()
    print_section(f"Running on {device.upper()}")

    print_success(f"Loading model: {checkpoint_path}")
    model_type, model = _build_model(checkpoint_path, xyz_path, device)
    print_success(f"Loading dataset: {xyz_path} ({len(model.test_dataloader.dataset)} structures)")

    data = _predict_efv(model)

    file_paths: Dict[str, str] = {}
    for key in ("energy", "force", "virial"):
        if key in data:
            file_paths[key] = os.path.join(out_dir, f"{stem}_{key}.npy")
            np.save(file_paths[key], data[key])

    units = "Energy eV/atom, Force eV/A"
    if "virial" in data:
        units += ", Virial eV/atom"

    print_section("Prediction Completed Successfully")
    print_kv("Output Dir", out_dir)
    print_kv("Model Type", model_type.upper())
    print_kv("Structures", str(len(data["energy"])))
    print_kv("Energy File", os.path.abspath(file_paths["energy"]))
    print_kv("Force File", os.path.abspath(file_paths["force"]))
    if "virial" in data:
        print_kv("Virial File", os.path.abspath(file_paths["virial"]))
    print_kv("Units", units)
    print_sep()
    print()
