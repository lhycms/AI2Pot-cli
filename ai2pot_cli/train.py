"""NEP training module -- reads a JSON/JSONC config and runs training."""

import json5
import os
from typing import Any, Dict, List

import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger

from ai2pot.data import ExtxyzDataset, ExtxyzDataModule
from ai2pot.models.potential_train import LitNep
from ai2pot.models.nep.nep_train_utils import NepDescriptorNormCallback
from ai2pot.models.potential_train_utils import EnergyShiftCallback


def _get_dtype(s: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if s not in mapping:
        raise ValueError(f"Unsupported torch_float_dtype: {s}. Choose from {list(mapping.keys())}.")
    return mapping[s]


def _resolve_type_map(type_map_cfg, trainset_path: str) -> List[int]:
    if type_map_cfg == "auto":
        return ExtxyzDataset.get_type_map(filename=trainset_path)
    if isinstance(type_map_cfg, list):
        return type_map_cfg
    raise ValueError(f"type_map must be 'auto' or a list of ints, got: {type_map_cfg}")


def _load_config(path: str) -> Dict[str, Any]:
    """Load a JSON/JSONC config file (supports // and /* */ comments)."""
    with open(path, "r") as f:
        return json5.load(f)


def run_train(config_path: str) -> None:
    config = _load_config(config_path)

    trainer_cfg: Dict[str, Any] = config["Trainer"]
    model_cfg: Dict[str, Any] = config["Model"]
    dataset_cfg: Dict[str, Any] = config["Dataset"]

    # --- Global settings ---
    seed: int = trainer_cfg.get("seed", 42)
    num_threads: int = trainer_cfg.get("num_threads", 16)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)

    dtype: torch.dtype = _get_dtype(dataset_cfg.get("torch_float_dtype", "float32"))

    # --- Resolve type_map ---
    trainset_path: str = dataset_cfg["trainset_path"]
    type_map: List[int] = _resolve_type_map(model_cfg["type_map"], trainset_path)

    # --- Build Model ---
    fit_virial: bool = model_cfg.get("fit_virial", dataset_cfg.get("has_virial", False))
    lit_nep = LitNep(
        type_map=type_map,
        umax_num_neigh_atoms=model_cfg["umax_num_neigh_atoms"],
        fit_virial=fit_virial,
        n_radial_basis=model_cfg["n_radial_basis"],
        n_angular_basis=model_cfg["n_angular_basis"],
        l_max=model_cfg["l_max"],
        chebyshev_size=model_cfg["chebyshev_size"],
        num_neurons=model_cfg["num_neurons"],
        rmax_radial=model_cfg["rmax_radial"],
        rmax_angular=model_cfg["rmax_angular"],
        zbl_rmax=model_cfg.get("zbl_rmax", 0.0),
        zbl_rmin=model_cfg.get("zbl_rmin", 0.0),
        lr_start=model_cfg["lr_start"],
        lr_end=model_cfg["lr_end"],
        e_wgt_start=model_cfg["e_wgt_start"],
        e_wgt_end=model_cfg["e_wgt_end"],
        f_wgt_start=model_cfg["f_wgt_start"],
        f_wgt_end=model_cfg["f_wgt_end"],
        v_wgt_start=model_cfg["v_wgt_start"],
        v_wgt_end=model_cfg["v_wgt_end"],
        max_clip_norm=model_cfg.get("max_clip_norm", 10),
    ).to(dtype)

    # --- Build DataModule ---
    datamodule = ExtxyzDataModule(
        trainset_path=trainset_path,
        validset_path=dataset_cfg["validset_path"],
        testset_path=dataset_cfg.get("testset_path"),
        predict_path=dataset_cfg.get("predict_path"),
        batch_size=dataset_cfg["batch_size"],
        rcut=dataset_cfg["rcut"],
        umax_num_neigh_atoms=dataset_cfg["umax_num_neigh_atoms"],
        pbc_xyz=dataset_cfg["pbc_xyz"],
        sort=dataset_cfg.get("sort", False),
        torch_float_dtype=dtype,
        has_virial=dataset_cfg.get("has_virial", False),
    )

    # --- Build Callbacks ---
    callbacks = []
    if trainer_cfg.get("enable_descriptor_norm", True):
        callbacks.append(NepDescriptorNormCallback())
    if trainer_cfg.get("enable_energy_shift", False):
        callbacks.append(EnergyShiftCallback())

    # --- Build Trainer ---
    csv_logger = CSVLogger(save_dir=trainer_cfg.get("save_dir", "lightning_logs"))
    trainer = L.Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 0),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 1),
        logger=csv_logger,
        callbacks=callbacks,
    )

    # --- Run ---
    trainer.fit(model=lit_nep, datamodule=datamodule)
