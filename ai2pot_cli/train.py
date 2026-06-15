"""Training module -- reads a JSON/JSONC config and runs NEP or MTP training."""

import json5
import os
import re
from typing import Any, Dict, List

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ai2pot.data import ExtxyzDataset, ExtxyzDataModule
from ai2pot.models.potential_train import LitNep, LitLinearMtp
from ai2pot.models.nep.nep_train_utils import NepDescriptorNormCallback
from ai2pot.models.mtp.linear_mtp_train_utils import LinearMtpDescriptorNormCallback
from ai2pot.models.potential_train_utils import EnergyShiftCallback

# Patch CosineAnnealingLR.load_state_dict to never restore T_max / eta_min
# from a checkpoint.  These must always reflect the current max_epochs so
# that extended training lands at the correct position on the cosine curve.
_orig_cosine_load = torch.optim.lr_scheduler.CosineAnnealingLR.load_state_dict


def _patched_cosine_load_state_dict(self, state_dict):
    state_dict.pop('T_max', None)
    state_dict.pop('eta_min', None)
    _orig_cosine_load(self, state_dict)


torch.optim.lr_scheduler.CosineAnnealingLR.load_state_dict = _patched_cosine_load_state_dict


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
    num_threads: int = int(os.environ.get("SLURM_CPUS_PER_TASK", trainer_cfg.get("num_threads", 16)))
    L.seed_everything(seed, workers=True)
    torch.set_num_threads(num_threads)

    dtype: torch.dtype = _get_dtype(dataset_cfg.get("torch_float_dtype", "float32"))

    # --- Resolve type_map ---
    trainset_path: str = dataset_cfg["trainset_path"]
    type_map: List[int] = _resolve_type_map(model_cfg["type_map"], trainset_path)

    # --- Detect model type ---
    is_mtp: bool = "mtp_level" in model_cfg
    fit_virial: bool = model_cfg.get("fit_virial", dataset_cfg.get("has_virial", False))

    # --- Build Model ---
    common_kwargs = dict(
        type_map=type_map,
        umax_num_neigh_atoms=model_cfg["umax_num_neigh_atoms"],
        fit_virial=fit_virial,
        chebyshev_size=model_cfg["chebyshev_size"],
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
        max_clip_norm=model_cfg.get("max_clip_norm", 10.0),
    )

    if is_mtp:
        lit_model = LitLinearMtp(
            mtp_level=model_cfg["mtp_level"],
            rmax=model_cfg["rmax"],
            rmin=model_cfg.get("rmin", 0.0),
            **common_kwargs,
        ).to(dtype)
    else:
        lit_model = LitNep(
            n_radial_basis=model_cfg["n_radial_basis"],
            n_angular_basis=model_cfg["n_angular_basis"],
            l_max=model_cfg["l_max"],
            num_neurons=model_cfg["num_neurons"],
            rmax_radial=model_cfg["rmax_radial"],
            rmax_angular=model_cfg["rmax_angular"],
            **common_kwargs,
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

    # --- Detect resume & build logger ---
    save_dir = trainer_cfg.get("save_dir", "./")
    resume_ckpt: str | None = trainer_cfg.get("resume_ckpt")
    is_resume = bool(resume_ckpt)

    if is_resume:
        # Parse version number from the checkpoint path so resumed training
        # stays in the same lightning_logs/version_X/ directory.
        m = re.search(r"version_(\d+)", resume_ckpt)
        ver = int(m.group(1)) if m else None
        csv_logger = CSVLogger(save_dir=save_dir, version=ver)
    else:
        csv_logger = CSVLogger(save_dir=save_dir)

    ckpt_dir = os.path.join(csv_logger.log_dir, "checkpoints")

    # CSVLogger opens metrics.csv in 'w' mode to write the header on first
    # log, which would overwrite history when resuming into the same version.
    # Rename the old file so it's preserved, then merge back after training.
    metrics_path = os.path.join(csv_logger.log_dir, "metrics.csv")
    metrics_prev = None
    if is_resume and os.path.isfile(metrics_path):
        metrics_prev = metrics_path + ".prev"
        os.rename(metrics_path, metrics_prev)

    # --- Build Callbacks ---
    callbacks = []
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=3,
        monitor="train/mse",
        mode="min",
        every_n_epochs=1,
        save_last=True,
        save_on_train_epoch_end=True,
    ))

    if not is_resume:
        if trainer_cfg.get("enable_descriptor_norm", True):
            if is_mtp:
                callbacks.append(LinearMtpDescriptorNormCallback())
            else:
                callbacks.append(NepDescriptorNormCallback())
        if trainer_cfg.get("enable_energy_shift", False):
            callbacks.append(EnergyShiftCallback())

    # --- Build Trainer ---
    trainer = L.Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 0),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 500),
        enable_progress_bar=trainer_cfg.get("enable_progress_bar", False),
        logger=csv_logger,
        callbacks=callbacks,
    )

    # --- Run ---
    trainer.fit(model=lit_model, datamodule=datamodule, ckpt_path=resume_ckpt)

    # --- Prepend old metrics on resume (CSVLogger overwrites the header) ---
    if metrics_prev is not None:
        with open(metrics_prev, 'r') as f:
            old = f.read()
        with open(metrics_path, 'r') as f:
            new = f.read()
        old_lines = [l for l in old.strip().split('\n') if l]
        new_lines = [l for l in new.strip().split('\n') if l]
        # Both have the same header; skip the duplicate from the new file.
        merged = '\n'.join(old_lines + new_lines[1:]) + '\n'
        with open(metrics_path, 'w') as f:
            f.write(merged)
        os.remove(metrics_prev)
