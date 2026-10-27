"""Shared generation logic for the MTP/NEP/NNMTP training input menus.

The generated copy of the template is rewritten on the fly with the chosen
trainset path, batch size, and training duration. For step-based training the
epoch count is derived from the number of frames in the training set, since
AI2Pot's trainer is limited by max_epochs.
"""

import json
import os
import re

from ai2pot_cli.menu import print_error, print_success, print_warning, show_generation_success

DEFAULT_TRAINSET = "./train.xyz"
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 200


def count_extxyz_frames(path: str) -> int:
    """Count frames in an extxyz file from its atom-count/comment line pairs."""
    frames = 0
    with open(path) as fp:
        while True:
            header = fp.readline()
            while header and not header.strip():
                header = fp.readline()
            if not header:
                return frames
            try:
                natoms = int(header.strip())
            except ValueError:
                raise ValueError(f"bad extxyz frame header: {header.strip()[:50]!r}")
            if not fp.readline():
                raise ValueError("unexpected end of file after frame header")
            for _ in range(natoms):
                if not fp.readline():
                    raise ValueError("unexpected end of file inside a frame")
            frames += 1


def steps_to_epochs(total_steps: int, n_frames: int, batch_size: int) -> int:
    """Epochs needed for *total_steps* optimizer steps (last batch is kept)."""
    steps_per_epoch = (n_frames + batch_size - 1) // batch_size
    return (total_steps + steps_per_epoch - 1) // steps_per_epoch


def render_training_input(text: str, max_epochs: int,
                          trainset_path: str, batch_size: int) -> str:
    """Replace the duration/dataset entries while keeping the template comments."""
    literals = {
        "max_epochs": str(max_epochs),
        "trainset_path": json.dumps(trainset_path),
        "batch_size": str(batch_size),
    }
    for key, literal in literals.items():
        pattern = re.compile(rf'^(?P<indent>[ \t]*)"{key}":[ \t]*[^,\n]+,', re.MULTILINE)
        text, count = pattern.subn(
            lambda match, k=key, v=literal: f'{match.group("indent")}"{k}": {v},', text)
        if count != 1:
            raise ValueError(f'Expected exactly one "{key}" entry in the training template.')
    return text


def _ask_int(prompt: str, default: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print_warning(f"Invalid integer: {raw}")
            continue
        if value <= 0:
            print_warning("Please enter a positive integer.")
            continue
        return value


def _ask_trainset_path() -> str:
    while True:
        path = input(f" Trainset path [default: {DEFAULT_TRAINSET}]: ").strip() or DEFAULT_TRAINSET
        if os.path.isfile(path):
            return path
        print_error(f"File not found: {path}")


def generate_training_input(template_path: str, output_path: str, model_name: str) -> None:
    """Write a training config from *template_path* with the user's choices."""
    print(" Select training duration mode:")
    print("   1)  Epochs (default)")
    print("   2)  Total steps")
    total_steps = None
    while True:
        mode = input(" Mode [1]: ").strip().lower()
        if mode in ("", "1", "epoch", "epochs"):
            max_epochs = _ask_int(
                f" Number of training epochs [default: {DEFAULT_EPOCHS}]: ",
                DEFAULT_EPOCHS)
            break
        if mode in ("2", "step", "steps"):
            total_steps = _ask_int(" Total number of training steps: ")
            break
        print_warning("Invalid mode; please enter 1 or 2.")

    trainset_path = _ask_trainset_path()
    batch_size = _ask_int(f" Batch size [default: {DEFAULT_BATCH_SIZE}]: ", DEFAULT_BATCH_SIZE)

    if total_steps is not None:
        try:
            n_frames = count_extxyz_frames(trainset_path)
        except (OSError, ValueError) as exc:
            print_error(f"Cannot read training set: {exc}")
            return
        if n_frames == 0:
            print_error(f"No frames found in {trainset_path}")
            return
        steps_per_epoch = (n_frames + batch_size - 1) // batch_size
        max_epochs = steps_to_epochs(total_steps, n_frames, batch_size)
        print_success(f"{n_frames} frames, {steps_per_epoch} steps/epoch "
                      f"-> max_epochs = {max_epochs} ({max_epochs * steps_per_epoch} steps)")

    with open(template_path, "r") as fp:
        text = fp.read()
    text = render_training_input(text, max_epochs, trainset_path, batch_size)
    with open(output_path, "w") as fp:
        fp.write(text)

    abs_path = os.path.abspath(output_path)
    show_generation_success(
        title=f"{model_name} Training Input Generated Successfully",
        output_path=abs_path,
        next_command=f"ai2pot-cli train --input {abs_path}",
    )
