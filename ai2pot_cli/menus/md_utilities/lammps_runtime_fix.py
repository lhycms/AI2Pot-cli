"""94) LAMMPS Runtime Fix.

Make a CUDA-built lmp_mpi_AI2Pot start on a node without a GPU by pointing
LD_LIBRARY_PATH at the CUDA runtime libraries shipped with pip PyTorch.
"""

import glob
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from ai2pot_cli.menu import (
    print_section, print_kv, print_sep,
    print_success, print_warning, print_error,
)

BINARY_NAME = "lmp_mpi_AI2Pot"
DEFAULT_ENV = "ai2pot_env"
CONDA_BASES = [
    "~/miniconda3",
    "~/anaconda3",
    "~/miniforge3",
    "~/mambaforge",
    "/opt/conda",
]
CUDA_LIB_PREFIXES = (
    "libcuda", "libcudart", "libcublas", "libcufft", "libcurand",
    "libcusolver", "libcusparse", "libcudnn", "libnccl", "libnv",
    "libnpp", "libcupti", "libcufile",
)

# (kind, label, directory)
LibDir = Tuple[str, str, str]


def _find_binary() -> Optional[str]:
    """Search $AI2POT_LMP, PATH, and common build locations."""
    candidates: List[str] = []
    env_lmp = os.environ.get("AI2POT_LMP")
    if env_lmp:
        candidates.append(os.path.expanduser(env_lmp))
    candidates.append(shutil.which(BINARY_NAME) or "")

    cwd = os.getcwd()
    candidates.extend([
        os.path.join(cwd, BINARY_NAME),
        os.path.join(cwd, "src", BINARY_NAME),
        os.path.join(os.path.expanduser("~"), "lammps", "src", BINARY_NAME),
        os.path.join(os.path.expanduser("~"), "lammps", BINARY_NAME),
        os.path.join(os.path.expanduser("~"), BINARY_NAME),
    ])
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidates.append(os.path.join(prefix, "bin", BINARY_NAME))

    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return os.path.abspath(path)
    return None


def _ask_binary() -> Optional[str]:
    """Return a usable binary path, prompting the user if auto-detection fails."""
    path = _find_binary()
    if path:
        print_success(f"Found: {path}")
        return path

    print_warning(f"{BINARY_NAME} not found automatically.")
    raw = input(f"  {'Binary path':<18}: ").strip()
    if not raw:
        print_error("A LAMMPS binary is required.")
        return None

    path = os.path.abspath(os.path.expanduser(raw))
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        print_error(f"Not an executable file: {path}")
        return None
    return path


def _ldd_missing(binary: str) -> Optional[List[str]]:
    """Return the names of libraries ldd resolves to 'not found'."""
    if shutil.which("ldd") is None:
        print_error("ldd is not available on this platform (Linux is required).")
        return None

    result = subprocess.run(["ldd", binary], capture_output=True, text=True)
    if result.returncode != 0 and not result.stdout.strip():
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        print_error(f"ldd failed: {detail}")
        return None

    missing = set()
    for line in result.stdout.splitlines():
        if "=>" not in line:
            continue
        name, target = line.split("=>", 1)
        if target.strip().startswith("not found"):
            missing.add(name.strip())
    return sorted(missing)


def _torch_info() -> Dict[str, object]:
    """Return torch version, CUDA availability, and install dir; 'error' on failure."""
    try:
        import torch
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "version": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "dir": os.path.dirname(os.path.abspath(torch.__file__)),
    }


def _library_dirs(torch_dir: Optional[str]) -> List[LibDir]:
    """Ordered candidate dirs: torch libs, pip NVIDIA wheels, then the CUDA toolkit."""
    dirs: List[LibDir] = []

    if torch_dir:
        torch_lib = os.path.join(torch_dir, "lib")
        if os.path.isdir(torch_lib):
            dirs.append(("torch", "Torch lib", torch_lib))
        nvidia_root = os.path.join(os.path.dirname(torch_dir), "nvidia")
        for lib_dir in sorted(glob.glob(os.path.join(nvidia_root, "*", "lib"))):
            if os.path.isdir(lib_dir):
                wheel = os.path.basename(os.path.dirname(lib_dir)).replace("_", "-")
                dirs.append(("nvidia", f"pip {wheel}", lib_dir))

    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        for sub in ("lib64", "lib"):
            cuda_lib = os.path.join(os.path.expanduser(cuda_home), sub)
            if os.path.isdir(cuda_lib):
                dirs.append(("cuda", f"CUDA_HOME/{sub}", cuda_lib))
    for cuda_lib in sorted(glob.glob("/usr/local/cuda*/lib64")):
        if os.path.isdir(cuda_lib):
            label = cuda_lib.replace("/usr/local/", "").replace("/lib64", " lib64")
            dirs.append(("cuda", label, cuda_lib))

    return dirs


def _resolve(missing: List[str], dirs: List[LibDir]) -> Dict[str, Optional[str]]:
    """Map each missing library to the first candidate dir that provides it."""
    resolved: Dict[str, Optional[str]] = {}
    for lib in missing:
        resolved[lib] = None
        for _kind, _label, lib_dir in dirs:
            candidate = os.path.join(lib_dir, lib)
            if os.path.isfile(candidate):
                resolved[lib] = candidate
                break
    return resolved


def _used_dirs(resolved: Dict[str, Optional[str]], dirs: List[LibDir]) -> List[str]:
    """Candidate dirs that supplied at least one missing library, in priority order."""
    found = {os.path.dirname(p) for p in resolved.values() if p}
    return [lib_dir for _kind, _label, lib_dir in dirs if lib_dir in found]


def _is_cuda_lib(name: str) -> bool:
    return name.startswith(CUDA_LIB_PREFIXES) or "torch_cuda" in name or "c10_cuda" in name


def _export_line(used_dirs: List[str]) -> str:
    joined = ":".join(used_dirs)
    return f'export LD_LIBRARY_PATH="{joined}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"'


def _script_content(used_dirs: List[str]) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "# Generated by ai2pot-cli (94: LAMMPS Runtime Fix)\n"
        f"{_export_line(used_dirs)}\n"
    )


def _conda_prefix() -> Optional[str]:
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and os.path.isdir(prefix):
        return prefix
    for base in CONDA_BASES:
        candidate = os.path.join(os.path.expanduser(base), "envs", DEFAULT_ENV)
        if os.path.isdir(candidate):
            return candidate
    return None


def _write_activate_hook(used_dirs: List[str]) -> Optional[str]:
    """Write (after confirmation) a conda activate.d hook. Returns the path, or None."""
    prefix = _conda_prefix()
    if not prefix:
        print_error("No conda environment detected; activate.d hook skipped.")
        return None

    target = os.path.join(prefix, "etc", "conda", "activate.d", "ai2pot_lammps_env.sh")
    print_kv("Target", target)
    answer = input("  Write this file? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print_warning("Skipped activate.d hook.")
        return None

    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(_script_content(used_dirs))
    return target


def _write_env_file(used_dirs: List[str]) -> str:
    path = os.path.expanduser("~/.ai2pot/lmp_env.sh")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_script_content(used_dirs))
    return path


def _apply(used_dirs: List[str]):
    """Step 4/4: let the user pick how the environment is delivered."""
    print_section("Step 4/4: Apply Runtime Environment")

    print("   1)  Print export line (use before LAMMPS / copy into PBS script)")
    print("   2)  Write ~/.ai2pot/lmp_env.sh (source it before running)")
    print("   3)  Write conda env activate.d (persistent, needs confirmation)")
    print()
    raw = input("  Apply [1/2/3, comma-separated] [default: 1]: ").strip() or "1"

    choices = {tok.strip() for tok in raw.split(",") if tok.strip()}
    if not choices or not choices.issubset({"1", "2", "3"}):
        print_warning(f"Invalid selection: {raw}. Nothing applied.")
        _exit_done()

    if "1" in choices:
        print()
        print_kv("Export line", _export_line(used_dirs))
        print()
        print_warning("Run this export before launching LAMMPS, or copy the line into your PBS job script.")
        print_warning("An export only affects the current shell (login node); job scripts need their own copy.")

    if "2" in choices:
        path = _write_env_file(used_dirs)
        print()
        print_success(f"Environment file written: {path}")
        print_kv("Use", f"source {path}")

    if "3" in choices:
        print()
        target = _write_activate_hook(used_dirs)
        if target:
            print_success(f"activate.d hook written: {target}")

    _exit_done()


def _exit_done():
    print_sep()
    print()
    sys.exit(0)


def lammps_runtime_fix():
    """Four-step wizard for running a CUDA-built LAMMPS binary on a CPU node."""
    print_section("LAMMPS Runtime Fix")

    # --- 1/4. Locate the binary ---
    print_section(f"Step 1/4: Locate {BINARY_NAME}")
    binary = _ask_binary()
    if binary is None:
        _exit_done()

    # --- 2/4. Health check ---
    print_section("Step 2/4: Check Runtime Libraries")
    print_kv("Binary", binary)

    missing = _ldd_missing(binary)
    if missing is None:
        _exit_done()

    torch_info = _torch_info()
    if "error" in torch_info:
        print_warning(f"PyTorch unavailable: {torch_info['error']}")
    else:
        print_kv("PyTorch", str(torch_info["version"]))
        cuda_state = "available" if torch_info["cuda"] else "not available (CPU fallback is normal)"
        print_kv("torch.cuda", cuda_state)

    if not missing:
        print()
        print_success("ldd is clean — the binary is already OK.")
        _exit_done()

    print()
    print_kv("Missing libs", str(len(missing)))
    for lib in missing:
        print_error(lib)

    # --- 3/4. Search CUDA runtime libraries ---
    print_section("Step 3/4: Search CUDA Runtime Libraries")

    torch_dir = None if "error" in torch_info else str(torch_info["dir"])
    dirs = _library_dirs(torch_dir)
    if dirs:
        for _kind, label, lib_dir in dirs:
            print_kv(label, lib_dir)
    else:
        print_warning("No torch or CUDA library directories found.")

    resolved = _resolve(missing, dirs)
    used_dirs = _used_dirs(resolved, dirs)
    unresolved = [lib for lib, path in resolved.items() if path is None]
    has_cuda_dirs = any(kind in ("nvidia", "cuda") for kind, _label, _dir in dirs)

    if unresolved:
        print()
        print_error("Still missing after adding every candidate directory:")
        for lib in unresolved:
            print_error(lib)
        if any(lib.startswith("libcuda.so") for lib in unresolved):
            print_warning("libcuda.so.1 comes from the NVIDIA driver, not from pip wheels or the CUDA toolkit.")
            print_warning("Run on a GPU compute node that has the driver.")
        if any(_is_cuda_lib(lib) for lib in unresolved):
            if not has_cuda_dirs:
                print_error("This machine has no CUDA runtime libraries (no pip NVIDIA wheels, no CUDA toolkit).")
            print_error(f"The CUDA build of {BINARY_NAME} cannot start on this machine.")
            print_kv("Next step", "2) Install LAMMPS with AI2Pot (CPU build)")
        else:
            print_warning("Install the missing system libraries, then run 94) again.")
        _exit_done()

    print()
    for lib, path in resolved.items():
        print_success(f"{lib} -> {path}")

    # --- 4/4. Apply ---
    _apply(used_dirs)
