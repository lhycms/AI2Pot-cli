"""Step-by-step AI2Pot source installation.

1021) Configure CUDA
1022) Install PyTorch
1023) Install AI2Pot
"""

import os
import shutil
import subprocess
import sys

from ai2pot_cli.menu import (
    show_numbered_menu, get_choice,
    print_section, print_kv, print_sep,
    print_success, print_warning, print_error,
)

from ai2pot_cli.commands import register

register(102, "ai2pot_cli.menus.install.install_source", "source_install_menu")
register(1021, "ai2pot_cli.menus.install.install_source", "_step1021_configure_cuda")
register(1022, "ai2pot_cli.menus.install.install_source", "_step1022_install_pytorch")
register(1023, "ai2pot_cli.menus.install.install_source", "_step1023_install_ai2pot")

# ── constants ───────────────────────────────────────────────────────
DEFAULT_ENV = "ai2pot_env"

# ── session state ───────────────────────────────────────────────────
_session = {
    "env_name": DEFAULT_ENV,
    "env_python": None,
}


# ── helpers ─────────────────────────────────────────────────────────

def _run(cmd, cwd=None, env=None):
    """Run a shell command with streaming output. Returns True on success."""
    print()
    print_kv("Running", cmd)
    print_sep()
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    print_sep()
    if result.returncode != 0:
        print_error(f"Command failed with exit code {result.returncode}")
        return False
    print_success("Done.")
    return True


def _detect_env_python():
    """Find the python executable inside the ai2pot_env environment."""
    env_name = _session.get("env_name") or DEFAULT_ENV

    cached = _session.get("env_python")
    if cached and os.path.isfile(cached):
        return cached

    for base in [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
        os.path.expanduser("~/mambaforge"),
        "/opt/conda",
    ]:
        python = os.path.join(base, "envs", env_name, "bin", "python")
        if os.path.isfile(python):
            _session["env_python"] = python
            return python

    return f"conda run -n {env_name} python"


def _python(cmd, cwd=None):
    """Run a python command inside the target environment."""
    return _run(f"{_detect_env_python()} {cmd}", cwd=cwd)


def _pip(cmd, cwd=None):
    """Run a pip command inside the target environment."""
    return _run(f"{_detect_env_python()} -m pip {cmd}", cwd=cwd)


def _exit_with_next(step, title):
    """Print next-step hint, separator, and exit."""
    print()
    print_kv("Next step", f"{step}) {title}")
    print_sep()
    print()
    sys.exit(0)


def _exit1022_with_reminder():
    """After 1022, direct user to the next step."""
    env_name = _session.get("env_name") or DEFAULT_ENV

    print()
    print_kv("Next step", f"1. conda activate {env_name}\n{' '*22}2. pip install ai2pot-cli\n{' '*22}3. 1023) Install AI2Pot")
    print_sep()
    print()
    sys.exit(0)


def _exit_done():
    """Print final separator and exit."""
    print_sep()
    print()
    sys.exit(0)


def _exit_with_usage():
    """Print usage hint after all source install steps are done, then exit."""
    print()
    print_success("All source install steps completed!")
    print()
    print_kv("Next step", "python -c \"import ai2pot; print(ai2pot.__version__)\"")
    _exit_done()


# ── step 1021: Configure CUDA ───────────────────────────────────────

def _step1021_configure_cuda():
    _require_ai2pot_source()
    print_section("Step 1021: Configure CUDA")

    cuda_home = os.environ.get("CUDA_HOME", "")
    nvcc_path = shutil.which("nvcc")

    print_kv("CUDA_HOME", cuda_home or "(not set)")
    print_kv("nvcc", nvcc_path or "(not found)")
    print()

    if cuda_home and nvcc_path:
        print_success("CUDA already configured.")
        _exit_with_next(1022, "Install PyTorch")

    if not cuda_home:
        cuda_home = input(f"  {'CUDA toolkit path':<18} [/usr/local/cuda]: ").strip() or "/usr/local/cuda"
        print()
        print(" Set the following in your shell:")
        print()
        print(f"  export CUDA_HOME={cuda_home}")
        print(f'  export PATH="${{CUDA_HOME}}/bin:${{PATH}}"')
        print(f'  export LD_LIBRARY_PATH="${{CUDA_HOME}}/lib64:${{LD_LIBRARY_PATH}}"')
        print()

    if not nvcc_path:
        print(" Verify: nvcc --version")
        print()

    done = input(" Configured? (y/n) [n]: ").strip()
    if done.lower() == 'y':
        print_success("CUDA configured.")
        _exit_with_next(1022, "Install PyTorch")

    print_warning("Configure CUDA first, then re-run this step.")
    print()
    sys.exit(0)


# ── step 1022: Install PyTorch ─────────────────────────────────────

def _step1022_install_pytorch():
    _require_ai2pot_source()
    print_section("Step 1022: Install PyTorch")

    # --- 2a. Check / create environment ---
    env_name = _session.get("env_name") or DEFAULT_ENV
    _session["env_name"] = env_name
    print_kv("Environment", env_name)

    env_exists = False
    for base in [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
        os.path.expanduser("~/mambaforge"),
        "/opt/conda",
    ]:
        if os.path.isdir(os.path.join(base, "envs", env_name)):
            env_exists = True
            break

    if not env_exists:
        py_ver = input(f"  {'Python version':<18} [3.11.13]: ").strip() or "3.11.13"
        print()
        if not _run(f"conda create -n {env_name} python={py_ver} -y"):
            sys.exit(1)
        _session.pop("env_python", None)
        print_success(f"Environment '{env_name}' created.")
        print()
    else:
        print_kv("Status", "already exists")
        print()

    # --- 2b. Check if torch is installed ---
    py = _detect_env_python()
    torch_installed = False
    result = subprocess.run(
        f"{py} -c \"import torch; print(torch.__version__); print(torch.version.cuda if torch.cuda.is_available() else 'CPU')\"",
        shell=True, capture_output=True, text=True,
    )

    if result.returncode == 0:
        lines = result.stdout.strip().splitlines()
        print_kv("PyTorch", lines[0] if lines else "unknown")
        print_kv("CUDA", lines[1] if len(lines) > 1 else "N/A")
        print()
        print_success("PyTorch already installed.")
        torch_installed = True

    if not torch_installed:
        # --- 2c. Install PyTorch ---
        print_kv("Options", "cpu / cu118 / cu121 / cu124")
        cuda = input(f"  {'CUDA version':<18} [cpu]: ").strip() or "cpu"
        print()

        index_url = f"https://download.pytorch.org/whl/{cuda}" if cuda != "cpu" else "https://download.pytorch.org/whl/cpu"

        if not _pip(f"install torch==2.4.0 --index-url {index_url}"):
            sys.exit(1)

        # --- 2d. Verify ---
        result = subprocess.run(
            f"{_detect_env_python()} -c \"import torch; print(torch.__version__); print(torch.version.cuda if torch.cuda.is_available() else 'CPU')\"",
            shell=True, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print_error("PyTorch verification failed.")
            print(result.stderr)
            sys.exit(1)

        lines = result.stdout.strip().splitlines()
        print_section("PyTorch Installed Successfully")
        print_kv("PyTorch", lines[0] if lines else "unknown")
        print_kv("CUDA", lines[1] if len(lines) > 1 else "N/A")

    # --- 2e. Install build dependencies (needed by ai2pot-cli as well) ---
    print_section("Installing Build Dependencies")

    src = os.getcwd()
    if not _pip("install -U pip setuptools wheel"):
        sys.exit(1)
    if not _pip("install scikit-build-core==0.12.2 cmake==4.3.2 pybind11==2.11.1"):
        sys.exit(1)

    req_path = os.path.join(src, "requirements-lock.txt")
    if os.path.isfile(req_path):
        if not _pip(f"install -r {req_path}"):
            sys.exit(1)
    else:
        print_warning(f"requirements-lock.txt not found in {src}")

    print_success("Build dependencies installed.")
    print()

    _exit1022_with_reminder()


# ── step 1023: Install AI2Pot ──────────────────────────────────────

def _step1023_install_ai2pot():
    _require_ai2pot_source()
    print_section("Step 1023: Install AI2Pot")

    # ensure env name is set
    env_name = _session.get("env_name") or DEFAULT_ENV
    _session["env_name"] = env_name

    src = os.getcwd()
    print_kv("Source", src)
    print()

    # --- 3a. Check if ai2pot already installed ---
    py = _detect_env_python()
    print_kv("Python", py)
    # cd away from source dir so Python can't import ai2pot from CWD
    result = subprocess.run(
        f"cd /tmp && {py} -c \"import ai2pot; print(ai2pot.__version__); from ai2pot.fromcc import nblist; print('nblist OK')\"",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode == 0:
        lines = result.stdout.strip().splitlines()
        print_kv("AI2Pot", lines[0] if lines else "unknown")
        print()
        print_success("AI2Pot already installed (compiled extensions verified).")
        _exit_with_usage()

    # --- 3b. Build AI2Pot ---
    print_section("Building AI2Pot")

    nproc = input(f"  {'CMAKE_BUILD_PARALLEL_LEVEL':<18} [16]: ").strip() or "16"
    print("  (CC: C compiler, e.g. gcc;  CXX: C++ compiler, e.g. g++.  Leave empty to auto-detect)")
    cc = input(f"  {'CC':<18} []: ").strip()
    cxx = input(f"  {'CXX':<18} []: ").strip()

    build_env = os.environ.copy()
    build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = nproc
    if cc:
        build_env["CC"] = cc
    if cxx:
        build_env["CXX"] = cxx
        build_env["CUDAHOSTCXX"] = cxx

    print()
    print_kv("CMAKE_BUILD_PARALLEL_LEVEL", nproc)
    if cc:
        print_kv("CC", cc)
    if cxx:
        print_kv("CXX", cxx)
        print_kv("CUDAHOSTCXX", cxx)
    print()

    if not _run(
        f"{_detect_env_python()} -m pip install -v --no-build-isolation --no-deps .",
        cwd=src, env=build_env,
    ):
        sys.exit(1)

    # --- 3c. Verify ---
    result = subprocess.run(
        f"cd /tmp && {_detect_env_python()} -c \"import ai2pot; print(ai2pot.__version__); from ai2pot.fromcc import nblist; print('nblist OK')\"",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print_error("AI2Pot verification failed.")
        print(result.stderr)
        sys.exit(1)

    print_section("AI2Pot Installed Successfully")
    print_kv("AI2Pot", result.stdout.strip())
    print_kv("Source", src)
    _exit_with_usage()


# ── step menu ───────────────────────────────────────────────────────

_STEPS = [
    (1021, "Configure CUDA",   "Set CUDA_HOME, PATH, LD_LIBRARY_PATH"),
    (1022, "Install PyTorch",  "Create conda env + pip install torch==2.4.0"),
    (1023, "Install AI2Pot",   "Install build deps + pip install -v --no-build-isolation --no-deps ."),
]

_STEP_FUNCS = {
    1021: _step1021_configure_cuda,
    1022: _step1022_install_pytorch,
    1023: _step1023_install_ai2pot,
}


def _require_ai2pot_source():
    """Ensure we are inside an AI2Pot source directory. Exits if not."""
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, "pyproject.toml")) and os.path.isdir(os.path.join(cwd, "ai2pot")):
        return
    print()
    print_error("Not inside an AI2Pot source directory.")
    print_kv("Current dir", cwd)
    print()
    print_warning("Please cd to the AI2Pot root directory and re-run this step.")
    print()
    sys.exit(1)


def source_install_menu():
    """Step-by-step source install sub-menu."""
    _require_ai2pot_source()

    while True:
        show_numbered_menu("Install AI2Pot from Source", _STEPS)
        choice = get_choice()
        from ai2pot_cli.commands import dispatch
        if dispatch(choice):
            continue
        if choice in _STEP_FUNCS:
            _STEP_FUNCS[choice]()  # each step calls sys.exit(0) when done
        elif choice == 9:
            return
        elif choice == 0:
            print_success("Bye.")
            sys.exit(0)
        else:
            print_warning(f"Invalid option: {choice}")
