"""Step-by-step AI2Pot source installation.

4121) Configure CUDA
4122) Install PyTorch
4123) Install AI2Pot
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

# ── session state ───────────────────────────────────────────────────
_session = {
    "source_dir": None,
    "env_name": None,
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
    """Find the python executable inside the target environment."""
    env_name = _session.get("env_name")
    if not env_name:
        return sys.executable

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
    """Print next-step hint and exit."""
    print()
    print_kv("Next step", f"{step}) {title}")
    print()
    sys.exit(0)


# ── step 4121: Configure CUDA ───────────────────────────────────────

def _step4121_configure_cuda():
    """Step 4121: Configure CUDA environment variables."""
    print_section("Step 4121: Configure CUDA")

    cuda_home = os.environ.get("CUDA_HOME", "")
    nvcc_path = shutil.which("nvcc")

    # already configured — skip silently
    if cuda_home and nvcc_path:
        print_kv("CUDA_HOME", cuda_home)
        print_kv("nvcc", nvcc_path)
        print()
        print_success("Step 4121 already completed.")
        _exit_with_next(4122, "Install PyTorch")

    # guide user to configure
    if not cuda_home:
        cuda_home = input(" CUDA toolkit path [/usr/local/cuda]: ").strip() or "/usr/local/cuda"
        print()
        print(" Set the following in your shell and re-run this step:")
        print()
        print(f"  export CUDA_HOME={cuda_home}")
        print(f'  export PATH="${{CUDA_HOME}}/bin:${{PATH}}"')
        print(f'  export LD_LIBRARY_PATH="${{CUDA_HOME}}/lib64:${{LD_LIBRARY_PATH}}"')
        print()

    if not nvcc_path:
        print(" Then verify the CUDA compiler is accessible:")
        print("  nvcc --version")
        print()

    done = input(" Have you configured CUDA as above? (y/n) [n]: ").strip()
    if done.lower() == 'y':
        print_success("CUDA configured.")
        _exit_with_next(4122, "Install PyTorch")
    else:
        print_warning("Please configure CUDA before proceeding.")
        print()
        sys.exit(0)


# ── step 4122: Install PyTorch ─────────────────────────────────────

def _step4122_install_pytorch():
    """Step 4122: Create environment and install PyTorch."""
    print_section("Step 4122: Install PyTorch")

    # --- 2a. Check / create Python environment ---
    env_name = _session.get("env_name")
    if not env_name:
        env_name = input(" Environment name [ai2pot]: ").strip() or "ai2pot"
        _session["env_name"] = env_name

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
        py_ver = input(" Python version [3.11.13]: ").strip() or "3.11.13"
        print()
        if not _run(f"conda create -n {env_name} python={py_ver} -y"):
            print_error("Failed to create conda environment.")
            sys.exit(1)
        _session.pop("env_python", None)
        print_success(f"Environment '{env_name}' created.")
        print()
    else:
        print_kv("Environment", f"'{env_name}' already exists")
        print()

    # --- 2b. Check if torch already installed ---
    py = _detect_env_python()
    check_cmd = f"{py} -c \"import torch; print(torch.__version__); print('CUDA', torch.version.cuda if torch.cuda.is_available() else 'CPU')\""
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout.strip())
        print()
        print_success("Step 4122 already completed.")
        _exit_with_next(4123, "Install AI2Pot")

    # --- 2c. Install PyTorch ---
    print()
    print_kv("Options", "cpu / cu118 / cu121 / cu124")
    cuda = input(" CUDA version [cpu]: ").strip() or "cpu"

    index_url = f"https://download.pytorch.org/whl/{cuda}" if cuda != "cpu" else "https://download.pytorch.org/whl/cpu"

    if not _pip(f"install torch==2.4.0 --index-url {index_url}"):
        print_error("Failed to install PyTorch.")
        sys.exit(1)

    # --- 2d. Verify ---
    print_section("Verifying PyTorch Installation")
    verify_cmd = f"{_detect_env_python()} -c \"import torch; print('torch', torch.__version__); print('CUDA', torch.version.cuda if torch.cuda.is_available() else 'CPU')\""
    result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
        print()
        print_success("PyTorch installed and verified.")
    else:
        print_error("PyTorch verification failed.")
        print(result.stderr)
        sys.exit(1)

    _exit_with_next(4123, "Install AI2Pot")


# ── step 4123: Install AI2Pot ──────────────────────────────────────

def _step4123_install_ai2pot():
    """Step 4123: Build and install AI2Pot from source."""
    print_section("Step 4123: Install AI2Pot")

    src = _session["source_dir"]
    print_kv("Source", src)
    print()

    # --- 3a. Check if ai2pot already installed ---
    py = _detect_env_python()
    check_cmd = f"{py} -c \"import ai2pot; print(ai2pot.__version__)\""
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print_kv("AI2Pot", result.stdout.strip())
        print()
        print_success("Step 4123 already completed.")
        print()
        print_kv("Next", "42) Install LAMMPS + AI2Pot")
        print()
        sys.exit(0)

    # --- 3b. Install build backend ---
    print_section("Installing Build Dependencies")
    print()

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

    # --- 3c. Build AI2Pot ---
    print_section("Building AI2Pot")
    print()

    nproc = input(" CMAKE_BUILD_PARALLEL_LEVEL [16]: ").strip() or "16"

    cc = input(" CC (leave blank for system default) []: ").strip()
    cxx = input(" CXX (leave blank for system default) []: ").strip()
    hostcxx = input(" CUDAHOSTCXX (leave blank for system default) []: ").strip()

    build_env = os.environ.copy()
    build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = nproc
    if cc:
        build_env["CC"] = cc
    if cxx:
        build_env["CXX"] = cxx
    if hostcxx:
        build_env["CUDAHOSTCXX"] = hostcxx

    pip_args = "-v --no-build-isolation --no-deps ."

    print()
    print_kv("CMAKE_BUILD_PARALLEL_LEVEL", nproc)
    if cc:
        print_kv("CC", cc)
    if cxx:
        print_kv("CXX", cxx)
    if hostcxx:
        print_kv("CUDAHOSTCXX", hostcxx)
    print_kv("pip args", pip_args)

    if not _run(f"{_detect_env_python()} -m pip install {pip_args}", cwd=src, env=build_env):
        print_error("AI2Pot build failed.")
        sys.exit(1)

    # --- 3d. Verify ---
    print_section("Verifying AI2Pot Installation")
    verify_cmd = f"{_detect_env_python()} -c \"import ai2pot; print('AI2Pot', ai2pot.__version__)\""
    result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
        print()
        print_success("AI2Pot installed successfully.")
    else:
        print_error("AI2Pot verification failed.")
        print(result.stderr)
        sys.exit(1)

    print_sep()
    print_success("All steps completed! AI2Pot is ready.")
    print()
    print_kv("Next", "42) Install LAMMPS + AI2Pot")
    print()
    sys.exit(0)


# ── step menu ───────────────────────────────────────────────────────

_STEPS = [
    (4121, "Configure CUDA",   "Set CUDA_HOME, PATH, LD_LIBRARY_PATH"),
    (4122, "Install PyTorch",  "Create conda env + pip install torch==2.4.0"),
    (4123, "Install AI2Pot",   "Install build deps + pip install -v --no-build-isolation --no-deps ."),
]

_STEP_FUNCS = {
    4121: _step4121_configure_cuda,
    4122: _step4122_install_pytorch,
    4123: _step4123_install_ai2pot,
}


def source_install_menu():
    """Step-by-step source install sub-menu.

    User is expected to already be inside the AI2Pot source directory.
    """
    _session["source_dir"] = os.getcwd()

    if not os.path.isfile(os.path.join(_session["source_dir"], "pyproject.toml")):
        print_warning("No pyproject.toml found in current directory.")
        print_warning(f"Make sure you are inside the AI2Pot source tree: {_session['source_dir']}")
        print()

    while True:
        show_numbered_menu("Install AI2Pot from Source", _STEPS)
        choice = get_choice()
        if choice in _STEP_FUNCS:
            _STEP_FUNCS[choice]()  # each step calls sys.exit(0) when done
        elif choice == 9:
            return
        elif choice == 0:
            print_success("Bye.")
            sys.exit(0)
        else:
            print_warning(f"Invalid option: {choice}")
