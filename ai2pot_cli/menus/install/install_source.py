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
    "source_dir": None,         # cwd = AI2Pot source root
    "env_name": None,           # conda env or venv name
    "env_python": None,         # path to python inside the env
    "step_done": {              # track which steps have been completed
        4121: False,
        4122: False,
        4123: False,
    },
}


# ── helpers ─────────────────────────────────────────────────────────

def _run(cmd, cwd=None, env=None):
    """Run a shell command with streaming output. Returns True on success."""
    print()
    print_kv("Running", cmd)
    print_sep()
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, env=env,
    )
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

    # already cached
    cached = _session.get("env_python")
    if cached and os.path.isfile(cached):
        return cached

    # check common conda locations
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

    # fallback: try conda run
    return f"conda run -n {env_name} python"


def _python(cmd, cwd=None):
    """Run a python command inside the target environment."""
    py = _detect_env_python()
    if py.startswith("conda run"):
        full = f"{py} {cmd}"
    else:
        full = f"{py} {cmd}"
    return _run(full, cwd=cwd)


def _pip(cmd, cwd=None):
    """Run a pip command inside the target environment."""
    py = _detect_env_python()
    if py.startswith("conda run"):
        full = f"{py} -m pip {cmd}"
    else:
        full = f"{py} -m pip {cmd}"
    return _run(full, cwd=cwd)


def _confirm_skip(step, title):
    """Ask user whether to skip a step that appears already done."""
    print()
    answer = input(f" Step {step} ({title}) appears done. Skip? (y/n) [y]: ").strip()
    if answer.lower() != 'n':
        _session["step_done"][step] = True
        print_success(f"Step {step} already completed. Skipping.")
        return True
    return False


def _next_hint(step, title):
    """Suggest the next step."""
    print()
    print_kv("Next step", f"{step}) {title}")
    print()


# ── step 4121: Configure CUDA ───────────────────────────────────────

def _step4121_configure_cuda():
    """Step 4121: Configure CUDA environment variables."""
    print_section("Step 4121: Configure CUDA")

    cuda_home = os.environ.get("CUDA_HOME", "")
    nvcc_path = shutil.which("nvcc")

    # check if already configured
    if cuda_home and nvcc_path:
        print_kv("CUDA_HOME", cuda_home)
        print_kv("nvcc", nvcc_path)
        if _confirm_skip(4121, "Configure CUDA"):
            return

    # check what's missing
    if not cuda_home:
        cuda_home = input(" CUDA toolkit path [/usr/local/cuda]: ").strip() or "/usr/local/cuda"
        print()
        print(" Set the following environment variables in your shell:")
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
        _session["step_done"][4121] = True
        print_success("CUDA configured.")
        _next_hint(4122, "Install PyTorch")
    else:
        print_warning("Please configure CUDA before proceeding.")


# ── step 4122: Install PyTorch ─────────────────────────────────────

def _step4122_install_pytorch():
    """Step 4122: Create environment and install PyTorch."""
    print_section("Step 4122: Install PyTorch")

    # --- 2a. Check / create Python environment ---
    env_name = _session.get("env_name")
    if env_name:
        print_kv("Environment", env_name)

    # detect if conda env already exists
    env_exists = False
    if env_name:
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

    if env_exists:
        print_kv("Status", "Environment already exists")
        recreate = input(" Recreate environment? (y/n) [n]: ").strip()
        if recreate.lower() == 'y':
            env_exists = False

    if not env_name or not env_exists:
        if not env_name:
            env_name = input(" Environment name [ai2pot]: ").strip() or "ai2pot"
            _session["env_name"] = env_name

        py_ver = input(" Python version [3.11.13]: ").strip() or "3.11.13"

        print()
        cmd = f"conda create -n {env_name} python={py_ver} -y"
        if not _run(cmd):
            print_error("Failed to create conda environment.")
            return

        # clear cached python path so it's re-detected
        _session.pop("env_python", None)
        print_success(f"Environment '{env_name}' created.")
        print()

    # --- 2b. Check if torch already installed ---
    env_name = _session["env_name"]  # may have been set above
    py = _detect_env_python()
    check_cmd = f"{py} -c \"import torch; print(torch.__version__); print(torch.version.cuda if torch.cuda.is_available() else 'CPU')\""
    print_kv("Checking", "PyTorch installation ...")
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        lines = result.stdout.strip().splitlines()
        print_kv("PyTorch", lines[0] if lines else "unknown")
        if len(lines) > 1:
            print_kv("CUDA", lines[1])
        if _confirm_skip(4122, "Install PyTorch"):
            return

    # --- 2c. Choose CUDA variant and install ---
    print()
    print_kv("Options", "cpu / cu118 / cu121 / cu124")
    cuda = input(" CUDA version [cpu]: ").strip() or "cpu"

    if cuda == "cpu":
        index_url = "https://download.pytorch.org/whl/cpu"
    else:
        index_url = f"https://download.pytorch.org/whl/{cuda}"

    cmd = f"install torch==2.4.0 --index-url {index_url}"
    if not _pip(cmd):
        print_error("Failed to install PyTorch.")
        return

    # --- 2d. Verify ---
    print_section("Verifying PyTorch Installation")
    verify_cmd = f"{_detect_env_python()} -c \"import torch; print('torch', torch.__version__); print('CUDA', torch.version.cuda if torch.cuda.is_available() else 'CPU')\""
    result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
        print_success("PyTorch installed and verified.")
    else:
        print_error("PyTorch verification failed.")
        print(result.stderr)
        return

    _session["step_done"][4122] = True
    _next_hint(4123, "Install AI2Pot")


# ── step 4123: Install AI2Pot ──────────────────────────────────────

def _step4123_install_ai2pot():
    """Step 4123: Build and install AI2Pot from source."""
    print_section("Step 4123: Install AI2Pot")

    src = _session["source_dir"]
    print_kv("Source", src)

    # --- 3a. Check if ai2pot already installed ---
    py = _detect_env_python()
    check_cmd = f"{py} -c \"import ai2pot; print(ai2pot.__version__)\""
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print_kv("AI2Pot", result.stdout.strip())
        if _confirm_skip(4123, "Install AI2Pot"):
            return

    # --- 3b. Install build backend ---
    print_section("Installing Build Dependencies")
    print()

    if not _pip("install -U pip setuptools wheel"):
        return

    if not _pip("install scikit-build-core==0.12.2 cmake==4.3.2 pybind11==2.11.1"):
        return

    req_path = os.path.join(src, "requirements-lock.txt")
    if os.path.isfile(req_path):
        if not _pip(f"install -r {req_path}"):
            return
    else:
        print_warning(f"requirements-lock.txt not found in {src}")

    # --- 3c. Build AI2Pot ---
    print_section("Building AI2Pot")
    print()

    nproc = input(" CMAKE_BUILD_PARALLEL_LEVEL [16]: ").strip() or "16"

    cc = input(" CC (leave blank to use default) []: ").strip()
    cxx = input(" CXX (leave blank to use default) []: ").strip()
    hostcxx = input(" CUDAHOSTCXX (leave blank to use default) []: ").strip()

    # build env
    build_env = os.environ.copy()
    build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = nproc
    if cc:
        build_env["CC"] = cc
    if cxx:
        build_env["CXX"] = cxx
    if hostcxx:
        build_env["CUDAHOSTCXX"] = hostcxx

    # construct pip command
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
    print()

    py_path = _detect_env_python()
    if py_path.startswith("conda run"):
        full_cmd = f"{py_path} -m pip install {pip_args}"
    else:
        full_cmd = f"{py_path} -m pip install {pip_args}"

    if not _run(full_cmd, cwd=src, env=build_env):
        print_error("AI2Pot build failed.")
        return

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
        return

    _session["step_done"][4123] = True
    print_sep()
    print_success("All steps completed! AI2Pot is ready.")
    print()


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

    # sanity check
    if not os.path.isfile(os.path.join(_session["source_dir"], "pyproject.toml")):
        print_warning("No pyproject.toml found in current directory.")
        print_warning(f"Make sure you are inside the AI2Pot source tree: {_session['source_dir']}")
        print()

    while True:
        show_numbered_menu("Install AI2Pot from Source", _STEPS)
        choice = get_choice()
        if choice in _STEP_FUNCS:
            _STEP_FUNCS[choice]()
        elif choice == 9:
            return
        elif choice == 0:
            print_success("Bye.")
            sys.exit(0)
        else:
            print_warning(f"Invalid option: {choice}")
