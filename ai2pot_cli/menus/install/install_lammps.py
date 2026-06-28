"""Step-by-step LAMMPS with AI2Pot installation.

201) Setup LAMMPS Source
202) Build LAMMPS
203) Verify LAMMPS
"""

import os
import subprocess
import sys

from ai2pot_cli.menu import (
    show_numbered_menu, get_choice,
    print_section, print_kv, print_sep,
    print_success, print_warning, print_error,
)

from ai2pot_cli.commands import register

register(201, "ai2pot_cli.menus.install.install_lammps", "_step201_setup_lammps")
register(202, "ai2pot_cli.menus.install.install_lammps", "_step202_build_lammps")
register(203, "ai2pot_cli.menus.install.install_lammps", "_step203_verify")

# ── constants ───────────────────────────────────────────────────────
DEFAULT_ENV = "ai2pot_env"

# ── session state ───────────────────────────────────────────────────
_session = {
    "lammps_dir": None,
    "ai2pot_src": None,
    "lmp_binary": None,
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
    for base in [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
        os.path.expanduser("~/mambaforge"),
        "/opt/conda",
    ]:
        python = os.path.join(base, "envs", DEFAULT_ENV, "bin", "python")
        if os.path.isfile(python):
            return python
    return f"conda run -n {DEFAULT_ENV} python"


def _get_pkg_root(package):
    """Get the install path of a Python package via the env's python."""
    py = _detect_env_python()
    result = subprocess.run(
        f"{py} -c \"import {package}, os; print(os.path.dirname({package}.__file__))\"",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print_error(f"Cannot locate package: {package}")
        print(result.stderr)
        return None
    return result.stdout.strip()


def _exit_with_next(step, title):
    """Print next-step hint, separator, and exit."""
    print()
    print_kv("Next step", f"{step}) {title}")
    print_sep()
    print()
    sys.exit(0)


def _exit_done():
    """Print final separator and exit."""
    print_sep()
    print()
    sys.exit(0)


# ── step 201: Setup LAMMPS Source ───────────────────────────────────

def _step201_setup_lammps():
    """Step 201: Locate LAMMPS source and copy AI2Pot interface files."""
    print_section("Step 201: Setup LAMMPS Source")

    # check if already done
    lammps_dir = _session.get("lammps_dir")
    ai2pot_src = _session.get("ai2pot_src", os.getcwd())
    if lammps_dir:
        dst_src = os.path.join(lammps_dir, "src", "AI2POT")
        if os.path.isfile(dst_src):
            print_kv("LAMMPS source", lammps_dir)
            print_kv("AI2Pot interface", os.path.join(ai2pot_src, "interface", "lammps"))
            print()
            print_success("Step 201 already completed.")
            _exit_with_next(202, "Build LAMMPS")

    # --- 201a. Locate LAMMPS ---
    prev = lammps_dir or ""
    lammps_dir = input(f"  {'LAMMPS source path':<18} [{prev}]: ").strip() or prev
    if not lammps_dir:
        print_warning("No path provided.")
        sys.exit(0)
    lammps_dir = os.path.abspath(lammps_dir)
    if not os.path.isdir(lammps_dir):
        print_error(f"Directory not found: {lammps_dir}")
        sys.exit(1)
    _session["lammps_dir"] = lammps_dir
    print_kv("LAMMPS source", lammps_dir)

    # --- 201b. Copy interface files ---
    src_dir = os.path.join(ai2pot_src, "interface", "lammps")
    if not os.path.isdir(src_dir):
        print_error(f"Interface directory not found: {src_dir}")
        sys.exit(1)

    dst_src = os.path.join(lammps_dir, "src")
    if not _run(f"cp {src_dir}/AI2POT {dst_src}/"):
        sys.exit(1)

    makefile_src = os.path.join(src_dir, "Makefile.mpi")
    if os.path.isfile(makefile_src):
        dst_make = os.path.join(lammps_dir, "src", "MAKE")
        os.makedirs(dst_make, exist_ok=True)
        if not _run(f"cp {makefile_src} {dst_make}/"):
            sys.exit(1)

    print_section("LAMMPS Source Setup Complete")
    print_kv("LAMMPS source", lammps_dir)
    print_kv("Interface", src_dir)
    _exit_with_next(202, "Build LAMMPS")


# ── step 202: Build LAMMPS ──────────────────────────────────────────

def _step202_build_lammps():
    """Step 202: Enable AI2POT package and build LAMMPS."""
    print_section("Step 202: Build LAMMPS")

    lammps = _session.get("lammps_dir")
    if not lammps:
        print_warning("LAMMPS source not set. Run Step 201 first.")
        print()
        sys.exit(0)

    src_dir = os.path.join(lammps, "src")
    print_kv("LAMMPS src", src_dir)
    print()

    # check if already built
    lmp_path = os.path.join(src_dir, "lmp_mpi")
    if os.path.isfile(lmp_path):
        print_kv("Binary", lmp_path)
        print()
        print_success("Step 202 already completed.")
        _exit_with_next(203, "Verify LAMMPS")

    # --- 202a. Enable AI2POT package ---
    if not _run("make yes-AI2POT", cwd=src_dir):
        sys.exit(1)
    print_success("AI2POT package enabled.")
    print()

    # --- 202b. Resolve package paths ---
    print(" Resolving package paths ...")
    torch_root = _get_pkg_root("torch")
    ai2pot_root = _get_pkg_root("ai2pot")

    if not torch_root or not ai2pot_root:
        print_error("Make sure torch and ai2pot are installed in ai2pot_env.")
        sys.exit(1)

    print_kv("TORCH_ROOT", torch_root)
    print_kv("AI2POT_ROOT", ai2pot_root)
    print()

    # --- 202c. Build ---
    nproc = os.cpu_count() or 4
    njobs = input(f"  {'Parallel jobs':<18} [{nproc}]: ").strip()
    njobs = int(njobs) if njobs else nproc

    cmd = (
        f"make -j {njobs} mpi "
        f"TORCH_ROOT={torch_root} "
        f"AI2POT_ROOT={ai2pot_root}"
    )

    if not _run(cmd, cwd=src_dir):
        sys.exit(1)

    _session["lmp_binary"] = lmp_path

    print_section("LAMMPS Build Completed")
    print_kv("Binary", lmp_path)
    print_kv("TORCH_ROOT", torch_root)
    print_kv("AI2POT_ROOT", ai2pot_root)
    _exit_with_next(203, "Verify LAMMPS")


# ── step 203: Verify LAMMPS ─────────────────────────────────────────

def _step203_verify():
    """Step 203: Verify LAMMPS installation."""
    print_section("Step 203: Verify LAMMPS")

    lmp = _session.get("lmp_binary")
    if not lmp:
        lmp = _session.get("lammps_dir", "")
        if lmp:
            lmp = os.path.join(lmp, "src", "lmp_mpi")
    if not lmp or not os.path.isfile(lmp):
        print_warning("LAMMPS binary not found. Run Step 202 first.")
        print()
        sys.exit(0)

    _session["lmp_binary"] = lmp
    print_kv("Binary", lmp)
    print()

    result = subprocess.run(f"{lmp} -h", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print_error("LAMMPS binary may not be working correctly.")
        print(result.stderr)
        sys.exit(1)

    first_line = result.stdout.strip().split("\n")[0] if result.stdout else ""
    print_success("LAMMPS binary is working.")
    if first_line:
        print_kv("Info", first_line[:60])

    print()
    print_success("All LAMMPS steps completed!")
    _exit_done()


# ── step menu ───────────────────────────────────────────────────────

_STEPS = [
    (201, "Setup LAMMPS Source", "Locate LAMMPS dir + copy AI2Pot interface files"),
    (202, "Build LAMMPS",        "make yes-AI2POT + make -j N mpi"),
    (203, "Verify LAMMPS",       "Check lmp_mpi binary works"),
]

_STEP_FUNCS = {
    201: _step201_setup_lammps,
    202: _step202_build_lammps,
    203: _step203_verify,
}


def lammps_step_menu():
    """Step-by-step LAMMPS with AI2Pot installation sub-menu."""
    from ai2pot_cli.menus.install.install_source import _require_ai2pot_source
    if not _require_ai2pot_source():
        return

    _session["ai2pot_src"] = os.getcwd()

    while True:
        if any(_session.values()):
            print_sep()
            if _session.get("lammps_dir"):
                print_kv("LAMMPS", _session["lammps_dir"])
            if _session.get("ai2pot_src"):
                print_kv("AI2Pot src", _session["ai2pot_src"])
            print_sep()

        show_numbered_menu("Install LAMMPS with AI2Pot", _STEPS)
        choice = get_choice()
        from ai2pot_cli.commands import dispatch
        if dispatch(choice):
            continue
        if choice in _STEP_FUNCS:
            _STEP_FUNCS[choice]()
        elif choice == 9:
            return
        elif choice == 0:
            print_success("Bye.")
            sys.exit(0)
        else:
            print_warning(f"Invalid option: {choice}")
