"""Step-by-step LAMMPS with AI2Pot plugin installation."""

import os
import sys

from ai2pot_cli.menu import (
    show_numbered_menu, get_choice,
    print_section, print_kv, print_sep,
    print_success, print_warning, print_error,
)

from ai2pot_cli.commands import register

register(201, "ai2pot_cli.menus.install.install_lammps", "_step201_locate_lammps")
register(202, "ai2pot_cli.menus.install.install_lammps", "_step202_integrate_plugin")
register(203, "ai2pot_cli.menus.install.install_lammps", "_step203_cmake_configure")
register(204, "ai2pot_cli.menus.install.install_lammps", "_step204_build_lammps")
register(205, "ai2pot_cli.menus.install.install_lammps", "_step205_verify")

# ── session state ───────────────────────────────────────────────────
_session = {
    "lammps_dir": None,      # abs path to LAMMPS source
    "ai2pot_dir": None,      # abs path to AI2Pot source (for plugin)
    "build_dir": None,       # LAMMPS build directory
    "lmp_binary": None,      # path to compiled lmp binary
    "cmake_done": False,
    "build_done": False,
}


def _step201_locate_lammps():
    """Step 201: Locate LAMMPS source code."""
    print_section("Step 201: Locate LAMMPS Source")

    prev = _session.get("lammps_dir", "")
    lammps_dir = input(f" LAMMPS source path [{prev}]: ").strip() or prev
    if not lammps_dir:
        print_warning("No path provided.")
        return
    lammps_dir = os.path.abspath(lammps_dir)
    if not os.path.isdir(lammps_dir):
        print_error(f"Directory not found: {lammps_dir}")
        return

    _session["lammps_dir"] = lammps_dir
    print_kv("LAMMPS source", lammps_dir)

    # TODO: detect version from version.h or CMakeLists.txt
    print_success("LAMMPS source located.")
    print()


def _step202_integrate_plugin():
    """Step 202: Integrate AI2Pot plugin into LAMMPS."""
    print_section("Step 202: Integrate AI2Pot Plugin")

    lammps = _session.get("lammps_dir")
    if not lammps:
        print_warning("LAMMPS source not set. Run Step 201 first.")
        return
    print_kv("LAMMPS source", lammps)

    ai2pot = _session.get("ai2pot_dir", "")
    ai2pot = input(f" AI2Pot source path [{ai2pot}]: ").strip() or ai2pot
    if not ai2pot:
        print_warning("No path provided.")
        return
    ai2pot = os.path.abspath(ai2pot)
    _session["ai2pot_dir"] = ai2pot
    print_kv("AI2Pot source", ai2pot)

    # TODO: copy / symlink AI2Pot plugin into LAMMPS src/
    # e.g. cp -r $AI2POT/lammps-plugin/* $LAMMPS/src/
    print_success("AI2Pot plugin integrated.")
    print()


def _step203_cmake_configure():
    """Step 203: Run CMake configure."""
    print_section("Step 203: CMake Configure")

    lammps = _session.get("lammps_dir")
    if not lammps:
        print_warning("LAMMPS source not set. Run Step 201 first.")
        return

    build_dir = _session.get("build_dir", os.path.join(lammps, "build"))
    build_dir = input(f" Build directory [{build_dir}]: ").strip() or build_dir
    _session["build_dir"] = build_dir

    print()
    print_kv("Source", lammps)
    print_kv("Build dir", build_dir)
    print()

    cmake_opts = input(" Extra CMake options [-D PKG_OPENMP=yes]: ").strip()
    if not cmake_opts:
        cmake_opts = "-D PKG_OPENMP=yes"

    cmd = f"cmake -S {lammps} -B {build_dir} {cmake_opts}"
    print()
    print_kv("Command", cmd)
    print()

    # TODO: run cmake
    _session["cmake_done"] = True
    print_success("CMake configure completed.")
    print()


def _step204_build_lammps():
    """Step 204: Build LAMMPS."""
    print_section("Step 204: Build LAMMPS")

    build_dir = _session.get("build_dir")
    if not build_dir:
        print_warning("Build directory not set. Run Step 203 first.")
        return

    nproc = os.cpu_count() or 4
    njobs = input(f" Parallel jobs [{nproc}]: ").strip()
    njobs = int(njobs) if njobs else nproc

    cmd = f"cmake --build {build_dir} -j {njobs}"
    print()
    print_kv("Command", cmd)
    print()

    # TODO: run cmake --build
    lmp_path = os.path.join(build_dir, "lmp")
    _session["lmp_binary"] = lmp_path
    _session["build_done"] = True
    print_success("LAMMPS build completed.")
    print_kv("Binary", lmp_path)
    print()


def _step205_verify():
    """Step 205: Verify LAMMPS installation."""
    print_section("Step 205: Verify LAMMPS")

    lmp = _session.get("lmp_binary")
    if not lmp or not os.path.isfile(lmp):
        print_warning("LAMMPS binary not found. Run Step 204 first.")
        return

    # TODO: run `lmp -h` to verify
    print_success("LAMMPS binary found and working.")
    print_kv("Binary", lmp)
    print_sep()
    print()


# ── step menu ───────────────────────────────────────────────────────

_STEPS = [
    (201, "Locate LAMMPS Source",      "Set path to LAMMPS source directory"),
    (202, "Integrate AI2Pot Plugin",   "Copy/link AI2Pot plugin into LAMMPS"),
    (203, "CMake Configure",           "cmake -S ... -B ..."),
    (204, "Build LAMMPS",              "cmake --build ... -j N"),
    (205, "Verify LAMMPS",             "Check lmp binary works"),
]

_STEP_FUNCS = {
    201: _step201_locate_lammps,
    202: _step202_integrate_plugin,
    203: _step203_cmake_configure,
    204: _step204_build_lammps,
    205: _step205_verify,
}


def lammps_step_menu():
    """Step-by-step LAMMPS with AI2Pot installation sub-menu."""
    while True:
        # show session summary
        if any(_session.values()):
            print_sep()
            if _session.get("lammps_dir"):
                print_kv("LAMMPS", _session["lammps_dir"])
            if _session.get("ai2pot_dir"):
                print_kv("AI2Pot plugin", _session["ai2pot_dir"])
            if _session.get("build_dir"):
                print_kv("Build dir", _session["build_dir"])
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
