"""Convert ORCA calculations to extxyz format.

Takes a multi-frame .xyz file and an ORCA template .inp, runs ORCA
for each frame, parses .engrad files, and writes a combined extxyz.
Requires ORCA to be installed and accessible.
"""

import os
import subprocess
import sys

from ai2pot_cli.menu import print_section, print_warning, print_error, print_kv, print_sep

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM

_PERIODIC_TABLE = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
    19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe',
    27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se',
    35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo',
    43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba',
}


def _atomic_symbol(z):
    return _PERIODIC_TABLE.get(z, 'X')


def _read_multi_xyz(filename):
    frames = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        try:
            natoms = int(lines[i].strip())
        except ValueError:
            i += 1
            continue
        atoms = []
        for j in range(i + 2, i + 2 + natoms):
            parts = lines[j].strip().split()
            atoms.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")
        frames.append({'natoms': natoms, 'atoms': atoms})
        i += 2 + natoms
    return frames


def _write_inp(template_inp, atoms, output_inp):
    with open(template_inp, 'r') as f:
        template_lines = f.readlines()
    start_idx = end_idx = None
    for i, line in enumerate(template_lines):
        if line.strip().startswith("* xyz"):
            start_idx = i + 1
        elif start_idx is not None and line.strip().startswith("*"):
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        raise ValueError("Template .inp does not contain a valid '* xyz' block.")
    new_lines = template_lines[:start_idx] + [a + "\n" for a in atoms] + template_lines[end_idx:]
    with open(output_inp, 'w') as f:
        f.writelines(new_lines)


def _parse_engrad(engrad_file):
    if not os.path.exists(engrad_file):
        raise FileNotFoundError(f"Engrad file not found: {engrad_file}")
    with open(engrad_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    natoms = int(lines[0])
    energy_hartree = float(lines[1])
    gradients = [float(lines[i]) for i in range(2, 2 + 3 * natoms)]
    atoms = []
    for i in range(2 + 3 * natoms, 2 + 3 * natoms + natoms):
        parts = lines[i].split()
        atomic_number = int(parts[0])
        x, y, z = map(float, parts[1:])
        atoms.append((atomic_number, x, y, z))
    return natoms, energy_hartree, gradients, atoms


def _write_xyz_frame(natoms, energy_ev, gradients, atoms, box_size, center_molecule):
    forces = [-g * HARTREE_BOHR_TO_EV_ANGSTROM for g in gradients]
    shift_x = shift_y = shift_z = 0.0
    if center_molecule:
        geo_center = [sum(atom[i + 1] for atom in atoms) / natoms for i in range(3)]
        shift_x = box_size / 2 - geo_center[0] * BOHR_TO_ANGSTROM
        shift_y = box_size / 2 - geo_center[1] * BOHR_TO_ANGSTROM
        shift_z = box_size / 2 - geo_center[2] * BOHR_TO_ANGSTROM

    lines = [f"{natoms}\n"]
    lines.append(
        f'Lattice="{box_size:.1f} 0.0 0.0 0.0 {box_size:.1f} 0.0 0.0 0.0 {box_size:.1f}" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy={energy_ev:.8f} '
        f'pbc="T T T" '
        f'config_type=orca2xyz\n')
    for i, atom in enumerate(atoms):
        z, x, y, z_coord = atom
        x = x * BOHR_TO_ANGSTROM + shift_x
        y = y * BOHR_TO_ANGSTROM + shift_y
        z = z_coord * BOHR_TO_ANGSTROM + shift_z
        fx, fy, fz = forces[3 * i: 3 * i + 3]
        lines.append(f"{_atomic_symbol(z)} {x:.8f} {y:.8f} {z:.8f} {fx:.8f} {fy:.8f} {fz:.8f}\n")
    return lines


def run_orca2xyz():
    """Interactive ORCA → extxyz converter."""
    print_section("ORCA → extxyz")
    print_warning("This tool runs ORCA as a subprocess. Ensure ORCA is installed and accessible.")

    xyz_file = input(" Multi-frame .xyz input file: ").strip()
    if not xyz_file or not os.path.isfile(xyz_file):
        print_error(f"File not found: {xyz_file}")
        return

    template_inp = input(" ORCA template .inp file: ").strip()
    if not template_inp or not os.path.isfile(template_inp):
        print_error(f"File not found: {template_inp}")
        return

    orca_exe = input(" Path to ORCA executable [default: orca]: ").strip() or "orca"

    box_str = input(" Box size for periodic lattice (A) [default: 50.0]: ").strip()
    box_size = float(box_str) if box_str else 50.0

    center_str = input(" Center molecule in box? (y/n) [default: y]: ").strip().lower()
    center_molecule = center_str != 'n'

    out_name = input(" Output file name [default: orca_trajectory.xyz]: ").strip() or "orca_trajectory.xyz"

    output_dir = "orca_calculations"
    os.makedirs(output_dir, exist_ok=True)

    frames = _read_multi_xyz(xyz_file)
    if not frames:
        print_error("No frames found in input .xyz file.")
        return

    problematic = []
    with open(out_name, 'w') as combined_file:
        for frame_idx, frame in enumerate(frames):
            fn = frame_idx + 1
            out_inp = os.path.join(output_dir, f"frame{fn}.inp")
            out_engrad = os.path.join(output_dir, f"frame{fn}.engrad")
            try:
                _write_inp(template_inp, frame['atoms'], out_inp)
                subprocess.run(f"{orca_exe} {out_inp} > {out_inp.replace('.inp', '.out')}",
                               shell=True, check=True)
                natoms, energy_h, gradients, atoms = _parse_engrad(out_engrad)
                energy_ev = energy_h * HARTREE_TO_EV
                frame_lines = _write_xyz_frame(natoms, energy_ev, gradients, atoms, box_size, center_molecule)
                combined_file.writelines(frame_lines)
            except Exception as e:
                problematic.append(fn)

    print_section("ORCA → extxyz Conversion Completed")
    print_kv("Output File", os.path.abspath(out_name))
    print_kv("Configurations", f"{len(frames) - len(problematic)}/{len(frames)}")
    if problematic:
        print_warning(f"Failed frames: {problematic}")
    print_sep()
    print()
    sys.exit(0)
