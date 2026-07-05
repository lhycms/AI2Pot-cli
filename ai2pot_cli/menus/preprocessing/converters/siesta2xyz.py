"""Convert SIESTA outputs to extxyz format.

Reads BASIS_ENTHALPY, *.FA, *STRUCT_OUT, relax.out from a directory,
extracts energy/forces/stress/lattice, and writes an extxyz file.
"""

import os
import sys
from pathlib import Path
import numpy as np

from ai2pot_cli.menu import print_section, print_warning, print_error, print_kv, print_sep

_ELEMENT_MAP = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
    30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
    37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba',
}


def run_siesta2xyz():
    """Interactive SIESTA → extxyz converter (single-frame, single directory)."""
    print_section("SIESTA → extxyz")

    work_dir = input(" Directory containing SIESTA outputs [default: .]: ").strip() or "."
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        print_error(f"Directory not found: {work_dir}")
        return

    out_name = input(" Output file name [default: siesta2nep.xyz]: ").strip() or "siesta2nep.xyz"

    # 1. Free energy from BASIS_ENTHALPY
    enthalpy_path = work_dir / "BASIS_ENTHALPY"
    if not enthalpy_path.is_file():
        print_error(f"BASIS_ENTHALPY not found in {work_dir}")
        return
    with open(enthalpy_path) as f:
        energy = None
        for line in f:
            if "The above number is the electronic (free)energy:" in line:
                energy = float(line.split(":")[-1])
                break
    if energy is None:
        print_error("Could not extract energy from BASIS_ENTHALPY")
        return

    # 2. Forces from *.FA
    fa_files = list(work_dir.glob("*.FA"))
    if not fa_files:
        print_error(f"No .FA file found in {work_dir}")
        return
    with open(fa_files[0]) as f:
        fa_lines = f.readlines()
    natoms = int(fa_lines[0])
    indices, forces = [], []
    for line in fa_lines[1:natoms + 1]:
        idx, fx, fy, fz = line.split()
        indices.append(int(idx))
        forces.append([float(fx), float(fy), float(fz)])
    indices = np.array(indices)
    forces = np.array(forces)

    # 3. Geometry from *STRUCT_OUT
    struct_files = list(work_dir.glob("*STRUCT_OUT"))
    if not struct_files:
        print_error(f"No STRUCT_OUT file found in {work_dir}")
        return
    with open(struct_files[0]) as f:
        s_lines = f.readlines()

    supercell = np.array([[float(x) for x in s_lines[i].split()] for i in range(3)])
    natoms_struct = int(s_lines[3])
    atom_lines = s_lines[4:4 + natoms_struct]

    atomic_numbers = []
    frac = []
    for ln in atom_lines:
        tokens = ln.split()
        atomic_numbers.append(int(tokens[1]))
        frac.append([float(x) for x in tokens[-3:]])

    frac = np.array(frac)
    wrapped_frac = frac % 1.0
    cart = wrapped_frac @ supercell
    atom_symbols = [_ELEMENT_MAP.get(z, f"X{z}") for z in atomic_numbers]

    # 4. Stress tensor from relax.out
    relax_path = work_dir / "relax.out"
    if not relax_path.is_file():
        print_error(f"relax.out not found in {work_dir}")
        return
    with open(relax_path) as f:
        r_lines = f.readlines()

    stress_indices = [idx for idx, line in enumerate(r_lines)
                       if "siesta: Stress tensor (static) (eV/Ang**3):" in line]
    if not stress_indices:
        virial_flat = [0.0] * 9
    else:
        si = stress_indices[-1]
        tensor_lines = r_lines[si + 1: si + 4]
        stress_tensor = []
        for ln in tensor_lines:
            parts = ln.split()
            stress_tensor.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
        stress_tensor = np.array(stress_tensor)
        virial_tensor = -stress_tensor * np.abs(np.linalg.det(supercell))
        virial_flat = virial_tensor.flatten()

    # 5. Write extxyz
    with open(out_name, 'w') as f:
        f.write(f"{natoms_struct}\n")
        lattice_str = " ".join([f"{x:.8f}" for x in supercell.flatten()])
        virial_str = " ".join([f"{x:.8f}" for x in virial_flat])
        metadata = (f'Lattice="{lattice_str}" '
                    f'Properties=species:S:1:pos:R:3:forces:R:3 '
                    f'energy={energy:.8f} '
                    f'virial="{virial_str}" '
                    f'pbc="T T T" '
                    f'config_type=siesta2nep')
        f.write(metadata + "\n")
        for i in range(natoms_struct):
            elem = atom_symbols[i]
            x, y, z = cart[i]
            fx, fy, fz = forces[i]
            f.write(f"{elem} {x:.8f} {y:.8f} {z:.8f} {fx:.8f} {fy:.8f} {fz:.8f}\n")

    print_section("SIESTA → extxyz Conversion Completed")
    print_kv("Output File", str(Path(out_name).resolve()))
    print_kv("Atoms", str(natoms_struct))
    print_kv("Energy (eV)", f"{energy:.6f}")
    print_sep()
    print()
    sys.exit(0)
