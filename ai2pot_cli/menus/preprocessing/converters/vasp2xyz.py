"""Convert VASP vasprun.xml output to extxyz format.

Reads vasprun.xml files recursively from a directory, extracts energy/forces/
stress/lattice, converts stress→virial, and writes a unified train.xyz.
"""

import os
import sys
import numpy as np
from ase.io import read, write
from tqdm import tqdm

from ai2pot_cli.menu import print_section, print_success, print_warning, print_error, print_kv, print_sep


def _convert_atoms(atom):
    """Convert VASP stress (eV/A^3) to virial (eV) and copy free_energy→energy."""
    xx, yy, zz, yz, xz, xy = -atom.calc.results['stress'] * atom.get_volume()
    atom.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
    atom.calc.results['energy'] = atom.calc.results['free_energy']
    del atom.calc.results['stress']
    del atom.calc.results['free_energy']


def _find_vasprun(start_path='.'):
    result = []
    for root, dirs, files in os.walk(start_path):
        if 'vasprun.xml' in files:
            result.append(os.path.join(root, 'vasprun.xml'))
    return result


def run_vasp2xyz():
    """Interactive VASP → extxyz converter."""
    print_section("VASP vasprun.xml → extxyz")

    search_dir = input(" Directory to scan for vasprun.xml [default: .]: ").strip() or "."
    if not os.path.isdir(search_dir):
        print_error(f"Directory not found: {search_dir}")
        return

    out_name = input(" Output file name [default: train.xyz]: ").strip() or "train.xyz"

    file_list = _find_vasprun(start_path=search_dir)
    if not file_list:
        print_warning(f"No vasprun.xml files found under: {search_dir}")
        return

    cnum = 0
    atoms_list, err_list = [], []
    for dir_name in tqdm(file_list, desc=" Processing", unit="file"):
        try:
            atoms = read(dir_name.strip('\n'), index=":")
        except Exception:
            err_list.append(dir_name)
            continue
        for ai in range(len(atoms)):
            _convert_atoms(atoms[ai])
            atoms_list.append(atoms[ai])
        cnum += len(atoms)

    if atoms_list:
        write(out_name, atoms_list, format='extxyz')

    print_section("VASP → extxyz Conversion Completed")
    print_kv("Output File", os.path.abspath(out_name))
    print_kv("Configurations", str(cnum))
    if err_list:
        print_warning(f"{len(err_list)} file(s) failed to parse")
    print_sep()
    print()
    sys.exit(0)
