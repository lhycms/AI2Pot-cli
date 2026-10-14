"""Convert existing VASP calculation results to extxyz format.

Scans a directory recursively for VASP results and writes a unified extxyz.
Per directory: prefers vasprun.xml; falls back to OUTCAR when absent.
Extracts energy/forces/stress/lattice and converts stress→virial.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write
from tqdm import tqdm

from ai2pot_cli.menu import print_section, print_warning, print_error, print_kv, print_sep


def _natural_sort_key(name: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(name))]


def _convert_atoms(atom):
    """Convert VASP stress (eV/A^3) to virial (eV) and copy free_energy→energy."""
    if 'stress' in atom.calc.results:
        xx, yy, zz, yz, xz, xy = -atom.calc.results['stress'] * atom.get_volume()
        atom.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
        del atom.calc.results['stress']
    if 'free_energy' in atom.calc.results:
        atom.calc.results['energy'] = atom.calc.results['free_energy']
        del atom.calc.results['free_energy']


def _collect_sources(search_path: Path):
    """Pair each directory with its VASP result file.

    Prefers vasprun.xml when present in a directory; otherwise falls back to
    OUTCAR in that directory.
    """
    sources = []
    for root, dirs, files in os.walk(search_path):
        dirs.sort(key=_natural_sort_key)
        if 'vasprun.xml' in files:
            sources.append(Path(root) / 'vasprun.xml')
        elif 'OUTCAR' in files:
            sources.append(Path(root) / 'OUTCAR')
    return sources


def run_vasp2xyz():
    """Interactive VASP results → extxyz converter."""
    print_section("VASP results → extxyz")

    search_dir = input(" Directory to scan for VASP results [default: .]: ").strip() or "."
    search_path = Path(search_dir)
    if not search_path.is_dir():
        print_error(f"Directory not found: {search_dir}")
        return

    out_name = input(" Output file name [default: vasp_dataset.xyz]: ").strip() or "vasp_dataset.xyz"

    sources = _collect_sources(search_path)
    if not sources:
        print_warning(f"No VASP results (vasprun.xml / OUTCAR) found under: {search_dir}")
        return

    err_count = 0
    all_atoms = []
    total = len(sources)
    for src in tqdm(sources, desc=" Processing", unit="file"):
        try:
            atoms = read(str(src), index=":")
        except Exception:
            err_count += 1
            continue
        for atom in atoms:
            _convert_atoms(atom)
            atom.info['config_type'] = 'vasp2xyz'
            all_atoms.append(atom)

    if all_atoms:
        write(out_name, all_atoms, format='extxyz')

    print_section("VASP → extxyz Conversion Completed")
    print_kv("Output File", os.path.abspath(out_name))
    print_kv("Configurations", str(len(all_atoms)))
    if err_count > 0:
        print_warning(f"{err_count}/{total} file(s) failed to parse")
    print_sep()
    print()
    sys.exit(0)
