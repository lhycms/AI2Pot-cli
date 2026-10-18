"""Convert existing VASP calculation results to extxyz format.

Scans a directory recursively for VASP results and writes a unified extxyz.
Per directory: prefers vasprun.xml; falls back to OUTCAR when absent.
Extracts energy/forces/stress/lattice and converts stress→virial.
POSCAR files without a successfully parsed labeled result in the same
directory are written to a separate unlabeled extxyz.
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


_POSCAR_RE = re.compile(r'POSCAR(?:[._-].*)?$')


def _collect_poscars(search_path: Path, exclude_dirs=()):
    """Collect POSCAR / POSCAR.* files, skipping directories with labels."""
    exclude = {Path(d).resolve() for d in exclude_dirs}
    poscars = []
    for root, dirs, files in os.walk(search_path):
        dirs.sort(key=_natural_sort_key)
        if Path(root).resolve() in exclude:
            continue
        for fname in sorted(files, key=_natural_sort_key):
            if _POSCAR_RE.fullmatch(fname):
                poscars.append(Path(root) / fname)
    return poscars


def run_vasp2xyz():
    """Interactive VASP results → extxyz converter."""
    print_section("VASP results → extxyz")

    search_dir = input(" Directory to scan for VASP results [default: .]: ").strip() or "."
    search_path = Path(search_dir)
    if not search_path.is_dir():
        print_error(f"Directory not found: {search_dir}")
        return

    out_name = input(" Output file name [default: vasp_dataset.xyz]: ").strip() or "vasp_dataset.xyz"
    unlabeled_name = input(" Unlabeled POSCAR output [default: vasp_dataset_unlabeled.xyz]: ").strip() or "vasp_dataset_unlabeled.xyz"

    if Path(out_name).resolve() == Path(unlabeled_name).resolve():
        print_error("Labeled and unlabeled outputs must be different files")
        return

    sources = _collect_sources(search_path)

    err_count = 0
    all_atoms = []
    labeled_dirs = set()
    total = len(sources)
    source_iter = tqdm(sources, desc=" Processing", unit="file") if sources else []
    for src in source_iter:
        try:
            atoms = read(str(src), index=":")
        except Exception:
            err_count += 1
            continue
        for atom in atoms:
            _convert_atoms(atom)
            atom.info['config_type'] = 'vasp2xyz'
            all_atoms.append(atom)
        if atoms:
            labeled_dirs.add(src.parent)

    poscars = _collect_poscars(search_path, exclude_dirs=labeled_dirs)
    if not sources and not poscars:
        print_warning(f"No VASP results (vasprun.xml / OUTCAR / POSCAR) found under: {search_dir}")
        return

    poscar_err = 0
    unlabeled_atoms = []
    poscar_iter = tqdm(poscars, desc=" POSCAR", unit="file") if poscars else []
    for src in poscar_iter:
        try:
            atom = read(str(src), format='vasp')
        except Exception:
            poscar_err += 1
            continue
        atom.info['config_type'] = 'vasp_poscar'
        unlabeled_atoms.append(atom)

    if all_atoms:
        write(out_name, all_atoms, format='extxyz')
    if unlabeled_atoms:
        write(unlabeled_name, unlabeled_atoms, format='extxyz')

    print_section("VASP → extxyz Conversion Completed")
    if all_atoms:
        print_kv("Output File", os.path.abspath(out_name))
    print_kv("Configurations", str(len(all_atoms)))
    if unlabeled_atoms:
        print_kv("Unlabeled File", os.path.abspath(unlabeled_name))
    print_kv("Unlabeled Configs", str(len(unlabeled_atoms)))
    if err_count > 0:
        print_warning(f"{err_count}/{total} file(s) failed to parse")
    if poscar_err > 0:
        print_warning(f"{poscar_err}/{len(poscars)} POSCAR file(s) failed to parse")
    print_sep()
    print()
    sys.exit(0)
