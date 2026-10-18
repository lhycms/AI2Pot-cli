"""Convert LAMMPS structure trajectories to extxyz format.

Scans a directory recursively for LAMMPS dump files (*.lammpstrj / *.dump)
and writes every frame to a unified extxyz. No energy/force labels are read.
Atom types are mapped to element symbols using the Masses section of a
LAMMPS data file (via read_data in an input script) or pair_coeff ordering.
Frames are streamed so large trajectories do not need to fit in memory.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from ase.io import iread, write
from tqdm import tqdm

from ai2pot_cli.menu import print_section, print_warning, print_error, print_kv, print_sep

_ELEMENT_SET = set(chemical_symbols[1:])


def _natural_sort_key(name: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(name))]


def _collect_trajectories(search_path: Path):
    """Collect LAMMPS dump files recursively."""
    trajectories = []
    for root, dirs, files in os.walk(search_path):
        dirs.sort(key=_natural_sort_key)
        for fname in sorted(files, key=_natural_sort_key):
            if fname.endswith('.lammpstrj') or fname.endswith('.dump'):
                trajectories.append(Path(root) / fname)
    return trajectories


def _count_frames(path: Path):
    """Count 'ITEM: TIMESTEP' markers in a text dump without parsing it."""
    marker = b'ITEM: TIMESTEP'
    count = 0
    carry = b''
    try:
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                data = carry + chunk
                count += data.count(marker)
                carry = data[-(len(marker) - 1):]
    except OSError:
        return 0
    return count


def _symbol_from_mass(mass: float):
    diffs = np.abs(atomic_masses[1:] - mass)
    idx = int(np.argmin(diffs))
    if diffs[idx] / max(mass, 1.0) < 0.02:
        return chemical_symbols[idx + 1]
    return None


def _parse_masses(data_path: Path):
    """Parse a LAMMPS data file Masses section into {type: element}."""
    try:
        lines = data_path.read_text(errors='ignore').splitlines()
    except OSError:
        return {}

    for i, line in enumerate(lines):
        if line.strip().lower().startswith('masses'):
            break
    else:
        return {}

    mapping = {}
    for line in lines[i + 1:]:
        stripped = line.strip()
        if not stripped:
            if mapping:
                break
            continue
        body, _, comment = stripped.partition('#')
        tokens = body.split()
        if len(tokens) < 2:
            if mapping:
                break
            continue
        try:
            itype = int(tokens[0])
            mass = float(tokens[1])
        except ValueError:
            if mapping:
                break
            continue
        symbol = None
        comment_tokens = comment.split()
        if comment_tokens:
            candidate = comment_tokens[0].capitalize()
            if candidate in _ELEMENT_SET:
                symbol = candidate
        if symbol is None:
            symbol = _symbol_from_mass(mass)
        if symbol:
            mapping[itype] = symbol
    return mapping


def _parse_input_script(inp_path: Path):
    """Return (read_data name, element order from pair_coeff)."""
    try:
        text = inp_path.read_text(errors='ignore')
    except OSError:
        return None, None

    read_data = None
    pair_elements = None
    for line in text.splitlines():
        tokens = line.split('#', 1)[0].split()
        if not tokens:
            continue
        keyword = tokens[0].lower()
        if keyword == 'read_data' and len(tokens) > 1:
            read_data = tokens[1]
        elif keyword == 'pair_coeff':
            pair_elements = [t.capitalize() for t in tokens[1:]
                             if t.capitalize() in _ELEMENT_SET]
    return read_data, pair_elements


def _input_candidates(directory: Path):
    candidates = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name == 'lmp.in' or name.endswith('.in') or name.startswith('in.') \
                or name.endswith('.lammps'):
            candidates.append(entry)
    return sorted(candidates, key=_natural_sort_key)


def _find_element_map(directory: Path):
    """Find a type→element mapping for one LAMMPS directory."""
    for inp in _input_candidates(directory):
        read_data, pair_elements = _parse_input_script(inp)
        if read_data:
            data_path = directory / read_data
            if not data_path.is_file():
                data_path = directory / Path(read_data).name
            if data_path.is_file():
                mapping = _parse_masses(data_path)
                if mapping:
                    return mapping
        if pair_elements:
            return {i + 1: sym for i, sym in enumerate(pair_elements)}

    data_files = list(directory.glob('*.data')) + list(directory.glob('data.*'))
    for data_path in sorted(data_files, key=_natural_sort_key):
        mapping = _parse_masses(data_path)
        if mapping:
            return mapping
    return {}


def _apply_element_map(atoms, mapping):
    """Replace LAMMPS type numbers with atomic numbers; return whether mapped."""
    types = [int(t) for t in atoms.numbers]
    mapped = bool(mapping) and set(types) <= set(mapping)
    if mapped:
        atoms.numbers = np.array([atomic_numbers[mapping[t]] for t in types])
    atoms.info['config_type'] = 'lammps2xyz'
    return mapped


def run_lammps2xyz():
    """Interactive LAMMPS trajectories → extxyz converter."""
    print_section("LAMMPS → extxyz")

    search_dir = input(" Directory to scan for LAMMPS trajectories [default: .]: ").strip() or "."
    search_path = Path(search_dir)
    if not search_path.is_dir():
        print_error(f"Directory not found: {search_dir}")
        return

    out_name = input(" Output file name [default: lammps_dataset.xyz]: ").strip() or "lammps_dataset.xyz"

    trajectories = _collect_trajectories(search_path)
    if not trajectories:
        print_warning(f"No LAMMPS trajectories (*.lammpstrj / *.dump) found under: {search_dir}")
        return

    error_count = 0
    frame_count = 0
    unmapped_dirs = set()
    mapping_cache = {}
    total = len(trajectories)
    frame_totals = {src: _count_frames(src) for src in trajectories}
    pbar = tqdm(total=sum(frame_totals.values()), desc=" Processing", unit="frame") \
        if sum(frame_totals.values()) else None

    with open(out_name, 'w') as fout:
        for src in trajectories:
            directory = src.parent
            if directory not in mapping_cache:
                mapping_cache[directory] = _find_element_map(directory)
            mapping = mapping_cache[directory]
            remaining = frame_totals[src]
            try:
                for atoms in iread(str(src), format='lammps-dump-text', parallel=False):
                    if not _apply_element_map(atoms, mapping):
                        unmapped_dirs.add(directory)
                    write(fout, atoms, format='extxyz')
                    frame_count += 1
                    remaining -= 1
                    if pbar:
                        pbar.update(1)
            except Exception:
                error_count += 1
                if pbar and remaining > 0:
                    pbar.update(remaining)
    if pbar:
        pbar.close()

    if frame_count == 0:
        Path(out_name).unlink(missing_ok=True)

    print_section("LAMMPS → extxyz Conversion Completed")
    if frame_count:
        print_kv("Output File", os.path.abspath(out_name))
    print_kv("Configurations", str(frame_count))
    if error_count > 0:
        print_warning(f"{error_count}/{total} file(s) failed to parse")
    if unmapped_dirs:
        print_warning(f"No element mapping found for {len(unmapped_dirs)} directory(ies); "
                      "atom types written as-is")
    print_sep()
    print()
    sys.exit(0)
