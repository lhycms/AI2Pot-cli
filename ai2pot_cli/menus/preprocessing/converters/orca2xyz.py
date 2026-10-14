"""Convert existing ORCA calculation results to extxyz format.

Scans a directory recursively for ORCA results and writes a unified extxyz.
Primary source: *.engrad files (energy/gradient/geometry in atomic units).
Fallback (when no .engrad exists): ORCA output files ('output', *.out),
parsing FINAL SINGLE POINT ENERGY + CARTESIAN GRADIENT + coordinates.
"""

import os
import re
import sys
from pathlib import Path
from typing import List

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
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce',
    59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy',
    67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',
    75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb',
    83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn',
}


def _atomic_symbol(z):
    return _PERIODIC_TABLE.get(int(z), 'X')


def _natural_sort_key(name: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(name))]


def _parse_engrad(path: Path) -> List[dict]:
    """Parse one or more energy/gradient blocks from an ORCA .engrad file."""
    with open(path, 'r', errors='ignore') as f:
        lines = [l.rstrip('\n') for l in f]

    frames = []
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not (s.startswith('#') and 'Number of atoms' in s):
            i += 1
            continue

        j = i + 1
        while j < n and not (lines[j].strip().startswith('#') and 'Number of atoms' in lines[j]):
            j += 1
        body = [l.strip() for l in lines[i + 1:j] if l.strip() and not l.strip().startswith('#')]
        i = j

        if len(body) < 2:
            continue
        natoms = int(body[0])
        energy_h = float(body[1])
        rest = body[2:]

        if len(rest) >= 4 * natoms:
            gradients = [float(v) for v in rest[:3 * natoms]]
            coord_lines = rest[3 * natoms:3 * natoms + natoms]
        else:
            gradients = [0.0] * (3 * natoms)
            coord_lines = rest[:natoms]

        symbols, positions = [], []
        for line in coord_lines:
            parts = line.split()
            symbols.append(_atomic_symbol(parts[0]))
            positions.append((float(parts[1]) * BOHR_TO_ANGSTROM,
                              float(parts[2]) * BOHR_TO_ANGSTROM,
                              float(parts[3]) * BOHR_TO_ANGSTROM))
        forces = [-g * HARTREE_BOHR_TO_EV_ANGSTROM for g in gradients]

        frames.append({
            'symbols': symbols,
            'positions': positions,
            'energy': energy_h * HARTREE_TO_EV,
            'forces': forces,
        })
    return frames


_ENERGY_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)')
_COORD_RE = re.compile(
    r'CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n-{3,}\n'
    r'((?:\s*[A-Za-z]{1,3}\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+)')
_GRADIENT_RE = re.compile(
    r'CARTESIAN GRADIENT\s*\n-{3,}\s*\n'
    r'((?:\s*\d+\s+[A-Za-z]{1,3}\s*:\s*[-\d.Ee+]+\s+[-\d.Ee+]+\s+[-\d.Ee+]+\s*\n)+)')


def _looks_like_orca_output(path: Path) -> bool:
    try:
        with open(path, 'r', errors='ignore') as f:
            head = f.read(4096)
    except OSError:
        return False
    return 'O   R   C   A' in head or ('ORCA' in head and 'Program Version' in head)


def _parse_orca_output(path: Path) -> List[dict]:
    """Parse the final energy/geometry/gradient from an ORCA output file."""
    with open(path, 'r', errors='ignore') as f:
        text = f.read()

    energy_matches = _ENERGY_RE.findall(text)
    if not energy_matches:
        raise ValueError("no FINAL SINGLE POINT ENERGY found")
    energy_h = float(energy_matches[-1])

    coord_matches = _COORD_RE.findall(text)
    if not coord_matches:
        raise ValueError("no CARTESIAN COORDINATES (ANGSTROEM) block found")
    symbols, positions = [], []
    for line in coord_matches[-1].strip().splitlines():
        parts = line.split()
        symbols.append(parts[0])
        positions.append((float(parts[1]), float(parts[2]), float(parts[3])))

    gradient_matches = _GRADIENT_RE.findall(text)
    if not gradient_matches:
        raise ValueError("no CARTESIAN GRADIENT block found")
    gradients = []
    for line in gradient_matches[-1].strip().splitlines():
        parts = line.replace(':', ' ').split()
        gradients.extend(float(v) for v in parts[2:5])

    if len(symbols) != len(gradients) // 3:
        raise ValueError("coordinate/gradient atom count mismatch")

    forces = [-g * HARTREE_BOHR_TO_EV_ANGSTROM for g in gradients]
    return [{
        'symbols': symbols,
        'positions': positions,
        'energy': energy_h * HARTREE_TO_EV,
        'forces': forces,
    }]


def _collect_sources(search_path: Path):
    """Pair each directory with its ORCA result files.

    Prefers *.engrad when present in a directory; otherwise falls back to
    ORCA output files ('output', *.out) in that directory.
    """
    sources = []
    for root, dirs, files in os.walk(search_path):
        dirs.sort(key=_natural_sort_key)
        engrads = sorted((Path(root) / f for f in files if f.endswith('.engrad')),
                         key=_natural_sort_key)
        if engrads:
            sources += [(p, True) for p in engrads]
            continue
        outs = sorted((Path(root) / f for f in files
                       if f == 'output' or f.endswith('.out') or f.endswith('.orca_out')),
                      key=_natural_sort_key)
        sources += [(p, False) for p in outs if _looks_like_orca_output(p)]
    return sources


def _format_frame(frame: dict, box_size: float, center_molecule: bool) -> str:
    symbols = frame['symbols']
    positions = frame['positions']
    natoms = len(symbols)

    shift = (0.0, 0.0, 0.0)
    if center_molecule and natoms > 0:
        geo_center = [sum(p[i] for p in positions) / natoms for i in range(3)]
        shift = tuple(box_size / 2 - geo_center[i] for i in range(3))

    lines = [f"{natoms}\n"]
    lines.append(
        f'Lattice="{box_size:.1f} 0.0 0.0 0.0 {box_size:.1f} 0.0 0.0 0.0 {box_size:.1f}" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy={frame["energy"]:.8f} '
        f'pbc="T T T" '
        f'config_type=orca2xyz\n')
    for i, sym in enumerate(symbols):
        x, y, z = positions[i]
        x, y, z = x + shift[0], y + shift[1], z + shift[2]
        fx, fy, fz = frame['forces'][3 * i: 3 * i + 3]
        lines.append(f"{sym} {x:.8f} {y:.8f} {z:.8f} {fx:.8f} {fy:.8f} {fz:.8f}\n")
    return ''.join(lines)


def run_orca2xyz():
    """Interactive ORCA results → extxyz converter."""
    print_section("ORCA results → extxyz")

    search_dir = input(" Directory to scan for ORCA results [default: .]: ").strip() or "."
    search_path = Path(search_dir)
    if not search_path.is_dir():
        print_error(f"Directory not found: {search_dir}")
        return

    out_name = input(" Output file name [default: orca_dataset.xyz]: ").strip() or "orca_dataset.xyz"

    box_str = input(" Box size for periodic lattice (A) [default: 50.0]: ").strip()
    try:
        box_size = float(box_str) if box_str else 50.0
    except ValueError:
        print_warning(f"Invalid box size '{box_str}', using 50.0")
        box_size = 50.0

    center_str = input(" Center molecule in box? (y/n) [default: y]: ").strip().lower()
    center_molecule = center_str != 'n'

    sources = _collect_sources(search_path)
    if not sources:
        print_warning(f"No ORCA .engrad or output files found under: {search_dir}")
        return

    error_counts = {'exception': 0}
    all_frames = []

    total = len(sources)
    for path, is_engrad in sources:
        parser = _parse_engrad if is_engrad else _parse_orca_output
        try:
            frames = parser(path)
        except Exception:
            error_counts['exception'] += 1
            continue
        for frame in frames:
            all_frames.append(_format_frame(frame, box_size, center_molecule))

    if all_frames:
        Path(out_name).write_text(''.join(all_frames), encoding='utf-8')

    print_section("ORCA → extxyz Conversion Completed")
    print_kv("Output File", os.path.abspath(out_name))
    print_kv("Configurations", str(len(all_frames)))
    if error_counts['exception'] > 0:
        print_warning(f"{error_counts['exception']}/{total} file(s) failed to parse")
    print_sep()
    print()
    sys.exit(0)
