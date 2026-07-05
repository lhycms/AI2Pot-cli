"""Convert CP2K output (.inp + .log) to extxyz format.

Scans for .inp files, matches them with .log files, extracts geometry/energy/
forces/stress, converts stress(GPa)→virial(eV), and writes a unified extxyz.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

from ai2pot_cli.menu import print_section, print_warning, print_error, print_kv, print_sep


def _natural_sort_key(name: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(name))]


def _find_log_for_inp(inp: Path) -> Optional[Path]:
    candidates = [
        inp.with_suffix('.log'),
        inp.parent / f"cp2k-{inp.stem}.log",
        inp.parent / f"cp2k_{inp.stem}.log",
        inp.parent / "cp2k.log",
        inp.parent / "output.log",
        inp.parent / f"{inp.stem}_output.log",
    ]
    for log in candidates:
        if log.is_file():
            return log
    logs = list(inp.parent.glob("*.log"))
    return logs[0] if logs else None


def _extract_lattice(content: str) -> List[float]:
    match = re.search(
        r'&CELL.*?A\s+([\d.\-E+\s]+)\s+B\s+([\d.\-E+\s]+)\s+C\s+([\d.\-E+\s]+).*?&END CELL',
        content, re.DOTALL | re.IGNORECASE)
    return [float(x) for group in match.groups() for x in group.split()] if match else [0.0] * 9


def _compute_volume(lattice: List[float]) -> float:
    if len(lattice) != 9:
        return 1.0
    a = np.array(lattice[0:3])
    b = np.array(lattice[3:6])
    c = np.array(lattice[6:9])
    return abs(np.dot(a, np.cross(b, c)))


def _extract_atoms(content: str) -> List[List]:
    match = re.search(r'&COORD\s*(.*?)&END COORD', content, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    atoms = []
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                atoms.append([parts[0]] + [float(x) for x in parts[1:4]])
    return atoms


def _extract_energy(content: str) -> float:
    match = re.search(r'ENERGY\| Total FORCE_EVAL.*?:\s+(-?\d[\d.\-Ee\+]*)', content)
    return float(match.group(1)) * 27.2113838565563 if match else float('nan')


def _extract_stress_gpa(content: str) -> List[float]:
    match = re.search(r'STRESS\|\s+Analytical stress tensor\s+\[GPa\]', content)
    if not match:
        return []
    pattern = r'STRESS\|\s+[xyz]\s+([-\d.E+\-]+)\s+([-\d.E+\-]+)\s+([-\d.E+\-]+)'
    matches = re.findall(pattern, content, re.IGNORECASE)
    if len(matches) >= 3:
        return [float(val) for row in matches[:3] for val in row]
    return []


def _stress_to_virial(stress_gpa: List[float], volume_ang3: float) -> List[float]:
    if len(stress_gpa) != 9 or volume_ang3 <= 0:
        return [0.0] * 9
    factor = volume_ang3 / 160.2176634
    return [s * factor for s in stress_gpa]


def _extract_forces(content: str) -> List[List[float]]:
    match = re.search(
        r'ATOMIC FORCES in \[a\.u\.\]\n\n # Atom\s+Kind\s+Element\s+X\s+Y\s+Z\n(.*?)(?=\n SUM OF ATOMIC FORCES)',
        content, re.DOTALL)
    if not match:
        return []
    forces = []
    for line in match.group(1).strip().splitlines():
        parts = line.split()
        if len(parts) >= 6:
            fx, fy, fz = [float(parts[i]) * 51.4220631857 for i in range(3, 6)]
            forces.append([fx, fy, fz])
    return forces


def _format_frame(lattice, atoms, energy, virial, forces, dir_id: str) -> str:
    lines = [f"{len(atoms)}\n"]
    lattice_str = ' '.join(f"{x:.10f}" for x in lattice)
    virial_str = ' '.join(f"{x:.10f}" for x in virial)
    info = (
        f'Lattice="{lattice_str}" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy={energy:.10f} '
        f'virial="{virial_str}" '
        f'pbc="T T T" '
        f'config_type=cp2k2xyz '
        f'dirID="{dir_id}"'
    )
    lines.append(info + '\n')
    for i, atom in enumerate(atoms):
        el, x, y, z = atom
        fx = fy = fz = 0.0
        if i < len(forces):
            fx, fy, fz = forces[i]
        lines.append(f"{el:2s} {x:14.8f} {y:14.8f} {z:14.8f} {fx:14.8f} {fy:14.8f} {fz:14.8f}\n")
    return ''.join(lines)


def run_cp2k2xyz():
    """Interactive CP2K .inp/.log → extxyz converter."""
    print_section("CP2K .inp + .log → extxyz")

    search_dir = input(" Directory to scan for .inp files [default: .]: ").strip() or "."
    search_path = Path(search_dir)
    if not search_path.is_dir():
        print_error(f"Directory not found: {search_dir}")
        return

    out_name = input(" Output file name [default: cp2k_dataset.xyz]: ").strip() or "cp2k_dataset.xyz"

    inp_files = sorted(search_path.rglob("*.inp"), key=_natural_sort_key)
    if not inp_files:
        print_warning("No .inp files found.")
        return

    error_counts = {
        'missing_coords': 0, 'log_not_found': 0, 'missing_energy': 0,
        'missing_forces': 0, 'missing_stress': 0, 'exception': 0,
    }
    success_count = 0
    all_frames = []

    for inp in inp_files:
        try:
            inp_text = inp.read_text(encoding='utf-8', errors='ignore')
            atoms = _extract_atoms(inp_text)
            if not atoms:
                error_counts['missing_coords'] += 1
                continue

            log_file = _find_log_for_inp(inp)
            if not log_file:
                error_counts['log_not_found'] += 1
                continue

            log_text = log_file.read_text(encoding='utf-8', errors='ignore')
            energy = _extract_energy(log_text)
            forces = _extract_forces(log_text)
            stress = _extract_stress_gpa(log_text)

            if energy != energy:  # NaN
                error_counts['missing_energy'] += 1
                continue
            if not forces:
                error_counts['missing_forces'] += 1
                continue
            if len(stress) != 9:
                error_counts['missing_stress'] += 1
                continue

            lattice = _extract_lattice(inp_text)
            volume = _compute_volume(lattice)
            virial = _stress_to_virial(stress, volume)
            dir_id = inp.parent.name

            all_frames.append(_format_frame(lattice, atoms, energy, virial, forces, dir_id))
            success_count += 1
        except Exception:
            error_counts['exception'] += 1
            continue

    if all_frames:
        Path(out_name).write_text(''.join(all_frames), encoding='utf-8')

    print_section("CP2K → extxyz Conversion Completed")
    print_kv("Output File", str(Path(out_name).resolve()))
    print_kv("Configurations", str(success_count))
    total = len(inp_files)
    failed = total - success_count
    if failed > 0:
        print_warning(f"{failed}/{total} files skipped:")
        for cat, cnt in error_counts.items():
            if cnt > 0:
                print(f"           {cat}: {cnt}")
    print_sep()
    print()
    sys.exit(0)
