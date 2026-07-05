"""Convert CASTEP .castep output to extxyz format.

Scans for .castep files, extracts lattice/energy/stress/coordinates/forces,
converts stress(GPa)→virial(eV), and writes an extxyz file.
"""

import os
import re
import sys
import numpy as np

from ai2pot_cli.menu import print_section, print_warning, print_error, print_kv, print_sep


def _parse_castep(filepath):
    """Parse a single .castep file and return frame data dict, or None on failure."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    m = re.search(r'Total number of ions in cell\s*=\s*(\d+)', content)
    if not m:
        return None
    natoms = int(m.group(1))

    m = re.search(r'Unit Cell\s*\n\s*-+\s*\n(?:.*?)\n(.*?)\n(.*?)\n(.*?)\n', content)
    if not m:
        return None
    lattice = []
    for i in range(1, 4):
        parts = m.group(i).split()
        lattice.extend([float(x) for x in parts[:3]])
    if len(lattice) != 9:
        return None

    m = re.search(r'Final energy\s*=\s*([\d.\-E\+]+)\s*eV', content)
    if not m:
        return None
    energy = float(m.group(1))

    # Stress tensor: locate "Stress Tensor" then read 3 numeric lines after the separator
    virial = None
    lines = content.splitlines()
    stress_start = None
    for idx, line in enumerate(lines):
        if 'Stress Tensor' in line:
            stress_start = idx
            break
    if stress_start is not None:
        stress_vals = []
        for offset in range(1, 10):  # scan up to 10 lines after "Stress Tensor"
            if stress_start + offset >= len(lines):
                break
            parts = lines[stress_start + offset].split()
            try:
                vals = [float(x) for x in parts[-3:]]
                if len(vals) == 3:
                    stress_vals.extend(vals)
                if len(stress_vals) >= 9:
                    break
            except ValueError:
                continue
        if len(stress_vals) == 9:
            vm = re.search(r'Current cell volume\s*=\s*([\d.\-E\+]+)', content)
            if vm:
                volume = float(vm.group(1))
                factor = volume / 160.217662
                virial = [s * factor for s in stress_vals]

    m = re.search(r'Fractional coordinates of atoms\s*\n\s*-+\s*\n(.*?)(?=\n\s*\n|\n\s*\*|\Z)', content, re.DOTALL)
    if not m:
        return None
    frac_lines = m.group(1).strip().splitlines()
    symbols = []
    frac_coords = []
    for line in frac_lines:
        parts = line.split()
        # Data lines: Element AtomNr u v w  (5 cols) or Element AtomNr X u v w (6 cols)
        # Header/separator lines will fail float conversion and be skipped
        if len(parts) >= 5:
            try:
                u = float(parts[-3])
                v = float(parts[-2])
                w = float(parts[-1])
                symbols.append(parts[0])
                frac_coords.append([u, v, w])
            except ValueError:
                continue

    lattice_3x3 = np.array(lattice).reshape(3, 3)
    cart = np.array(frac_coords) @ lattice_3x3

    # Forces: locate "Forces" header then read natoms numeric lines
    forces = np.zeros((natoms, 3))
    force_start = None
    for idx, line in enumerate(lines):
        if 'Forces' in line:
            force_start = idx
            break
    if force_start is not None:
        fcount = 0
        for offset in range(1, 20):
            if force_start + offset >= len(lines) or fcount >= natoms:
                break
            parts = lines[force_start + offset].split()
            try:
                vals = [float(x) for x in parts[-3:]]
                if len(vals) == 3:
                    forces[fcount] = vals
                    fcount += 1
            except ValueError:
                continue

    return {
        'natoms': natoms, 'lattice': lattice, 'energy': energy,
        'virial': virial, 'symbols': symbols, 'cart': cart, 'forces': forces,
    }


def run_castep2exyz():
    """Interactive CASTEP → extxyz converter."""
    print_section("CASTEP .castep → extxyz")

    search_dir = input(" Directory to scan for .castep files [default: .]: ").strip() or "."
    if not os.path.isdir(search_dir):
        print_error(f"Directory not found: {search_dir}")
        return

    out_name = input(" Output file name [default: NEP-dataset.xyz]: ").strip() or "NEP-dataset.xyz"

    castep_files = []
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f.endswith('.castep'):
                castep_files.append(os.path.join(root, f))

    if not castep_files:
        print_warning("No .castep files found.")
        return

    success = 0
    skipped = 0
    with open(out_name, 'w') as fo:
        for fp in castep_files:
            data = _parse_castep(fp)
            if data is None:
                skipped += 1
                continue

            fo.write(f"{data['natoms']}\n")
            lattice_str = " ".join(f"{x:.6f}" for x in data['lattice'])
            parts = [
                f'Lattice="{lattice_str}"',
                f'Properties=species:S:1:pos:R:3:forces:R:3',
                f'energy={data["energy"]:.6f}',
            ]
            if data['virial'] is not None:
                virial_str = " ".join(f"{x:.6f}" for x in data['virial'])
                parts.append(f'virial="{virial_str}"')
            parts.append('pbc="T T T"')
            parts.append('config_type=castep2exyz')
            fo.write(' '.join(parts) + '\n')
            for j in range(data['natoms']):
                sym = data['symbols'][j]
                x, y, z = data['cart'][j]
                fx, fy, fz = data['forces'][j]
                fo.write(f"{sym} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\n")
            success += 1

    if success == 0:
        print_warning("No structures were successfully parsed.")
        return

    print_section("CASTEP → extxyz Conversion Completed")
    print_kv("Output File", os.path.abspath(out_name))
    print_kv("Configurations", f"{success}/{len(castep_files)}")
    if skipped > 0:
        print_warning(f"{skipped} file(s) skipped (unable to parse)")
    print_sep()
    print()
    sys.exit(0)
