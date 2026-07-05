"""Convert ABACUS output (running_scf.log / abacus.json) to extxyz format.

Recursively scans a directory for ABACUS SCF outputs, extracts energy/forces/
stress/lattice, converts stress→virial, and writes train.xyz.
"""

import os
import sys
import json
import numpy as np
from ase.io import read

from ai2pot_cli.menu import print_section, print_warning, print_error, print_kv, print_sep


def _get_scf_nmax(root):
    """Read scf_nmax from INPUT file."""
    input_file = os.path.join(root, "INPUT")
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            for line in f:
                if 'scf_nmax' in line.strip():
                    return int(line.split()[1])
    return 100


def run_abacus2xyz():
    """Interactive ABACUS → extxyz converter."""
    print_section("ABACUS → extxyz")

    search_dir = input(" Directory to scan for ABACUS outputs [default: .]: ").strip() or "."
    search_dir = os.path.abspath(search_dir)
    if not os.path.isdir(search_dir):
        print_error(f"Directory not found: {search_dir}")
        return

    out_name = input(" Output file name [default: train.xyz]: ").strip() or "train.xyz"

    frame_count = 0
    err_count = 0

    with open(out_name, 'w') as fout:
        for root, dirs, files in os.walk(search_dir):
            scf_count, Total_Time, virial = 0, 0, None

            if "running_scf.log" in files:
                log_file = os.path.join(root, "running_scf.log")
                scf_nmax = _get_scf_nmax(root)
                with open(log_file, 'r') as fl:
                    for line in fl:
                        scf_count += line.count("ALGORITHM")
                        Total_Time += line.count("Total  Time")

                if scf_count == scf_nmax or Total_Time == 0:
                    continue

                try:
                    atoms = read(log_file, format='abacus-out')
                except Exception:
                    err_count += 1
                    continue

                natoms = len(atoms)
                cell = np.concatenate([atoms.get_cell()[0], atoms.get_cell()[1], atoms.get_cell()[2]])
                energy = atoms.get_potential_energy()
                if atoms.calc.get_stress() is not None:
                    xx, yy, zz, yz, xz, xy = atoms.calc.get_stress()
                    stresses = [xx, xy, xz, xy, yy, yz, xz, yz, zz]
                    virial = [f"{-s:.10f}" for s in [stress * atoms.get_volume() for stress in stresses]]
                symbols = atoms.get_chemical_symbols()
                positions = atoms.get_positions()
                forces = atoms.get_forces()

            elif "abacus.json" in files and "running_scf.log" not in files:
                json_file = os.path.join(root, "abacus.json")
                scf_nmax = _get_scf_nmax(root)
                with open(json_file, 'r', encoding='utf-8') as fj:
                    data = json.load(fj)
                scf_count = len(data['output'][0]['scf'])
                if scf_count == scf_nmax:
                    continue

                natoms = data['init']['natom']
                energy = data['output'][0]['energy']
                cells = data['output'][0]['cell']
                cell = np.concatenate([cells[0], cells[1], cells[2]])
                symbols = data['init']['label']
                positions = data['output'][0]['coordinate']
                forces = data['output'][0]['force']
                if 'stress' in open(json_file, 'r').read():
                    volume = np.abs(np.dot(cells[0], np.cross(cells[1], cells[2])))
                    stresses = data['output'][0]['stress']
                    stress = np.concatenate([stresses[0], stresses[1], stresses[2]])
                    virial = [s * (volume / 1602.1766208) for s in stress]
            else:
                continue

            fout.write(f"{natoms}\n")
            lattice_str = " ".join(map(str, cell))
            if virial is not None:
                virial_str = " ".join(map(str, virial))
                fout.write(f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy} virial="{virial_str}" pbc="T T T" config_type=abacus2xyz\n')
            else:
                fout.write(f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy} pbc="T T T" config_type=abacus2xyz\n')
            for i in range(natoms):
                fout.write(f"{symbols[i]:<20}{positions[i][0]:20.10f}{positions[i][1]:20.10f}{positions[i][2]:20.10f}{forces[i][0]:20.10f}{forces[i][1]:20.10f}{forces[i][2]:20.10f}\n")
            frame_count += 1

    if frame_count == 0:
        print_warning("No valid ABACUS outputs found.")
        return

    print_section("ABACUS → extxyz Conversion Completed")
    print_kv("Output File", os.path.abspath(out_name))
    print_kv("Configurations", str(frame_count))
    if err_count > 0:
        print_warning(f"{err_count} file(s) failed to parse")
    print_sep()
    print()
    sys.exit(0)
