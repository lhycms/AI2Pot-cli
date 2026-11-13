"""Convert ASE extended-XYZ (energy / forces / virial) to mlip-3 cfg format.

Scans a directory recursively for extxyz files (*.xyz / *.extxyz) and writes
every configuration to a unified mlip-3 cfg file.

The element order entered by the user defines the integer atom types written
to the cfg (Ge,Sb,Te -> 0,1,2 by default), matching the project's
LAMMPS/POTCAR ordering. The virial in the extxyz file is in eV with the
"positive = compressed" convention, which is exactly what mlip-3 expects for
the PlusStress field.
"""

import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

from ai2pot_cli.menu import print_section, print_warning, print_error, print_kv, print_sep

KEYVAL = re.compile(r'([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*("([^"]*)"|\{[^}]*\}|[^\s]+)')


def _natural_sort_key(name: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(name))]


def _collect_extxyz(search_path: Path):
    """Collect extended-XYZ files recursively."""
    sources = []
    for root, dirs, files in os.walk(search_path):
        dirs.sort(key=_natural_sort_key)
        for fname in sorted(files, key=_natural_sort_key):
            if fname.lower().endswith('.xyz') or fname.lower().endswith('.extxyz'):
                sources.append(Path(root) / fname)
    return sources


def _parse_header(line):
    fields = {}
    for m in KEYVAL.finditer(line):
        key, raw = m.group(1), m.group(2)
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
        elif raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        fields[key] = raw
    return fields


def _parse_properties(props):
    tokens = props.split(":")
    if len(tokens) % 3 != 0:
        raise ValueError("malformed Properties field: %s" % props)
    columns = []
    for i in range(0, len(tokens), 3):
        name, kind, count = tokens[i], tokens[i + 1], int(tokens[i + 2])
        columns.append((name, kind, count))
    return columns


def _convert_frame(lines, pos, type_of_element):
    natoms = int(lines[pos].strip())
    if natoms <= 0:
        raise ValueError("bad atom count on line %d" % (pos + 1))
    header = _parse_header(lines[pos + 1])
    pos += 2

    if "Lattice" not in header:
        raise ValueError("frame at line %d has no Lattice" % (pos - 1))
    lattice = [float(x) for x in header["Lattice"].split()]
    if len(lattice) != 9:
        raise ValueError("Lattice must have 9 values")

    if "energy" not in header:
        raise ValueError("frame at line %d has no energy" % (pos - 1))
    energy = float(header["energy"])

    if "virial" not in header:
        raise ValueError("frame at line %d has no virial" % (pos - 1))
    w = [float(x) for x in header["virial"].split()]
    if len(w) != 9:
        raise ValueError("virial must have 9 values")
    # xyz stores the 3x3 tensor as 9 numbers; symmetrize (symmetric by physics)
    # and pick xx yy zz yz xz xy for PlusStress.
    xx, xy, xz = w[0], w[1], w[2]
    yx, yy, yz = w[3], w[4], w[5]
    zx, zy, zz = w[6], w[7], w[8]
    virial = (
        0.5 * (xx + xx),
        0.5 * (yy + yy),
        0.5 * (zz + zz),
        0.5 * (yz + zy),
        0.5 * (xz + zx),
        0.5 * (xy + yx),
    )

    if "Properties" not in header:
        raise ValueError("frame at line %d has no Properties field" % (pos - 1))
    columns = _parse_properties(header["Properties"])
    names = [c[0] for c in columns]
    if "species" not in names or "pos" not in names or "forces" not in names:
        raise ValueError("frame at line %d lacks species/pos/forces" % (pos - 1))

    if pos + natoms > len(lines):
        raise ValueError("truncated frame: expected %d atom lines" % natoms)

    atoms = []
    for k in range(natoms):
        tokens = lines[pos + k].split()
        cursor = 0
        record = {}
        for name, kind, count in columns:
            if kind in ("S", "R", "I"):
                record[name] = tokens[cursor:cursor + count]
                cursor += count
            else:
                raise ValueError("unsupported column kind %s" % kind)
        elem = record["species"][0]
        if elem not in type_of_element:
            raise ValueError(
                "element %r not in element order %s"
                % (elem, sorted(type_of_element)))
        xyz = [float(v) for v in record["pos"]]
        fx, fy, fz = [float(v) for v in record["forces"]]
        atoms.append((k + 1, type_of_element[elem], xyz, fx, fy, fz))

    pbc = header.get("pbc", "T T T").split()
    if pbc != ["T", "T", "T"]:
        raise ValueError("frame at line %d is not fully periodic (pbc=%s)"
                         % (pos - 1, header.get("pbc")))

    return natoms, lattice, energy, virial, atoms, pos + natoms


def _write_cfg(fout, natoms, lattice, energy, virial, atoms):
    fout.write("BEGIN_CFG\n")
    fout.write(" Size\n")
    fout.write("    %d\n" % natoms)
    fout.write(" Supercell\n")
    for r in range(3):
        fout.write("    %18.10f %18.10f %18.10f\n"
                   % (lattice[3 * r], lattice[3 * r + 1], lattice[3 * r + 2]))
    fout.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z"
               "           fx          fy          fz\n")
    for (idx, typ, xyz, fx, fy, fz) in atoms:
        fout.write("    %6d %4d %18.10f %18.10f %18.10f %14.10f %14.10f %14.10f\n"
                   % (idx, typ, xyz[0], xyz[1], xyz[2], fx, fy, fz))
    fout.write(" Energy\n")
    fout.write("    %.10f\n" % energy)
    fout.write(" PlusStress:  xx          yy          zz"
               "          yz          xz          xy\n")
    fout.write("     %.10f %.10f %.10f %.10f %.10f %.10f\n" % virial)
    fout.write("END_CFG\n\n")


def _read_frames(src: Path, type_of_element):
    with open(src, 'r', errors='ignore') as f:
        lines = f.read().splitlines()

    frames = []
    pos = 0
    while pos < len(lines):
        while pos < len(lines) and not lines[pos].strip():
            pos += 1
        if pos >= len(lines):
            break
        frame = _convert_frame(lines, pos, type_of_element)
        frames.append(frame)
        pos = frame[-1]
    return frames


def run_xyz2cfg():
    """Interactive extxyz → mlip-3 cfg converter."""
    print_section("extxyz → MLIP cfg")

    search_dir = input(" Directory to scan for extxyz files [default: .]: ").strip() or "."
    search_path = Path(search_dir)
    if not search_path.is_dir():
        print_error(f"Directory not found: {search_dir}")
        return

    out_name = input(" Output file name [default: cfg_dataset.cfg]: ").strip() or "cfg_dataset.cfg"

    elem_str = input(" Element order for atom types (comma-separated) [default: Ge,Sb,Te]: ").strip()
    elements = [e.strip() for e in (elem_str or "Ge,Sb,Te").split(",") if e.strip()]
    type_of_element = {e: i for i, e in enumerate(elements)}

    sources = _collect_extxyz(search_path)
    if not sources:
        print_warning(f"No extxyz files (*.xyz / *.extxyz) found under: {search_dir}")
        return

    error_count = 0
    frame_count = 0
    total = len(sources)

    pbar = tqdm(desc=" Processing", unit="frame")
    with open(out_name, 'w') as fout:
        for src in sources:
            try:
                frames = _read_frames(src, type_of_element)
            except Exception:
                error_count += 1
                continue
            for natoms, lattice, energy, virial, atoms, _ in frames:
                _write_cfg(fout, natoms, lattice, energy, virial, atoms)
                frame_count += 1
                pbar.update(1)
    pbar.close()

    if frame_count == 0:
        Path(out_name).unlink(missing_ok=True)

    print_section("extxyz → MLIP cfg Conversion Completed")
    if frame_count:
        print_kv("Output File", os.path.abspath(out_name))
    print_kv("Configurations", str(frame_count))
    if error_count > 0:
        print_warning(f"{error_count}/{total} file(s) failed to parse")
    print_sep()
    print()
    sys.exit(0)
