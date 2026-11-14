"""ExtXYZ standardization -- normalize labels and convert stress/virial."""

import hashlib
import os
import re

import numpy as np

from ai2pot_cli.menu import print_section, print_success, print_warning, print_error, print_kv


NET_FORCE_TOL = 1e-3      # eV/A
FORCE_OUTLIER_TOL = 1e2   # eV/A


def _virial_6_to_9(v_6: np.ndarray) -> np.ndarray:
    """6-comp [xx,yy,zz,yz,xz,xy] -> 9-comp [xx,xy,xz, yx,yy,yz, zx,zy,zz]."""
    v_6 = np.atleast_2d(v_6)
    n = v_6.shape[0]
    v_9 = np.zeros((n, 9), dtype=v_6.dtype)
    v_9[:, 0] = v_6[:, 0]   # xx
    v_9[:, 1] = v_6[:, 5]   # xy
    v_9[:, 2] = v_6[:, 4]   # xz
    v_9[:, 3] = v_6[:, 5]   # yx = xy
    v_9[:, 4] = v_6[:, 1]   # yy
    v_9[:, 5] = v_6[:, 3]   # yz
    v_9[:, 6] = v_6[:, 4]   # zx = xz
    v_9[:, 7] = v_6[:, 3]   # zy = yz
    v_9[:, 8] = v_6[:, 2]   # zz
    return v_9


def _normalize_energy_keys(comment: str):
    """Keep exactly one energy key, dropping duplicates such as free_energy.

    ASE's ``get_potential_energy()`` reads ``energy``; a frame carrying only
    ``free_energy`` is renamed so it stays trainable.  When both exist,
    ``energy`` is kept and the others are dropped.

    Returns ``(new_comment, renamed, dropped)``.
    """
    pattern = re.compile(r'\s*\b(\w*energy)\s*=\s*(-?[\d.eE+-]+)')
    names = [m.group(1) for m in pattern.finditer(comment)]
    if not names:
        return comment, False, 0
    if "energy" in names:
        keep = "energy"
    elif "free_energy" in names:
        keep = "free_energy"
    else:
        keep = names[0]

    dropped = 0

    def _keep_or_drop(m):
        nonlocal dropped
        if m.group(1) == keep:
            return m.group(0)
        dropped += 1
        return ""

    new_comment = pattern.sub(_keep_or_drop, comment)
    renamed = keep != "energy"
    if renamed:
        new_comment = re.sub(
            r'\b' + re.escape(keep) + r'\s*=', 'energy=', new_comment, count=1)
    return new_comment, renamed, dropped


def _normalize_comment_fields(comment: str, vol: float):
    """Normalize energy / virial-stress keys in comment-line key=value pairs.

    Returns ``(new_comment, energy_renamed, energy_dropped,
    stress_conv, virial6_expand)``.
    """
    comment, energy_renamed, energy_dropped = _normalize_energy_keys(comment)
    stress_conv = False
    virial6_expand = False

    # ASE writer artifact (frame counter, no physical meaning)
    comment = re.sub(r'\s*\bindex\s*=\s*[-+]?\d+', '', comment)

    # virial_stress -> virial (rename key only, data unchanged)
    comment = re.sub(r'\bvirial_stress\b', 'virial', comment)

    # stress="..." -> virial="..."  (virial = -stress * vol)
    def _stress_to_virial(m):
        nonlocal stress_conv
        stress_conv = True
        vals = np.array([float(x) for x in m.group(1).split()])
        virial_vals = -vals * vol
        if len(vals) == 6:
            virial_vals = _virial_6_to_9(virial_vals.reshape(1, -1))[0]
        return 'virial="' + " ".join(f"{v:.12e}" for v in virial_vals) + '"'

    comment = re.sub(r'\bstress\s*=\s*"([^"]*)"', _stress_to_virial, comment)

    # virial="..." 6-comp -> 9-comp
    def _expand_virial(m):
        vals = [float(x) for x in m.group(1).split()]
        if len(vals) == 6:
            nonlocal virial6_expand
            virial6_expand = True
            v9 = _virial_6_to_9(np.array(vals).reshape(1, -1))[0]
            return 'virial="' + " ".join(f"{v:.12e}" for v in v9) + '"'
        return m.group(0)

    comment = re.sub(r'\bvirial\s*=\s*"([^"]*)"', _expand_virial, comment)

    return comment, energy_renamed, energy_dropped, stress_conv, virial6_expand


def _first_frame_labels(path: str):
    """Parse positions / forces / virial / periodicity from the first frame.

    Returns ``(positions, forces, virial, periodic)``; any missing label is
    ``None``. Extraction is purely text-level (no ASE dependency).
    """
    with open(path, "r") as f:
        line = f.readline()
        while line and line.strip() == "":
            line = f.readline()
        if not line:
            return None
        try:
            natoms = int(line.strip())
        except ValueError:
            return None
        comment = f.readline()
        if not comment:
            return None
        atom_lines = [f.readline() for _ in range(natoms)]
        if any(not al for al in atom_lines):
            return None

    props_m = re.search(r"Properties=(\S+)", comment)
    if not props_m:
        return None
    offsets: dict[str, tuple[int, int]] = {}
    parts = props_m.group(1).split(":")
    col = 0
    for j in range(0, len(parts) - 2, 3):
        name, ptype, count = parts[j], parts[j + 1], int(parts[j + 2])
        if ptype == "R":
            offsets[name] = (col, count)
            col += count

    if offsets.get("pos", (0, 0))[1] != 3:
        return None
    pos_start = offsets["pos"][0]
    f_start, f_count = offsets.get("forces", (None, 0))

    positions = np.empty((natoms, 3))
    forces = np.empty((natoms, 3)) if f_count == 3 else None
    try:
        for k, aline in enumerate(atom_lines):
            tokens = aline.split()
            positions[k] = [float(t) for t in tokens[1 + pos_start: 1 + pos_start + 3]]
            if forces is not None:
                forces[k] = [float(t) for t in tokens[1 + f_start: 1 + f_start + 3]]
    except (ValueError, IndexError):
        return None

    # ASE reads virial/stress as 3x3 in Fortran order.
    virial = None
    vir_m = re.search(r'\bvirial\s*=\s*"([^"]*)"', comment)
    if vir_m:
        vals = [float(x) for x in vir_m.group(1).split()]
        if len(vals) == 9:
            virial = np.array(vals).reshape(3, 3, order="F")
    elif offsets.get("virial", (0, 0))[1] == 9:
        v_start = offsets["virial"][0]
        tokens = atom_lines[0].split()
        try:
            vals = [float(t) for t in tokens[1 + v_start: 1 + v_start + 9]]
            virial = np.array(vals).reshape(3, 3, order="F")
        except (ValueError, IndexError):
            virial = None

    pbc_m = re.search(r'\bpbc\s*=\s*"([^"]*)"', comment)
    if pbc_m:
        periodic = any(t.upper().startswith("T") for t in pbc_m.group(1).split())
    else:
        periodic = re.search(r'\bLattice\s*=\s*"', comment) is not None

    return positions, forces, virial, periodic


def _check_force_virial(path: str):
    """Compare ``Σ_i r_i (x) F_i`` with the stored virial on the first frame.

    Exact only for non-periodic systems: under PBC the stress virial carries
    cell-derivative and periodic-image terms that the force sum cannot
    reproduce, even for a correct conversion.
    """
    try:
        data = _first_frame_labels(path)
    except (OSError, ValueError):
        data = None
    if data is None:
        return "skip", "first frame could not be parsed"
    positions, forces, virial, periodic = data
    if virial is None:
        return "skip", "no virial label"
    if forces is None:
        return "skip", "no force labels"
    if periodic:
        return "skip", "periodic cell: Σr⊗F ≠ stress virial under PBC"

    w_rf = np.einsum("ia,ib->ab", positions, forces)
    norm = np.linalg.norm(virial)
    rel = np.linalg.norm(w_rf - virial) / norm if norm > 0 else np.linalg.norm(w_rf)
    if rel < 1e-2:
        return "pass", f"Σr⊗F vs virial rel. dev. {rel:.2e}"
    return "warn", f"Σr⊗F vs virial rel. dev. {rel:.1%}"


def _parse_properties(comment: str):
    props_m = re.search(r"Properties=(\S+)", comment)
    if not props_m:
        return None, []
    parts = props_m.group(1).split(":")
    if len(parts) % 3:
        return None, []
    props = [[parts[j], parts[j + 1], int(parts[j + 2])]
             for j in range(0, len(parts), 3)]
    return props_m.group(1), props


def _column_map(props):
    col_map = {}
    col = 0
    for name, ptype, count in props:
        if ptype == "R":
            col_map[name] = (col, count)
            col += count
    return col_map


def _convert_frame(comment: str, atom_lines: list[str], stats: dict):
    """Normalize one frame in place (as strings); returns new lines."""
    natoms = len(atom_lines)

    lattice_m = re.search(r'Lattice="([^"]*)"', comment)
    vol = 1.0
    lattice_vals = None
    if lattice_m:
        lattice_vals = lattice_m.group(1).split()
        if len(lattice_vals) == 9:
            vol = abs(np.linalg.det(np.array(lattice_vals, dtype=float).reshape(3, 3)))

    old_props_str, old_props_list = _parse_properties(comment)
    new_comment = comment
    new_props_list = [p[:] for p in old_props_list]
    stress_idx = None
    virial_idx = None

    symbols = None
    nums = None
    try:
        rows = [aline.split() for aline in atom_lines]
        symbols = [r[0] for r in rows]
        nums = np.array([[float(x) for x in r[1:]] for r in rows])
    except (ValueError, IndexError):
        symbols = None
        nums = None

    if old_props_list:
        for idx, p in enumerate(old_props_list):
            if p[0] == "stress":
                stress_idx = idx
            if p[0] == "virial":
                virial_idx = idx

        # Normalize property names: *energy* -> energy, *force* -> forces
        for p in old_props_list:
            if "energy" in p[0] and p[0] != "energy":
                p[0] = "energy"
                stats["n_energy_label"] += 1
            if "force" in p[0] and p[0] != "forces":
                p[0] = "forces"
                stats["n_force_label"] += 1

        col_map = _column_map(old_props_list)

        # --- label quality checks on the source frame ---
        f_off = col_map.get("forces")
        if f_off is not None and f_off[1] == 3 and nums is not None:
            forces = nums[:, f_off[0]:f_off[0] + 3]
            if np.all(np.isfinite(forces)):
                net = float(np.linalg.norm(forces.sum(axis=0)))
                stats["max_net_force"] = max(stats["max_net_force"], net)
                if net > NET_FORCE_TOL:
                    stats["n_bad_net_force"] += 1
                fmax = float(np.max(np.abs(forces)))
                stats["max_abs_force"] = max(stats["max_abs_force"], fmax)
                if fmax > FORCE_OUTLIER_TOL:
                    stats["n_force_outliers"] += 1
            else:
                stats["n_nonfinite"] += 1

        pos_off = col_map.get("pos")
        if (symbols is not None and nums is not None
                and pos_off is not None and pos_off[1] == 3):
            digest = hashlib.md5()
            digest.update(f"{natoms}|{' '.join(symbols)}|".encode())
            digest.update(np.round(nums[:, pos_off[0]:pos_off[0] + 3], 5).tobytes())
            if lattice_vals is not None:
                digest.update(np.round(np.array(lattice_vals, dtype=float), 5).tobytes())
            h = digest.digest()
            if h in stats["_seen_geoms"]:
                stats["n_duplicates"] += 1
            else:
                stats["_seen_geoms"].add(h)

        # Build new properties list
        new_props_list = []
        stress_count = old_props_list[stress_idx][2] if stress_idx is not None else 0
        if stress_idx is not None and stress_count in (6, 9):
            stats["n_stress"] += 1
            for p in old_props_list:
                if p[0] == "stress":
                    new_props_list.append(["virial", "R", 9])
                elif p[0] == "virial":
                    continue  # drop duplicate virial
                else:
                    new_props_list.append(p[:])
        elif virial_idx is not None and old_props_list[virial_idx][2] == 6:
            stats["n_virial6"] += 1
            for p in old_props_list:
                if p[0] == "virial":
                    new_props_list.append(["virial", "R", 9])
                else:
                    new_props_list.append(p[:])
        else:
            new_props_list = [p[:] for p in old_props_list]

        new_props_str = ":".join(f"{n}:{t}:{c}" for n, t, c in new_props_list)
        new_comment = comment.replace(
            f"Properties={old_props_str}", f"Properties={new_props_str}"
        )

    # Normalize comment-line key=value fields (energy, stress, virial)
    new_comment, e_renamed, e_dropped, s_conv, v6_exp = _normalize_comment_fields(
        new_comment, vol)
    if e_renamed:
        stats["n_energy_label"] += 1
    if e_dropped:
        stats["n_energy_key_frames"] += 1
        m = {k: float(v) for k, v in re.findall(
            r'\s*\b(\w*energy)\s*=\s*(-?[\d.eE+-]+)', comment)}
        if "energy" in m and "free_energy" in m and natoms:
            stats["max_energy_diff"] = max(
                stats["max_energy_diff"],
                abs(m["energy"] - m["free_energy"]) / natoms)
    if s_conv:
        stats["n_stress"] += 1
    if v6_exp:
        stats["n_virial6"] += 1

    # --- transform atom data if stress/virial columns changed ---
    needs_transform = (
        (stress_idx is not None and old_props_list[stress_idx][2] in (6, 9))
        or (virial_idx is not None and old_props_list[virial_idx][2] == 6)
    )

    if needs_transform and old_props_list:
        if nums is None:
            raise ValueError("cannot parse atom lines for stress/virial conversion")
        col_map = _column_map(old_props_list)
        new_atom_lines = []
        for symbol, row in zip(symbols, nums):
            new_parts = []
            for name, ptype, count in new_props_list:
                if ptype != "R":
                    continue
                if name == "virial" and count == 9 and stress_idx is not None:
                    s_start, s_count = col_map["stress"]
                    virial_vals = -row[s_start:s_start + s_count] * vol
                    if s_count == 6:
                        virial_vals = _virial_6_to_9(virial_vals.reshape(1, -1))[0]
                    new_parts.append(virial_vals)
                elif name == "virial" and count == 9 and virial_idx is not None:
                    v_start, v_count = col_map["virial"]
                    virial_6 = row[v_start:v_start + v_count]
                    new_parts.append(_virial_6_to_9(virial_6.reshape(1, -1))[0])
                elif name in col_map:
                    old_start, old_c = col_map[name]
                    new_parts.append(row[old_start:old_start + old_c])
            new_nums = np.concatenate(new_parts) if new_parts else row
            new_atom_lines.append(
                symbol + " " + " ".join(f"{v:.12e}" for v in new_nums) + "\n")
    else:
        new_atom_lines = atom_lines

    # --- label coverage on the converted frame ---
    if re.search(r'\benergy\s*=', new_comment):
        stats["n_frames_energy"] += 1
    if any("force" in n and t == "R" and c == 3 for n, t, c in new_props_list):
        stats["n_frames_forces"] += 1
    has_virial = bool(re.search(r'\bvirial\s*=\s*"', new_comment))
    if has_virial or any(n == "virial" and t == "R" and c == 9
                         for n, t, c in new_props_list):
        stats["n_frames_virial"] += 1

    return new_comment, new_atom_lines


def _has_newlines(path: str) -> bool:
    with open(path, "r") as f:
        first = f.readline()
    return first.endswith("\n") if first else False


def standardize_extxyz(extxyz_path: str, output_path: str | None = None):
    """Standardize an extxyz file.

    1. Keep a single ``energy`` key (``free_energy`` is dropped; renamed if alone).
    2. Rename any label containing "energy" to ``energy``, any containing "force" to ``forces``.
    3. If ``stress`` is present, convert to ``virial`` (virial = -stress * volume);
       if both ``stress`` and ``virial`` exist, keep only ``virial``.
    4. If ``virial`` is 6-component, expand to 9-component (3x3 tensor).
    5. For non-periodic output, cross-check the stored virial against
       ``Σ_i r_i (x) F_i`` on the first frame (skipped under PBC).
    6. Report label coverage, force balance, force outliers and duplicate geometries.
    """
    if output_path is None:
        output_path = extxyz_path

    if not _has_newlines(extxyz_path):
        print()
        print_error(f"{os.path.basename(extxyz_path)}: file has no newline characters. "
                    "Split frames into separate lines first.")
        print()
        return

    in_place = output_path == extxyz_path
    tmp_path = output_path + ".tmp"
    bak_path = extxyz_path + ".bak" if in_place else None

    stats = {
        "frames": 0,
        "n_energy_label": 0,
        "n_force_label": 0,
        "n_stress": 0,
        "n_virial6": 0,
        "n_energy_key_frames": 0,
        "max_energy_diff": 0.0,
        "n_frames_energy": 0,
        "n_frames_forces": 0,
        "n_frames_virial": 0,
        "n_bad_net_force": 0,
        "max_net_force": 0.0,
        "n_force_outliers": 0,
        "max_abs_force": 0.0,
        "n_nonfinite": 0,
        "n_duplicates": 0,
        "_seen_geoms": set(),
    }

    try:
        with open(extxyz_path, "r") as fin, open(tmp_path, "w") as fout:
            while True:
                line = fin.readline()
                if not line:
                    break
                if line.strip() == "":
                    fout.write(line)
                    continue
                try:
                    natoms = int(line.strip())
                except ValueError:
                    fout.write(line)
                    continue

                comment = fin.readline()
                if not comment:
                    raise ValueError("truncated frame: missing comment line")
                atom_lines = [fin.readline() for _ in range(natoms)]
                if any(not al for al in atom_lines):
                    raise ValueError("truncated frame: missing atom lines")

                new_comment, new_atoms = _convert_frame(comment, atom_lines, stats)
                fout.write(line if line.endswith("\n") else line + "\n")
                fout.write(new_comment if new_comment.endswith("\n")
                           else new_comment + "\n")
                for aline in new_atoms:
                    fout.write(aline if aline.endswith("\n") else aline + "\n")
                stats["frames"] += 1
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if in_place:
        os.replace(extxyz_path, bak_path)
    os.replace(tmp_path, output_path)

    # --- summary ---
    changes = []
    if stats["n_energy_label"] > 0:
        changes.append(f"*energy* → energy ({stats['n_energy_label']} label(s))")
    if stats["n_force_label"] > 0:
        changes.append(f"*force* → forces ({stats['n_force_label']} label(s))")
    if stats["n_stress"] > 0:
        changes.append(f"stress → virial ({stats['n_stress']} frame(s))")
    if stats["n_virial6"] > 0:
        changes.append(f"virial 6-comp → 9-comp ({stats['n_virial6']} frame(s))")

    n = stats["frames"]
    print_section("Standardize ExtXYZ")
    print_kv("Input", extxyz_path)
    print_kv("Output", output_path)
    if bak_path:
        print_kv("Backup", bak_path)
    if changes:
        print_success(" | ".join(changes))
    else:
        print_success("no changes needed")

    if n:
        print_kv("Frames", str(n))
        print_kv("Labels", f"energy {stats['n_frames_energy']}/{n} | "
                           f"forces {stats['n_frames_forces']}/{n} | "
                           f"virial {stats['n_frames_virial']}/{n}")
        if stats["n_energy_key_frames"]:
            msg = f"normalized in {stats['n_energy_key_frames']} frame(s)"
            if stats["max_energy_diff"] > 0:
                msg += f", max |free_energy-energy| = {stats['max_energy_diff']:.2e} eV/atom"
            print_kv("Energy Keys", msg)
        if stats["n_bad_net_force"]:
            print_warning(
                f"force balance: {stats['n_bad_net_force']}/{n} frame(s) with "
                f"|ΣF| > {NET_FORCE_TOL:.0e} eV/A (max {stats['max_net_force']:.2e})")
        else:
            print_kv("Force Balance",
                     f"OK (max |ΣF| = {stats['max_net_force']:.2e} eV/A)")
        if stats["n_force_outliers"]:
            print_warning(
                f"force outliers: {stats['n_force_outliers']} frame(s) with max|F| > "
                f"{FORCE_OUTLIER_TOL:.0e} eV/A (file max {stats['max_abs_force']:.2e})")
        if stats["n_nonfinite"]:
            print_warning(f"non-finite force values in {stats['n_nonfinite']} frame(s)")
        if stats["n_duplicates"]:
            print_warning(
                f"duplicate geometries: {stats['n_duplicates']} frame(s) repeat "
                "an earlier structure")

    status, msg = _check_force_virial(output_path)
    if status == "pass":
        print_kv("Virial Check", f"PASS (frame 1, {msg})")
    elif status == "warn":
        print_warning(f"Virial check (frame 1): {msg}")
    else:
        print_kv("Virial Check", f"skipped ({msg})")
    print()
