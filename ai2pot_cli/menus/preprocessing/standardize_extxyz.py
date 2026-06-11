"""ExtXYZ standardization -- normalize labels and convert stress/virial."""

import re
import os
import numpy as np

from ai2pot_cli.menu import print_section, print_success, print_warning, print_error, print_kv


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


def _normalize_comment_fields(comment: str, vol: float):
    """Normalize energy / virial-stress keys in comment-line key=value pairs.

    Returns ``(new_comment, energy_renamed, stress_conv, virial6_expand)``.
    """
    energy_renamed = False
    stress_conv = False
    virial6_expand = False

    # 1. *energy*=scalar -> energy=scalar
    if re.search(r'\b(?!energy\b)\w*energy\w*\s*=\s*[-\d]', comment):
        energy_renamed = True
    comment = re.sub(r'\b\w*energy\w*\s*=\s*(-?[\d.eE+-]+)', r'energy=\1', comment)

    # 2. virial_stress -> virial (rename key only, data unchanged)
    comment = re.sub(r'\bvirial_stress\b', 'virial', comment)

    # 3. stress="..." -> virial="..."  (virial = -stress * vol)
    def _stress_to_virial(m):
        nonlocal stress_conv
        stress_conv = True
        vals = np.array([float(x) for x in m.group(1).split()])
        virial_vals = -vals * vol
        if len(vals) == 6:
            virial_vals = _virial_6_to_9(virial_vals.reshape(1, -1))[0]
        return 'virial="' + " ".join(f"{v:.12e}" for v in virial_vals) + '"'

    comment = re.sub(r'\bstress\s*=\s*"([^"]*)"', _stress_to_virial, comment)

    # 4. virial="..." 6-comp -> 9-comp
    def _expand_virial(m):
        vals = [float(x) for x in m.group(1).split()]
        if len(vals) == 6:
            nonlocal virial6_expand
            virial6_expand = True
            v9 = _virial_6_to_9(np.array(vals).reshape(1, -1))[0]
            return 'virial="' + " ".join(f"{v:.12e}" for v in v9) + '"'
        return m.group(0)

    comment = re.sub(r'\bvirial\s*=\s*"([^"]*)"', _expand_virial, comment)

    return comment, energy_renamed, stress_conv, virial6_expand


def standardize_extxyz(extxyz_path: str, output_path: str | None = None):
    """Standardize an extxyz file.

    1. Rename any label containing "energy" to ``energy``, any containing "force" to ``forces``.
    2. If ``stress`` is present, convert to ``virial`` (virial = -stress * volume);
       if both ``stress`` and ``virial`` exist, keep only ``virial``.
    3. If ``virial`` is 6-component, expand to 9-component (3x3 tensor).
    """
    if output_path is None:
        output_path = extxyz_path

    with open(extxyz_path, "r") as f:
        content = f.read()

    if "\n" not in content:
        print()
        print_error(f"{os.path.basename(extxyz_path)}: file has no newline characters. "
                    "Split frames into separate lines first.")
        print()
        return

    lines = content.split("\n")
    new_lines: list[str] = []
    i = 0

    n_energy = 0
    n_force = 0
    n_stress = 0
    n_virial6 = 0

    while i < len(lines):
        line = lines[i]

        # Preserve blank lines
        if line.strip() == "":
            new_lines.append(line)
            i += 1
            continue

        # Detect frame header (natoms)
        try:
            natoms = int(line.strip())
        except ValueError:
            new_lines.append(line)
            i += 1
            continue

        new_lines.append(line)  # natoms
        i += 1

        comment = lines[i]

        # --- volume from Lattice ---
        lattice_m = re.search(r'Lattice="([^"]*)"', comment)
        vol = 1.0
        if lattice_m:
            vals = [float(x) for x in lattice_m.group(1).split()]
            if len(vals) == 9:
                vol = abs(np.linalg.det(np.array(vals).reshape(3, 3)))

        # --- parse Properties ---
        props_m = re.search(r"Properties=(\S+)", comment)
        new_comment = comment

        stress_idx: int | None = None
        virial_idx: int | None = None
        old_props_list: list[list] = []   # [[name, type, count], ...]
        old_ncols = 0

        if props_m:
            old_props_str = props_m.group(1)
            parts = old_props_str.split(":")
            j = 0
            while j < len(parts):
                name, ptype, count = parts[j], parts[j+1], int(parts[j+2])
                old_props_list.append([name, ptype, count])
                if ptype == "R":
                    old_ncols += count
                j += 3

            for idx, p in enumerate(old_props_list):
                if p[0] == "stress":
                    stress_idx = idx
                if p[0] == "virial":
                    virial_idx = idx

            # Normalize property names: *energy* -> energy, *force* -> forces
            for p in old_props_list:
                if "energy" in p[0] and p[0] != "energy":
                    p[0] = "energy"
                    n_energy += 1
                if "force" in p[0] and p[0] != "forces":
                    p[0] = "forces"
                    n_force += 1

            # Build new properties list
            new_props_list = []
            if stress_idx is not None:
                n_stress += 1
                for p in old_props_list:
                    if p[0] == "stress":
                        new_props_list.append(["virial", "R", 9])
                    elif p[0] == "virial":
                        continue  # drop duplicate virial
                    else:
                        new_props_list.append(p[:])
            elif virial_idx is not None and old_props_list[virial_idx][2] == 6:
                n_virial6 += 1
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
        new_comment, e_renamed, s_conv, v6_exp = _normalize_comment_fields(new_comment, vol)
        if e_renamed:
            n_energy += 1
        if s_conv:
            n_stress += 1
        if v6_exp:
            n_virial6 += 1

        new_lines.append(new_comment)
        i += 1

        # --- read atom lines ---
        atom_lines = []
        for _ in range(natoms):
            atom_lines.append(lines[i])
            i += 1

        # --- transform atom data ---
        needs_transform = (stress_idx is not None) or (
            virial_idx is not None and old_props_list[virial_idx][2] == 6
        )

        if needs_transform and props_m:
            for aline in atom_lines:
                tokens = aline.strip().split()
                symbol = tokens[0]
                nums = np.array([float(t) for t in tokens[1:]])

                # Build column-offset map from OLD properties
                col_map = {}
                col = 0
                for name, ptype, count in old_props_list:
                    if ptype == "R":
                        col_map[name] = (col, count)
                        col += count

                # Rebuild numeric data in the NEW property order
                new_parts = []
                for name, ptype, count in new_props_list:
                    if ptype != "R":
                        continue
                    if name == "virial" and count == 9 and stress_idx is not None:
                        # stress -> virial conversion
                        s_start, s_count = col_map.get("stress", (0, 0))
                        stress_vals = nums[s_start : s_start + s_count]
                        virial_6 = -stress_vals * vol
                        virial_9 = _virial_6_to_9(virial_6.reshape(1, -1))[0]
                        new_parts.append(virial_9)
                    elif name == "virial" and count == 9 and virial_idx is not None:
                        # virial 6 -> 9 expansion
                        v_start, v_count = col_map["virial"]
                        virial_6 = nums[v_start : v_start + v_count]
                        virial_9 = _virial_6_to_9(virial_6.reshape(1, -1))[0]
                        new_parts.append(virial_9)
                    elif name in col_map:
                        old_start, old_c = col_map[name]
                        new_parts.append(nums[old_start : old_start + old_c])

                new_nums = np.concatenate(new_parts) if new_parts else nums
                new_line = symbol + " " + " ".join(f"{v:.12e}" for v in new_nums)
                new_lines.append(new_line + "\n")
        else:
            for aline in atom_lines:
                new_lines.append(aline)

    # --- write output: backup original, write standardized file ---
    if output_path == extxyz_path:
        bak_path = extxyz_path + ".bak"
        os.rename(extxyz_path, bak_path)
        with open(output_path, "w") as f:
            f.write("".join(new_lines))
    else:
        bak_path = None
        with open(output_path, "w") as f:
            f.write("".join(new_lines))

    # --- summary ---
    changes = []
    if n_energy > 0:
        changes.append(f"*energy* → energy ({n_energy} label(s))")
    if n_force > 0:
        changes.append(f"*force* → forces ({n_force} label(s))")
    if n_stress > 0:
        changes.append(f"stress → virial ({n_stress} frame(s))")
    if n_virial6 > 0:
        changes.append(f"virial 6-comp → 9-comp ({n_virial6} frame(s))")

    print_section("Standardize ExtXYZ")
    print_kv("Input", extxyz_path)
    print_kv("Output", output_path)
    if bak_path:
        print_kv("Backup", bak_path)
    if changes:
        print_success(" | ".join(changes))
    else:
        print_success("no changes needed")
    print()
