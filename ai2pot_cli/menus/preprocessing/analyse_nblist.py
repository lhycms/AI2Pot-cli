"""Dataset analysis -- neighbour and pair-distance statistics for NEP setup."""

from typing import List

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.data import chemical_symbols
from tqdm import tqdm

from ai2pot.core.nblist import Nblist


def analyse_dataset(extxyz_path: str,
                        rcut: float,
                        umax_num_neigh_atoms: int = 1200):
    """Analyse an extxyz dataset and report neighbour / distance statistics.

    Args:
        extxyz_path: Path to the extxyz file.
        rcut: Cutoff radius for the neighbour list (Angstrom).
        umax_num_neigh_atoms: Internal buffer size for max neighbours during analysis.
    """
    frames: List[Atoms] = ase_read(extxyz_path, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    n_frames: int = len(frames)
    species_set = set()
    total_atoms = 0
    all_numneigh: List[np.ndarray] = []
    all_pair_dists: List[np.ndarray] = []
    all_nn_dists: List[np.ndarray] = []  # per-atom nearest-neighbour distance
    energies: List[float] = []
    forces: List[np.ndarray] = []
    virials: List[np.ndarray] = []

    print(f"\n Analysing {n_frames} frame(s) with rcut = {rcut:.2f} A ...\n")
    for atoms in tqdm(frames, desc=" Progress", ncols=80, bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
        species_set.update(atoms.get_atomic_numbers())
        total_atoms += len(atoms)

        nblist: Nblist = Nblist.from_ase(atoms, rcut=rcut, umax_num_neigh_atoms=umax_num_neigh_atoms)
        numneigh: np.ndarray = nblist.numneigh  # (inum,)
        all_numneigh.append(numneigh)

        # distances is (inum, umax_num_neigh_atoms) with zero-padding for unused slots
        # build 2D mask to extract only valid entries
        nn_mask = np.arange(umax_num_neigh_atoms)[None, :] < numneigh[:, None]
        all_pair_dists.append(nblist.distances[nn_mask])

        # ---- per-atom nearest-neighbour distance ----
        dists_2d: np.ndarray = nblist.distances
        dists_masked: np.ndarray = np.where(nn_mask, dists_2d, np.inf)
        nn_d: np.ndarray = np.min(dists_masked, axis=1) # 沿着列方向求最小值
        nn_d = nn_d[np.isfinite(nn_d)]  # 删除没邻居的中心原子
        all_nn_dists.append(nn_d)

        # energy, force, virial
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
        try:
            # virial: np.ndarray = -stress * atoms.get_volume()
            virial: np.ndarray = atoms.info["virial"]
            virials.append(virial)
        except:
            pass


    all_numneigh: np.ndarray = np.concatenate(all_numneigh)
    all_pair_dists = np.concatenate(all_pair_dists)
    all_nn_dists = np.concatenate(all_nn_dists)
    forces = np.concatenate(forces).reshape(-1)
    try:
        virials = np.concatenate(virials).reshape(-1)
    except:
        pass
    print(forces)
    print(virials)

    # ---- Compute statistics ----
    max_nn = int(np.max(all_numneigh))
    min_nn = int(np.min(all_numneigh))
    mean_nn = float(np.mean(all_numneigh))
    median_nn = float(np.median(all_numneigh))

    min_pair_dist = float(np.min(all_pair_dists))
    min_nn_dist = float(np.min(all_nn_dists))
    mean_nn_dist = float(np.mean(all_nn_dists))
    p10_pair_dist = float(np.percentile(all_pair_dists, 10))

    natoms_per_frame = np.asarray([len(atoms) for atoms in frames], dtype=float)

    energies = np.asarray(energies, dtype=float)
    energies_per_atom = energies / natoms_per_frame

    energy_mean = float(np.mean(energies_per_atom))
    energy_std = float(np.std(energies_per_atom))
    energy_min = float(np.min(energies_per_atom))
    energy_max = float(np.max(energies_per_atom))

    force_mean = float(np.mean(forces))
    force_std = float(np.std(forces))
    force_max = float(np.max(forces))
    force_min = float(np.min(forces))

    has_virial = len(virials) > 0

    # ---- Print report ----
    sep = " " + "=" * 62

    print()
    print(sep)
    print("  Dataset Analysis")
    print(sep)
    print(f"  File              : {extxyz_path}")
    print(f"  Frames / atoms    : {n_frames} / {total_atoms}")
    species_z = sorted(species_set)
    species_symbols = [chemical_symbols[z] for z in species_z]
    print(
        f"  Species           : {len(species_set)} "
        f"({', '.join(species_symbols)})"
    )
    print(f"  Cutoff            : {rcut:.2f} A")
    print()
    print(f"  Neighbours                  : min {min_nn} | mean {mean_nn:.1f} | "
        f"median {median_nn:.1f} | max {max_nn}")
    print(f"  Interatomic Distance        : min {min_pair_dist:.4f} A | "
        f"p10 {p10_pair_dist:.4f} A")
    print(f"  Nearest-neighbour distance  : min {min_nn_dist:.4f} A | "
        f"mean {mean_nn_dist:.4f} A")
    print()

    print("  Label Statistics")
    print(
    f"  {'Energy per atom':<18}: "
    f"mean {energy_mean:>10.6f} eV/atom | "
    f"std {energy_std:>8.6f} | "
    f"max {energy_max:>10.6f} | "
    f"min {energy_min:>10.6f}"
    )

    print(
        f"  {'Force magnitude':<18}: "
        f"mean {force_mean:>10.4f} eV/A    | "
        f"std {force_std:>8.4f} | "
        f"max {force_max:>10.4f} | "
        f"min {force_min:>10.4f}"
    )

    print(
        f"  {'Virial':<18}: "
        f"{'found' if has_virial else 'not found'}"
    )
    print()


    print("  Recommendation")
    suggested_umax = int(max_nn * 1.3) + 1
    print(f"  - umax_num_neigh_atoms >= {suggested_umax} "
        f"(max {max_nn} + 30%)")

    if min_pair_dist < 2.0:
        print("  - WARNING: short pair distance detected; consider enabling ZBL.")
    else:
        print("  - ZBL is probably not necessary based on the minimum distance.")

    if max_nn >= umax_num_neigh_atoms:
        print(f"  - WARNING: neighbour buffer was reached "
            f"({max_nn}/{umax_num_neigh_atoms}); rerun with a larger buffer.")

    print(sep)
    print()
