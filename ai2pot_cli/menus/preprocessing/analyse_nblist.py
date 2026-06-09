"""Dataset analysis -- neighbour and pair-distance statistics for NEP setup."""

import numpy as np

from ase.io import read as ase_read
from ai2pot.nblist import Nblist


def run_analyse(extxyz_path: str, rcut: float, max_neigh_buf: int = 500):
    """Analyse an extxyz dataset and report neighbour / distance statistics.

    Args:
        extxyz_path: Path to the extxyz file.
        rcut: Cutoff radius for the neighbour list (Angstrom).
        max_neigh_buf: Internal buffer size for max neighbours during analysis.
    """
    frames = ase_read(extxyz_path, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    n_frames = len(frames)

    all_numneigh = []
    all_pair_dists = []
    all_nn_dists = []  # per-atom nearest-neighbour distance
    species_set = set()
    total_atoms = 0

    for atoms in frames:
        species_set.update(atoms.get_atomic_numbers())
        total_atoms += len(atoms)

        nbl = Nblist.from_ase(atoms, rcut=rcut, umax_num_neigh_atoms=max_neigh_buf)
        numneigh = nbl.numneigh  # (inum,)
        all_numneigh.append(numneigh)

        inum = nbl.inum
        # distances is (inum, max_neigh_buf) with zero-padding for unused slots
        # build 2D mask to extract only valid entries
        nn_mask = np.arange(max_neigh_buf)[None, :] < numneigh[:, None]
        all_pair_dists.append(nbl.distances[nn_mask])

        # ---- per-atom nearest-neighbour distance ----
        dists_2d = nbl.distances
        dists_masked = np.where(nn_mask, dists_2d, np.inf)
        nn_d = np.min(dists_masked, axis=1)
        nn_d = nn_d[np.isfinite(nn_d)]
        all_nn_dists.append(nn_d)

    all_numneigh = np.concatenate(all_numneigh)
    all_pair_dists = np.concatenate(all_pair_dists)
    all_nn_dists = np.concatenate(all_nn_dists)

    # ---- Compute statistics ----
    max_nn = int(np.max(all_numneigh))
    min_nn = int(np.min(all_numneigh))
    mean_nn = float(np.mean(all_numneigh))
    median_nn = float(np.median(all_numneigh))

    min_dist = float(np.min(all_pair_dists))
    min_nn_dist = float(np.min(all_nn_dists))
    mean_nn_dist = float(np.mean(all_nn_dists))
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    dist_pct = np.percentile(all_pair_dists, pcts)
    nn_pct = np.percentile(all_nn_dists, pcts)

    # ---- Print report ----
    sep = " " + "=" * 62

    print()
    print(sep)
    print("  Dataset Analysis Report")
    print(sep)
    print(f"  Input file:              {extxyz_path}")
    print(f"  Cutoff radius:           {rcut:.2f} A")
    print(f"  Number of frames:        {n_frames}")
    print(f"  Number of species:       {len(species_set)}  (Z = {sorted(species_set)})")
    print(f"  Total atoms:             {total_atoms}")
    print()
    print(f"  --- Neighbour Statistics  (rcut = {rcut:.1f} A) ---")
    print(f"  Max neighbours:          {max_nn}")
    print(f"  Min neighbours:          {min_nn}")
    print(f"  Mean neighbours:         {mean_nn:.1f}")
    print(f"  Median neighbours:       {median_nn:.1f}")
    print()
    print(f"  --- Pair-distance Distribution (all pairs within rcut) ---")
    print(f"  Minimum:                 {min_dist:.4f} A")
    for p, v in zip(pcts, dist_pct):
        print(f"  P{p:02d}:                     {v:.4f} A")
    print()
    print(f"  --- Nearest-neighbour Distance (per atom) ---")
    print(f"  Minimum:                 {min_nn_dist:.4f} A")
    print(f"  Mean:                    {mean_nn_dist:.4f} A")
    for p, v in zip(pcts, nn_pct):
        print(f"  P{p:02d}:                     {v:.4f} A")
    print()
    print(f"  --- Recommendations ---")
    suggested_umax = int(max_nn * 1.1) + 1
    print(f"  -> umax_num_neigh_atoms >= {suggested_umax}"
          f"  (max observed: {max_nn} + 10% margin)")

    if min_dist < 1.0:
        p10 = dist_pct[pcts.index(10)]
        print(f"  -> WARNING: Min pair distance {min_dist:.3f} A < 1.0 A.")
        print(f"     ZBL potential is recommended for short-range repulsion.")
        print(f"     Suggested zbl_rmax ~ {p10:.2f} A  (P10 of pair distances)")
        print(f"     Suggested zbl_rmin ~ {min_dist:.2f} A  (minimum pair distance)")
    else:
        print(f"  -> Min distance {min_dist:.3f} A >= 1.0 A. ZBL may not be needed.")

    if max_nn >= max_neigh_buf:
        print(f"  -> WARNING: Max neighbours ({max_nn}) hit the analysis buffer"
              f" ({max_neigh_buf}).")
        print(f"     Re-run with a larger buffer for accurate statistics.")
    print(sep)
    print()
