"""Convert DeepMD (DeePMD-kit) dataset to extxyz format.

Reads DeepMD standard format (type.raw, set.*/box.npy, coord.npy, energy.npy,
force.npy, virial.npy) and writes a unified extxyz file.
"""

import os
import sys
import glob
import numpy as np

from ai2pot_cli.menu import print_section, print_warning, print_error, print_kv, print_sep


def _cond_load(fname):
    if os.path.isfile(fname):
        return np.load(fname)
    return None


def _load_type(folder):
    data = {}
    data['atom_types'] = np.loadtxt(os.path.join(folder, 'type.raw'), ndmin=1).astype(int)
    ntypes = np.max(data['atom_types']) + 1
    data['atom_numbs'] = []
    for ii in range(ntypes):
        data['atom_numbs'].append(np.count_nonzero(data['atom_types'] == ii))
    data['atom_names'] = []
    if os.path.isfile(os.path.join(folder, 'type_map.raw')):
        with open(os.path.join(folder, 'type_map.raw')) as fp:
            my_type_map = fp.read().split()
    else:
        my_type_map = [f'Type_{ii}' for ii in range(ntypes)]
    for ii in range(len(data['atom_numbs'])):
        data['atom_names'].append(my_type_map[ii])
    data['my_type_map'] = my_type_map
    return data


def _load_set(folder):
    cells = np.load(os.path.join(folder, 'box.npy'))
    coords = np.load(os.path.join(folder, 'coord.npy'))
    eners = _cond_load(os.path.join(folder, 'energy.npy'))
    forces = _cond_load(os.path.join(folder, 'force.npy'))
    virs = _cond_load(os.path.join(folder, 'virial.npy'))
    set_types = _cond_load(os.path.join(folder, 'real_atom_types.npy'))
    return cells, coords, eners, forces, virs, set_types


def _to_system_data(folder):
    data = _load_type(folder)
    data['docname'] = folder
    sets = sorted(glob.glob(os.path.join(folder, 'set.*')))
    all_cells, all_coords, all_eners, all_forces, all_virs = [], [], [], [], []
    real_set_types = []
    for s in sets:
        cells, coords, eners, forces, virs, set_types = _load_set(s)
        nframes = np.reshape(cells, [-1, 3, 3]).shape[0]
        all_cells.append(np.reshape(cells, [nframes, 3, 3]))
        all_coords.append(np.reshape(coords, [nframes, -1, 3]))
        if eners is not None and eners.size > 0:
            all_eners.append(np.reshape(eners, [nframes]))
        if forces is not None and forces.size > 0:
            all_forces.append(np.reshape(forces, [nframes, -1, 3]))
        if virs is not None and virs.size > 0:
            all_virs.append(np.reshape(virs, [nframes, 9]))
        if set_types is not None and set_types.size > 0:
            real_set_types.append(np.reshape(set_types, [nframes, -1]))

    data['frames'] = sum(np.reshape(c, [-1, 3, 3]).shape[0] for c in all_cells)
    data['cells'] = np.concatenate(all_cells, axis=0)
    data['coords'] = np.concatenate(all_coords, axis=0)
    if all_eners:
        data['energies'] = np.concatenate(all_eners, axis=0)
    if all_forces:
        data['forces'] = np.concatenate(all_forces, axis=0)
    if all_virs:
        data['virials'] = np.concatenate(all_virs, axis=0)
    if real_set_types:
        data['set_types'] = np.concatenate(real_set_types, axis=0)
    return data


def _read_multi_deepmd(folder):
    data_multi = {}
    list_dir = []
    for dirpath, filedir, filename in os.walk(folder):
        if 'type.raw' in filename:
            list_dir.append(dirpath)

    for i, fi in enumerate(list_dir):
        idata = _to_system_data(fi)
        if 'virials' in idata and len(idata['virials']) == idata['frames']:
            idata['has_virial'] = np.ones(idata['frames'], dtype=bool)
        else:
            idata['has_virial'] = np.zeros(idata['frames'], dtype=bool)
        if 'set_types' in idata and len(idata['set_types']) == idata['frames']:
            idata['has_set_types'] = np.ones(idata['frames'], dtype=bool)
        else:
            idata['has_set_types'] = np.zeros(idata['frames'], dtype=bool)
        data_multi[i] = idata

    nframes = int(np.sum([data_multi[i]['frames'] for i in data_multi]))
    data = {'nframe': nframes}
    data['atom_numbs'] = np.zeros(nframes)
    data['has_virial'] = np.zeros(nframes)
    data['virials'] = np.zeros((nframes, 9))
    data['has_set_types'] = np.zeros(nframes)
    data['energies'] = np.zeros(nframes)
    data['cells'] = np.zeros((nframes, 9))
    data['type_maps'] = {}
    data['atom_names'] = {}
    data['atom_types'] = {}
    data['coords'] = {}
    data['forces'] = {}
    data['set_types'] = {}

    ifr = -1
    for i in data_multi:
        atom_names = [data_multi[i]['atom_names'][j] for j in data_multi[i]['atom_types']]
        for j in range(data_multi[i]['frames']):
            ifr += 1
            data['atom_numbs'][ifr] = len(data_multi[i]['atom_types'])
            data['has_virial'][ifr] = data_multi[i]['has_virial'][j]
            data['has_set_types'][ifr] = data_multi[i]['has_set_types'][j]
            data['energies'][ifr] = data_multi[i]['energies'][j]
            if data['has_virial'][ifr]:
                data['virials'][ifr] = data_multi[i]['virials'][j]
            if data['has_set_types'][ifr]:
                data['set_types'][ifr] = data_multi[i]['set_types'][j]
            data['cells'][ifr] = np.reshape(data_multi[i]['cells'][j], 9)
            data['type_maps'][ifr] = data_multi[i]['my_type_map']
            data['atom_names'][ifr] = atom_names
            data['atom_types'][ifr] = data_multi[i]['atom_types']
            data['coords'][ifr] = data_multi[i]['coords'][j]
            data['forces'][ifr] = data_multi[i]['forces'][j]

    return data


def _dump_xyz(data, out_path):
    with open(out_path, 'w') as fo:
        for i in range(data['nframe']):
            fo.write(f"{int(data['atom_numbs'][i])}\n")
            parts = [
                f'Lattice="{" ".join(map(str, data["cells"][i]))}"',
                f'Properties=species:S:1:pos:R:3:forces:R:3',
                f'energy={data["energies"][i]}',
            ]
            if data['has_virial'][i]:
                parts.append(f'virial="{" ".join(map(str, data["virials"][i]))}"')
            parts.append('pbc="T T T"')
            parts.append('config_type=dp2xyz')
            fo.write(' '.join(parts) + '\n')

            if data['has_set_types'][i]:
                for j in range(int(data['atom_numbs'][i])):
                    elem = data['type_maps'][i][data['set_types'][i][j]]
                    coord = " ".join(map(str, data['coords'][i][j]))
                    force = " ".join(map(str, data['forces'][i][j]))
                    fo.write(f"{elem} {coord} {force}\n")
            else:
                for j in range(int(data['atom_numbs'][i])):
                    elem = data['atom_names'][i][j]
                    coord = " ".join(map(str, data['coords'][i][j]))
                    force = " ".join(map(str, data['forces'][i][j]))
                    fo.write(f"{elem} {coord} {force}\n")


def run_dp2xyz():
    """Interactive DeepMD → extxyz converter."""
    print_section("DeepMD → extxyz")

    input_dir = input(" DeepMD dataset directory: ").strip()
    if not input_dir or not os.path.isdir(input_dir):
        print_error(f"Directory not found: {input_dir}")
        return

    out_name = input(" Output file name [default: train.xyz]: ").strip() or "train.xyz"
    data = _read_multi_deepmd(input_dir)
    _dump_xyz(data, out_name)

    print_section("DeepMD → extxyz Conversion Completed")
    print_kv("Output File", os.path.abspath(out_name))
    print_kv("Configurations", str(data['nframe']))
    print_sep()
    print()
    sys.exit(0)
