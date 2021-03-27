from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ranks = comm.Get_size()

import numpy as np
import pygio
import h5py
import sys
import os

vars_cc_all = [
    'fof_halo_tag',
    'infall_fof_halo_mass',
    'core_tag',
    'tree_node_index',
    'infall_tree_node_mass',
    'central',
    'host_core'
]
vars_cc_min = [
    'core_tag',
    'tree_node_index',
    'infall_tree_node_mass',
    'central',
    'host_core'
]

dtypes_cc_all = {
    'fof_halo_tag': np.int64,
    'infall_fof_halo_mass': np.float32,
    'core_tag': np.int64,
    'tree_node_index': np.int64,
    'infall_tree_node_mass': np.float32,
    'central': np.int32,
    'host_core': np.int64
}

input_dir = '/cosmo/scratch/rangel/Farpoint/core_properties'
outout_dir = '/cosmo/scratch/isultan/Farpoint'
flist = os.listdir(input_dir)
steps = sorted([int(fn.split('.')[0]) for fn in flist if not ('#' in fn)])

def fname(step, mode):
    if mode=='i':
        return f'{input_dir}/{step}.coreproperties'
    elif mode=='o':
        return f'{outout_dir}/{step}.corepropertiesreduced.hdf5'

def vars_cc(step):
    if step==499 or step==247:
        return vars_cc_all
    else:
        return vars_cc_min

# read locally
# step = 42

for step in steps:
    if rank == 0:
        print(f'Starting step {step}.')
    fin = fname(step, 'i')
    fout = fname(step, 'o')

    data = pygio.read_genericio(fin, vars_cc(step))
    # get local number of elements from the first element in dictionary
    num_elems = len(next(iter(data.values())))
    # reduce total number of elements
    num_elems_total = comm.allreduce(num_elems)

    num_elems_all = comm.allgather(num_elems)


    if rank == 0:
        print(f"Reading file with {ranks} ranks")
        print(f"Total number of particles: {num_elems_total}")
        print("The data contains the following variables:")
        for k, d in data.items():
            print(f"\t{k:5s}, dtype={d.dtype}")
    sys.stdout.flush()
    comm.Barrier()
    for i in range(ranks):
        if i == rank:
            print(f"rank {rank} read {num_elems} elements")
        comm.Barrier()

    f = h5py.File(fout, 'w', driver='mpio', comm=comm)

    for k in vars_cc(step):
        dset = f.create_dataset(k, (num_elems_total,), dtype=dtypes_cc_all[k])
        dset[ sum(num_elems_all[:rank]) : sum(num_elems_all[:rank+1]) ] = data[k]
        comm.Barrier()

    f.close()
    comm.Barrier()