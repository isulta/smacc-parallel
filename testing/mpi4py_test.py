# from mpi4py import MPI
# import numpy as np
# import pygio
# import h5py
# nproc = MPI.COMM_WORLD.Get_size()   # Size of communicator
# iproc = MPI.COMM_WORLD.Get_rank()   # Ranks in communicator
# inode = MPI.Get_processor_name()    # Node where this MPI process runs
# if iproc == 0:
# 	print("This code is a test for mpi4py.")
# for i in range(0,nproc):
# 	MPI.COMM_WORLD.Barrier()
# 	if iproc == i:
# 		print('Rank %d out of %d' % (iproc,nproc))
# MPI.Finalize()

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ranks = comm.Get_size()

import numpy as np
import pygio
import itk

# read locally
data = pygio.read_genericio("/cosmo/scratch/rangel/Farpoint/core_properties/42.coreproperties", ['x', 'y', 'z'])
# get local number of elements from the first element in dictionary
num_elems = len(next(iter(data.values())))
# reduce total number of elements
num_elems_total = comm.allreduce(num_elems)
if rank == 0:
    print(f"Reading file with {ranks} ranks")
    print(f"Total number of particles: {num_elems_total}")
    print("The data contains the following variables:")
    for k, d in data.items():
        print(f"\t{k:5s}, dtype={d.dtype}")

comm.Barrier()
for i in range(ranks):
    if i == rank:
        print(f"rank {rank} read {num_elems} elements")
    comm.Barrier()