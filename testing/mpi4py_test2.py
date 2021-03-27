from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

dataint = rank
data = comm.allgather(dataint)
print(rank, data)

f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=comm)

dset = f.create_dataset('test', (40,), dtype='i')
dset[rank*10:(rank+1)*10] = rank

f.close()