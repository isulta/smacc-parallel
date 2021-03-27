from mpi4py import MPI
from itk import h5_write_dict_parallel, h5_write_dict
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

r0 = {
    'a': np.arange(0,10),
    'b': np.linspace(0,100,10),
}
r1 = {
    'a': np.arange(10,15),
    'b': np.linspace(0,255,5),
}
r2 = {
    'a': np.arange(15,25),
    'b': np.linspace(69,420,10),
}
r3 = {
    'a': np.arange(25,30),
    'b': np.linspace(0,10,5),
}
r = {
    0:r0,
    1:r1,
    2:r2,
    3:r3
}

def printr(s, root=0):
    if rank == root:
        print(f'rank {root}: {s}', flush=True)

if __name__ == '__main__':
    h5_write_dict_parallel(comm, rank, r[rank], ['a', 'b'], {'a':np.int64, 'b':np.float32}, 'h5test.hdf5')
    h5_write_dict( f'{499}.{rank}.ccprev.hdf5', r[rank], 'ccprev' )
    printr('Done saving!')