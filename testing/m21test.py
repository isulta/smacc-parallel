from mpi4py import MPI
from itk import many_to_one_parallel
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

arr_root = {
    0: np.array([0,2,1,10]),
    1: np.array([11, 99, 99, 11, 99]),
    2: np.array([0,2,1,10, 99, 101]),
    3: np.array([57, 2, 2, 2, 2, 2, 0])
}

arr_local = {
    0: np.array([1,10,15,20]),
    1: np.array([11,13,0]),
    2: np.array([57,2]),
    3: np.array([99,101])
}

data_local = {
    0: np.array([60, 61, 62, 63]),
    1: np.array([64, 65, 66]),
    2: np.array([67, 68]),
    3: np.array([69, 70])
}

for root in range(4):
    if rank == root:
        ar = arr_root[root]
    else:
        ar = None
    Data = many_to_one_parallel(comm, rank, root, ar, arr_local[rank], np.int64, data_local[rank], np.int64)
    if rank == root:
        print(rank, Data)
        print()