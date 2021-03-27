from mpi4py import MPI
from itk import intersect1d_parallel
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

arr_root = {
    0: np.array([0,2,1,10]),
    1: np.array([11, 99, 80]),
    2: np.array([81, 83]),
    3: np.array([57, 2])
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
    # root = 0
    if rank == root:
        ar = arr_root[root]
    else:
        ar = None
    recvbuf_idx1, recvbuf_data = intersect1d_parallel(comm, rank, root, ar, arr_local[rank], np.int64, data_local[rank], np.int64)
    if rank == root:
        print(rank, recvbuf_idx1)
        print(rank, ar[recvbuf_idx1])
        print(rank, recvbuf_data)

        x = np.zeros_like(ar) - 101
        x[recvbuf_idx1] = recvbuf_data
        print(rank, x)
        print()