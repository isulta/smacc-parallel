# drop_missing from https://stackoverflow.com/a/46573391
from numba import njit
import numpy as np

@njit
def drop_missing(intersect,sample):
    i=j=k=0
    new_intersect=np.empty_like(intersect)
    while i< intersect.size and j < sample.size:
            if intersect[i]==sample[j]: # the 99% case
                new_intersect[k]=intersect[i]
                k+=1
                i+=1
                j+=1
            elif intersect[i]<sample[j]:
                i+=1
            else : 
                j+=1
    return new_intersect[:k]  

@njit
def intersect1d_numba(intersect,sample):
    i = j = k = 0
    idx1 = np.empty(len(intersect), dtype=np.int64)
    idx2 = np.empty(len(intersect), dtype=np.int64)
    while i < intersect.size and j < sample.size:
            if intersect[i] == sample[j]:
                idx1[k] = i
                idx2[k] = j
                k += 1
                i += 1
                j += 1
            elif intersect[i] < sample[j]:
                i += 1
            else: 
                j += 1
    return idx1[:k], idx2[:k]