import time
from line_profiler import LineProfiler
import numpy as np

def csr_to_ij(csr_indptr, csr_indices):
    n_rows = len(csr_indptr) - 1
    ij_list = []
    for i in range(n_rows):
        start_idx = csr_indptr[i]
        end_idx = csr_indptr[i+1]
        for idx in range(start_idx, end_idx):
            j = csr_indices[idx]
            ij_list.append((i, j))
    return ij_list


def gustav_mult_opt(iter_space, data1, data2):
    start_time_init = time.time()
    max_i = max(iter_space, key=lambda t: t[0])[0]
    max_k = max(iter_space, key=lambda t: t[2])[2]
    outshape = (max_i + 1, max_k + 1)

    c = np.zeros((outshape[0], outshape[1]))
    ptr1 = 0
    ptr2 = 0
    d1_indptr = data1.indptr
    d1_indices = data1.indices
    d1_sorted = csr_to_ij(d1_indptr, d1_indices)
    d1_dict = {coord: idx for idx, coord in enumerate(d1_sorted)}  # Convert to dictionary

    d2_indptr = data2.indptr
    d2_indices = data2.indices
    d2_sorted = csr_to_ij(d2_indptr, d2_indices)
    d2_dict = {coord: idx for idx, coord in enumerate(d2_sorted)}  # Convert to dictionary

    endtime = time.time()
    print(f"Overhead completed. Time {endtime - start_time_init}")
    start_time = time.time()
    for i, j, k in iter_space:
        # Access dictionary for faster index lookup
        ptr1 = d1_dict[(i, j)]
        ptr2 = d2_dict[(j, k)]

        c[i][k] += data1.data[ptr1] * data2.data[ptr2]

    endtime = time.time()
    print(f"Total time of looping: {endtime-start_time}")
    print(f"Total time: {endtime-start_time_init}")
    return c
