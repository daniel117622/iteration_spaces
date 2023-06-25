import numpy as np
def gustav_mult_opt(iter_space , data1 , data2, outshape):
    c = [[0 for _ in range(outshape[1])] for _ in range(outshape[0])]
    curr_iter = 0
    ptr1 = 0
    ptr2 = 0
    d1_unsorted = [(i, j,'A') for i, j, _ in iter_space]
    d2_unsorted = [(j, k,'B') for _, j, k in iter_space]

    d_sorted = sorted((d1_unsorted + d2_unsorted), key=lambda t: t[0] * outshape[0] + t[1])

    for i,j,k in iter_space:
        ptr1 = d_sorted.index((i, j, 'A')) - sum(t[2] == 'B' for t in d_sorted[:d_sorted.index((i, j, 'A'))])
        ptr2 = d_sorted.index((j, k, 'B')) - sum(t[2] == 'A' for t in d_sorted[:d_sorted.index((j, k, 'B'))])

        c[i][k] += data1[ptr1] * data2[ptr2]
        curr_iter += 1


    return c

