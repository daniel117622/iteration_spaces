from matrix_algos import *
from scipy.sparse import csr_matrix
from polytope_optimized_algorithms import gustav_mult_opt
import time
from numba import cuda,float64
import tensorflow as tf

def complete_test():
    DIMENSIONS = 1024

    A = create_sparse_matrix(DIMENSIONS,DIMENSIONS,DIMENSIONS,  seed = 42)
    B = create_sparse_matrix(DIMENSIONS,DIMENSIONS,DIMENSIONS,  seed = 55)

    csrA = csr_matrix(A)
    csrB = csr_matrix(B)


    tp1 = np.column_stack((csrA.nonzero()[0], csrA.nonzero()[1])).tolist()
    tp2 = np.column_stack((csrB.nonzero()[0], csrB.nonzero()[1])).tolist()
    iter_space = iter_predic(tp1,tp2)

    gustav_mult_opt(iter_space, csrA, csrB)

    result = np.matmul(A, B)

    return csrA,csrB,iter_space


if __name__ == "__main__":
    from line_profiler import LineProfiler
    ################
    #COMPLETE TESTS#
    ################ 
    profiler = LineProfiler()
    profiler.add_function(complete_test)
    profiler.enable_by_count()

    csrA,csrB,iter_space = complete_test()

    profiler.disable_by_count()
    with open("complete.txt","w") as fd:
        profiler.print_stats(stream=fd, output_unit=1e-06)
    ################
    #SPECIFIC TESTS#
    ################ 
    profiler = LineProfiler()
    profiler.add_function(gustav_mult_opt)
    profiler.enable_by_count()

    gustav_mult_opt(iter_space,csrA,csrB)

    profiler.disable_by_count()
    with open("gustav_mult.txt","w") as fd:
        profiler.print_stats(stream=fd, output_unit=1e-06)
