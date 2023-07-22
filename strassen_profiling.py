from matrix_algos import strassen_multiply, create_sparse_matrix, iterative_strassen_multiply
import random
import numpy as np
import time
import line_profiler

def test(A,B):
    X1 = strassen_multiply(A,B)
    X2 = iterative_strassen_multiply(A,B)
    return X1, X2

n = 6
start_time = time.time()
DIMENSIONS = 2**n
A = create_sparse_matrix(DIMENSIONS,DIMENSIONS , DIMENSIONS**2 , seed=random.randint(1,1000))
B = create_sparse_matrix(DIMENSIONS,DIMENSIONS , DIMENSIONS**2 , seed=random.randint(1,1000))
PROFILING = True

if PROFILING:
    profiler = line_profiler.LineProfiler()
    profiler.add_function(test)
    profiler.enable()
    X1,X2 = test(A,B)
    with open('strassen_compare.txt', 'w') as fd:
        profiler.print_stats(stream=fd, output_unit=1e-06)

    profiler = line_profiler.LineProfiler()
    profiler.add_function(iterative_strassen_multiply)
    profiler.enable()
    iterative_strassen_multiply(A,B)
    profiler.print_stats()
    assert np.allclose(X1,X2,rtol = 1e-3)
else:
    X1 = iterative_strassen_multiply(A,B)
    X2 = strassen_multiply(A,B)
    assert np.allclose(X1,X2,rtol = 1e-3)

