from matrix_algos import *
from scipy.sparse import csr_matrix
from polytope_optimized_algorithms import gustav_mult_opt
import time

DIMENSIONS = 4096

A = create_sparse_matrix(DIMENSIONS,DIMENSIONS,2500 ,  seed = 59)
B = create_sparse_matrix(DIMENSIONS,DIMENSIONS,5620 ,  seed = 55)


csrA = csr_matrix(A)
csrB = csr_matrix(B)

def iter_predic(tp1,tp2): 
    predict = []
    for a_tuple in tp1:
        for b_tuple in tp2:
            if a_tuple[1] == b_tuple[0]:
                predict.append([*a_tuple , b_tuple[1]]) 
    return predict

print("\n" * 1)

start_time = time.time()
tp1 = np.column_stack((csrA.nonzero()[0], csrA.nonzero()[1]))
tp2 = np.column_stack((csrB.nonzero()[0], csrB.nonzero()[1]))
iter_space = iter_predic(tp1,tp2)
end_time = time.time()
print(f"Computing of iteration space: {end_time - start_time}")
# # Run gustav_mult_opt 100,000 times
g = np.array(gustav_mult_opt(iter_space, csrA, csrB))

start_time = time.time()
normal_mult = np.array(gustav_mult(A, B))
end_time = time.time()
print(f"\nGustav time: {end_time-start_time}")

start_time = time.time()
result = np.matmul(A, B)
end_time = time.time()
print(f"np.matmul: {end_time-start_time}")

#from line_profiler import LineProfiler
#profiler = LineProfiler()
#profiler.add_function(gustav_mult_opt)
#profiler.enable_by_count()
#gustav_mult_opt(iter_space, csrA, csrB)
#profiler.disable_by_count()
#profiler.print_stats()
#
