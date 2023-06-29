from matrix_algos import *
from scipy.sparse import csr_matrix
from polytope_optimized_algorithms import gustav_mult_opt
import time

A = create_sparse_matrix(1024,1024,6548 ,  seed = 59)
B = create_sparse_matrix(1024,1024,5000 ,  seed = 55)


csrA = csr_matrix(A)
csrB = csr_matrix(B)

def iter_predic(tp1,tp2): 
    predict = []
    for a_tuple in tp1:
        marriage_seeker = a_tuple[1]
        for b_tuple in tp2:
            if marriage_seeker == b_tuple[0]:
                predict.append([*a_tuple , b_tuple[1]])
    return predict

#print(csrA)
print("\n" * 3)
print("============== RESULTS ==============")

tp1 = np.column_stack((csrA.nonzero()[0], csrA.nonzero()[1]))
tp2 = np.column_stack((csrB.nonzero()[0], csrB.nonzero()[1]))
iter_space = iter_predic(tp1,tp2)
# Run gustav_mult_opt 100,000 times
g = np.array(gustav_mult_opt(iter_space, csrA, csrB))

start_time = time.time()
normal_mult = np.array(gustav_mult(A, B))
end_time = time.time()
print(f"Gustav time: {end_time-start_time}")

start_time = time.time()
result = np.matmul(A, B)
end_time = time.time()
print(f"np.matmul: {end_time-start_time}")

