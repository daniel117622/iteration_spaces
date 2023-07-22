from matrix_algos import *
from polytope_optimized_algorithms import gustav_mult_opt
import random
from time import time
from scipy.sparse import csr_matrix
from statistics import mean
DIMENSIONS = 128

time_gustav = []


def trial(sparsity): # Value between 0 and 1
    avg_time_opt = []
    number_of_items = int(sparsity * 16384)
    i = 0
    while i < 3:
        A = create_sparse_matrix(DIMENSIONS,DIMENSIONS , number_of_items, seed=random.randint(1,1000))
        B = create_sparse_matrix(DIMENSIONS,DIMENSIONS , number_of_items , seed=random.randint(1,1000))
        start_time = time()

        csrA = csr_matrix(A)
        csrB = csr_matrix(B)
        tp1 = np.column_stack((csrA.nonzero()[0], csrA.nonzero()[1])).tolist()
        tp2 = np.column_stack((csrB.nonzero()[0], csrB.nonzero()[1])).tolist()

        iter_space = iter_predic_opt(tp1,tp2)
        if len(iter_space) == 0:
            continue

        gustav_mult_opt(iter_space, csrA, csrB)        

        end_time = time() - start_time
        avg_time_opt.append(end_time)
        i += 1
    i = 0    
    avg_time = []
    while i < 3:
        A = create_sparse_matrix(DIMENSIONS,DIMENSIONS , number_of_items, seed=random.randint(1,1000))
        B = create_sparse_matrix(DIMENSIONS,DIMENSIONS , number_of_items , seed=random.randint(1,1000))
        start_time = time()
        gustav_mult(A,B)
        end_time = time() - start_time
        avg_time.append(end_time)
        i += 1

    return {"optimized" : mean(avg_time_opt) , "normal" : mean(avg_time)}
    
optimized = []
normal = []
for sparsity in np.linspace(0.01,0.95,10):
    print(f"{sparsity}")
    T = trial(sparsity)
    normal.append(T["normal"])
    optimized.append(T["optimized"])

print(normal)
print(optimized)