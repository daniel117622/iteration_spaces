from matrix_algos import strassen_multiply, create_sparse_matrix, iterative_strassen_multiply
import random
import numpy as np
import time

n = 3
start_time = time.time()
DIMENSIONS = 2**n
A = create_sparse_matrix(DIMENSIONS,DIMENSIONS , DIMENSIONS**2 , seed=random.randint(1,1000))
B = create_sparse_matrix(DIMENSIONS,DIMENSIONS , DIMENSIONS**2 , seed=random.randint(1,1000))

iter_space,strassen_result = iterative_strassen_multiply(A, B)
print(f"Matrix of size: {2**n} : {len(iter_space) - DIMENSIONS**3} : Time = {(time.time()-start_time):.4f}")
dot_result = np.dot(A, B)

count_loop1 = 0
count_loop2 = 0
for iteration in iter_space:
    if iteration[0] != 0:
        count_loop2 += 1
    else:
        count_loop1 += 1

print(count_loop1)
print(count_loop2)

assert np.allclose(strassen_result, dot_result, rtol=1e-4)

