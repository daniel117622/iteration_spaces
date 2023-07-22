from matrix_algos import *
from scipy.sparse import csr_matrix

DIMENSIONS = 0

A = create_sparse_matrix(DIMENSIONS,DIMENSIONS,DIMENSIONS,  seed = 42)
B = create_sparse_matrix(DIMENSIONS,DIMENSIONS,DIMENSIONS,  seed = 55)

csrA = csr_matrix(A)
csrB = csr_matrix(B)

