
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def compute_iteration_space(nnz_indexes_matrix_A, nnz_indexes_matrix_B):
    iteration_space = set()

    for i in range(len(nnz_indexes_matrix_A)):
        for j in range(len(nnz_indexes_matrix_B)):
            iteration_space.add((nnz_indexes_matrix_A[i][0], nnz_indexes_matrix_B[j][1]))

    return iteration_space

def create_sparse_matrix(x, y, nnz, seed = None):
    if nnz > x * y:
        raise ValueError("Number of nonzero elements cannot exceed matrix dimensions.")

    # Set the seed for reproducibility
    np.random.seed(seed)

    # Generate an empty matrix filled with zeros
    sparse_matrix = np.zeros((x, y))

    # Randomly select indices for nonzero elements
    indices = np.random.choice(x * y, nnz, replace=False)

    # Set the nonzero elements to random values
    sparse_matrix.flat[indices] = np.random.rand(nnz)

    return sparse_matrix

def classic_multiply(matrix1, matrix2):
    result = []

    # Get the dimensions of the matrices
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])

    # Check if the matrices can be multiplied
    if cols1 != rows2:
        print("Error: Incompatible matrix dimensions.")
        return result

    # Perform matrix multiplication
    result = [[0] * cols2 for _ in range(rows1)]

    variable_values = {0: [] , 1 : [] , 2 : []} 

    # There are no dependencies in this algorithm. With n**3 cores we can perform it.
    for i in range(rows1):
        for j in range(cols2):
            dot_product = 0
            for k in range(cols1):
                variable_values[2].append( (i,j,k) )
                dot_product += matrix1[i][k] * matrix2[k][j]
            result[i][j] = dot_product

    return result 

def gustav_mult(matrix1, matrix2):
    p = len(matrix1)
    q = len(matrix1[0])
    r = len(matrix2[0])
    
    c = [[0] * r for _ in range(p)]

    for i in range(p):
        for j in range(q):
            if matrix1[i][j] != 0:
                for k in range(r):
                    if matrix2[j][k] != 0:
                        c[i][k] += matrix1[i][j] * matrix2[j][k]


    return c

def iter_predic(tp1, tp2):
    tp1 = np.array(tp1)
    tp2 = np.array(tp2)

    mask = np.equal.outer(tp1[:, 1], tp2[:, 0])
    indices = np.nonzero(mask)
    predict = np.column_stack((tp1[indices[0]], tp2[indices[1]][:, 1]))

    return predict.tolist()
    