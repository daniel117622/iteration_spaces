
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import collections
from collections import namedtuple

Task = namedtuple('Task', ['a', 'b', 'index'])

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
    
def iter_predic_opt(tp1, tp2): #Currently not tested for correctness
    dict1 = collections.defaultdict(list)
    dict2 = collections.defaultdict(list)

    for tp in tp1:
        dict1[tp[1]].append(tp)

    for tp in tp2:
        dict2[tp[0]].append(tp)

    result = []

    for key in dict1:
        if key in dict2:
            for val1 in dict1[key]:
                for val2 in dict2[key]:
                    result.append((val1[0], val1[1], val2[1]))

    return result


def strassen_multiply(a, b):
    # Check if matrices are 1x1
    if a.shape == (1, 1):
        return a * b

    # Split matrices into quadrants
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape
    split = max(max(a_rows, a_cols, b_rows, b_cols) // 2, 1)
    a11 = a[:split, :split]
    a12 = a[:split, split:]
    a21 = a[split:, :split]
    a22 = a[split:, split:]
    b11 = b[:split, :split]
    b12 = b[:split, split:]
    b21 = b[split:, :split]
    b22 = b[split:, split:]

    # Recursive matrix multiplications
    p1 = strassen_multiply(a11 + a22, b11 + b22)
    p2 = strassen_multiply(a21 + a22, b11)
    p3 = strassen_multiply(a11, b12 - b22)
    p4 = strassen_multiply(a22, b21 - b11)
    p5 = strassen_multiply(a11 + a12, b22)
    p6 = strassen_multiply(a21 - a11, b11 + b12)
    p7 = strassen_multiply(a12 - a22, b21 + b22)

    # Compute the resulting quadrants
    c11 = p1 + p4 - p5 + p7
    c12 = p3 + p5
    c21 = p2 + p4
    c22 = p1 - p2 + p3 + p6

    # Combine quadrants into the resulting matrix
    c_top = np.hstack((c11, c12))
    c_bottom = np.hstack((c21, c22))
    c = np.vstack((c_top, c_bottom))

    return c


def iterative_strassen_multiply(a, b):
    n = a.shape[0]
    depth = int(np.log2(n))
    stack_size = (7**(depth+1) - 1) // (7 - 1)
    c_stack = [None] * stack_size
    i_stack = [None] * stack_size
    task_stack = [None] * stack_size

    task_stack[0] = Task(a, b, 0)

    iter_space = []

    for i in range(stack_size):
        task = task_stack[i]
        iter_space.append([i,0])
        # Base case, matrix is 1x1
        if task.a.shape == (1, 1):
            c_stack[i] = task.a * task.b
            continue

        # Split matrices
        split = task.a.shape[0] // 2
        a11, a12, a21, a22 = task.a[:split, :split], task.a[:split, split:], task.a[split:, :split], task.a[split:, split:]
        b11, b12, b21, b22 = task.b[:split, :split], task.b[:split, split:], task.b[split:, :split], task.b[split:, split:]

        # Push tasks into the stack
        base_index = 7*i
        task_stack[base_index + 1] = Task(a11 + a22, b11 + b22, base_index + 1)
        task_stack[base_index + 2] = Task(a21 + a22, b11, base_index + 2)
        task_stack[base_index + 3] = Task(a11, b12 - b22, base_index + 3)
        task_stack[base_index + 4] = Task(a22, b21 - b11, base_index + 4)
        task_stack[base_index + 5] = Task(a11 + a12, b22, base_index + 5)
        task_stack[base_index + 6] = Task(a21 - a11, b11 + b12, base_index + 6)
        task_stack[base_index + 7] = Task(a12 - a22, b21 + b22, base_index + 7)

        # Record the indices of the sub-results for this task
        i_stack[i] = []
        for j in range(1, 8):
            i_stack[i].append(base_index + j)
    # Combine results
    for i in reversed(range(stack_size - (n-1)**3)): #Cut the meaningless c_stack values
        # Fetch sub-results
        p_values = []
        for j in i_stack[i]:
            p_values.append(c_stack[j])
          
        p1, p2, p3, p4, p5, p6, p7 = p_values

        # Compute the resulting quadrants
        c11 = p1 + p4 - p5 + p7
        c12 = p3 + p5
        c21 = p2 + p4
        c22 = p1 - p2 + p3 + p6

        # Combine quadrants into the resulting matrix
        c_top = np.hstack((c11, c12))
        c_bottom = np.hstack((c21, c22))
        c_stack[i] = np.vstack((c_top, c_bottom))

    return iter_space,c_stack[0]