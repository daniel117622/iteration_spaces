from matrix_algos import *
from scipy.sparse import csr_matrix
from polytope_optimized_algorithms import gustav_mult_opt

A = create_sparse_matrix(5,5,3 ,  seed = 58)
B = create_sparse_matrix(5,5,3 ,  seed = 57)

iter_space , c = gustav_mult(A,B)

csrA = csr_matrix(A)
csrB = csr_matrix(B)

tp1 = np.column_stack((csrA.nonzero()[0], csrA.nonzero()[1]))
#print(*tp1, sep=" , ")

tp2 = np.column_stack((csrB.nonzero()[0], csrB.nonzero()[1]))
#print(*tp2, sep=" , ")

print(f"Real iter space: {iter_space}")
#tp1 and tp2 are the pairs of nnz elements of each of the matrixes
def iter_predic(tp1,tp2): 
    predict = []
    for a_tuple in tp1:
        marriage_seeker = a_tuple[1]
        for b_tuple in tp2:
            if marriage_seeker == b_tuple[0]:
                predict.append([*a_tuple , b_tuple[1]])
    return predict

iter_space = iter_predic(tp1,tp2)
print(f"Predicted iter space: {iter_space}")
#print(csrA)
print(csrA.toarray())
print("\n" * 3)
#print(csrB)
print(csrB.toarray())
print("\n" * 3)
print("============== RESULTS ==============")

c = classic_multiply(A,B)
print(np.array(c))
print("\n" * 2)
g = gustav_mult_opt(iter_space , csrA.data,  csrB.data, (5,5) )
print(np.array(g))

