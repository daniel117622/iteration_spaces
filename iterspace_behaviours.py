from matrix_algos import *
from scipy.sparse import csr_matrix
from polytope_optimized_algorithms import gustav_mult_opt
from mpl_toolkits.mplot3d import Axes3D

A = create_sparse_matrix( 8 , 8 , 0 ,  seed = 15)
B = create_sparse_matrix( 8 , 8 , 0 ,  seed = 15)


csrA = csr_matrix(A)
csrB = csr_matrix(B)

def iter_predic(tp1,tp2): 
    predict = []
    iter_space = []
    i = 0
    for a_tuple in tp1:        
        marriage_seeker = a_tuple[1]
        j = 0
        for b_tuple in tp2:
            if marriage_seeker == b_tuple[0]:
                predict.append([*a_tuple , b_tuple[1]])
                iter_space.append((i,j))
                j = j + 1
        i = i + 1
    return predict , iter_space


tp1 = np.column_stack((csrA.nonzero()[0], csrA.nonzero()[1]))
tp2 = np.column_stack((csrB.nonzero()[0], csrB.nonzero()[1]))
iter_space, auto_iter = iter_predic(tp1,tp2) # predict the iterspace of this thing

x_values = [t[0] for t in auto_iter]
y_values = [t[1] for t in auto_iter]

print(auto_iter)
plt.scatter(x_values, y_values)

plt.show()