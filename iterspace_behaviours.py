from matrix_algos import *
from scipy.sparse import csr_matrix
from polytope_optimized_algorithms import gustav_mult_opt
from mpl_toolkits.mplot3d import Axes3D
from typing import List

A = create_sparse_matrix( 8 , 8 , 12 ,  seed = 1)
B = create_sparse_matrix( 8 , 8 , 12 ,  seed = 15)


csrA = csr_matrix(A)
csrB = csr_matrix(B)

def tuple_matcher(tp1 : List[List[int]],tp2 : List[List[int]]): 
    predict = []
    myspace = []
    i : int = 0
    for a_tuple in tp1:
        j : int = 0
        for b_tuple in tp2:
            if a_tuple[1] == b_tuple[0]:
                predict.append([*a_tuple, b_tuple[1]])
                myspace.append([i,j])
                j += 1
                pass
        i += 1
    return predict, myspace
#To predict the iterspace. Count the  number of 

tp1 = np.column_stack((csrA.nonzero()[0], csrA.nonzero()[1]))
tp2 = np.column_stack((csrB.nonzero()[0], csrB.nonzero()[1]))
iter_space , myspace = tuple_matcher(tp1,tp2)
# Run gustav_mult_opt 100,000 times
print(*iter_space, sep=" ")
print("\n")
print(*tp1, sep=" ")
print("\n")
print(*tp2, sep=" ")
print()
x_values = [t[0] for t in myspace]
y_values = [t[1] for t in myspace]

plt.scatter(x_values, y_values, marker='x', color='r')

plt.xlim(0, max(x_values))  # Set x-axis limits from 0 to maximum x value

plt.xticks(range(-1,int(max(x_values))+2))  # Set x-axis grid to integer coordinates
plt.yticks(range(int(max(y_values))+10))  # Set y-axis grid to integer coordinates

plt.gca().set_aspect('equal')  # Set x and y axes to be equal and make grid square

plt.grid(True)  # Add grids to the plot
plt.show()