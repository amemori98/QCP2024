from simulator import Sparse, SparseRep, Dense, state, CNOT, H, X, Y, Z, T, S
#from main import general_grovers_algorithm
import numpy as np
import time
import scipy
#testing

# Testing the performance of Dense vs Sparse Matrix class

# matrix multiplication and tensor product of NxN matrices

time_sparse = np.empty([0], dtype=float)
time_dense = np.empty([0], dtype=float)

for N in np.arange(20, 101, 20):

    a = scipy.sparse.random(N, N, density=0.1).toarray() # can modify to test density
    b = scipy.sparse.random(N, N, density=0.1).toarray()
    asparse = Sparse(a)
    bsparse = Sparse(b)
    t1 = time.time()
    test = asparse*bsparse # can change this to test the tensor product
    t2 = time.time()
    t3 = t2-t1
    time_sparse = np.append(time_sparse, [t3])

    adense = Dense(a)
    bdense = Dense(b)
    t4 = time.time()
    test = adense*bdense
    t5 = time.time()
    t6 = t5-t4
    time_dense = np.append(time_dense, [t6])




"""
a = np.array([[3, 2+3j, 1], [0, 0+2j, 0], [0, 3, 4+1j]])
b = np.array([[5, 1, 2], [7, 4, 7], [1, 2, 3]])
a = Dense(a,"A")
b = Dense(b)
print(a+b)
print(a%b)
print(a.scalar(2))
print(a.transpose())
print(a-b)
print(a*b)
print(a.adjoint())
print(a.transpose().id)
print(a.adjoint().id)

We can use something similar to this code, to test the performance of the sparse matrix class against the dense matrix class. 
Could be useful to include in the report and something to talk about during the presentation. For 16x16 matrices, the performance of the sparse matrix class is around 26x faster than the dense matrix class. We would have to test this on different size matrices and see how much the performance difference is. Also have to test the kronecker product.
"""

