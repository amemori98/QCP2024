#use simulator to implement classes of matrices etc. -> simulator interface should be completely separate

import numpy as np
import time  # to check performance, no point using sparse if its slower than normal matrix
import scipy  # for generating random sparse matrices, testing purposes


class Matrix(object):
    """
    Class for initial testing of the quantum computer simulator. 
    Also used for testing the performance of the sparse matrix class
    """

    def __init__(self, array):
        """
        Converts a 2D np.ndarray into a matrix object.
        """
        self.matrix = array
        self.rows, self.cols = array.shape

    def __mul__(self, other):
        """
        Matrix multiplication of 2 matrices
        """
        assert (self.cols == other.rows
                ), "Can only multiply 2 matrices of dimensions (m,n) and (n,p)"
        multiply = np.zeros((self.rows, other.cols), dtype=complex)
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(other.rows):
                    multiply[i, j] += self.matrix[i, k] * other.matrix[k, j]
        return Matrix(multiply)

    def __mod__(self, other):
        """
        Kronecker product of two matrices
        """
        tensor = np.zeros((self.rows * other.rows, self.cols * other.cols),
                          dtype=complex)

        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(other.rows):
                    for l in range(other.cols):
                        tensor[i * other.rows + k, j * other.cols +
                               l] = self.matrix[i, j] * other.matrix[k, l]

        return Matrix(tensor)

    def __add__(self, other):
        """
        Add two matrices. This is needed to ensure that matrices added together are of the same dimension.
        """
        assert (self.rows == other.rows
                and self.cols == other.cols), "Matrix dimensions do not match"
        return Matrix(self.matrix + other.matrix)

    def __sub__(self, other):
        """
        Subtract two matrices. This is needed to ensure that matrices added together are of the same dimension.
        """
        assert (self.rows == other.rows
                and self.cols == other.cols), "Matrix dimensions do not match"
        return Matrix(self.matrix - other.matrix)

    def transpose(self):
        """
        Transpose a matrix
        """
        zero = np.zeros((self.cols, self.rows), dtype=complex)
        for i in range(self.cols):
            for j in range(self.rows):
                zero[i, j] = self.matrix[j, i]
        return Matrix(zero)

    def scalar(self, scale):
        """
        Multiply a matrix by a scalar
        """
        return Matrix(self.matrix * scale)

    def __str__(self):
        return str(self.matrix)


class SparseRep(object):
    """
    This class exists to distinguish between an array or a sparse matrix when running the constructor for the Sparse matrix class. Also used to return a sparse matrix when multiplying or using the kronecker product for 2 sparse matrices.
    """

    def __init__(self, elements, indices, rows, cols):

        self.elements = elements
        self.indices = indices
        self.rows = rows
        self.cols = cols


class Sparse(object):
    # data type issue with numpy
    """
    2D sparse matrix class with entries of type complex
    """

    def __init__(self, array):
        """
        Sparse(np.ndarray) -> Converts the array into a sparse matrix. Elements of the matrix are stores in an array whose elements are in the following format [Matrix element] [row, column]. Row and column are zero indexed.

        Sparse(SparseRep) -> Converts the sparse representation into a sparse matrix. This is used to distinguish between converting an array containing a matrix and an array containing a sparse representation of a matrix [Matrix element, row, column).

        Sparse(Matrix) -> Converts a dense matrix into a sparse matrix.
        The rows and columns attributes refer to the maximum rows and columns of the matrix. 
        """
        assert isinstance(array, (np.ndarray, SparseRep, Matrix)), "Can only convert an array, SparseRep or Matrix classes"

        # converts array to sparse matrix
        if isinstance(array, np.ndarray):
            self.rows, self.cols = array.shape
            elements = []
            indices = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if array[i, j] != 0:
                        elements.append(array[i, j])
                        indices.append([i, j])
            self.elements = np.array(elements, dtype = complex)
            self.indices = np.array(indices, dtype = int)

        # converts Matrix to sparse matrix
        elif isinstance(array, Matrix):
            self.rows, self.cols = array.rows, array.cols
            elements = []
            indices = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if array.matrix[i, j] != 0:
                        elements.append(array.matrix[i, j])
                        indices.append([i, j])
            self.elements = np.array(elements, dtype = complex)
            self.indices = np.array(indices, dtype = int)

        # stores sparse representation as sparse matrix
        else:
            self.elements = array.elements
            self.indices = array.indices
            self.rows = array.rows
            self.cols = array.cols
        self.shape = np.array((self.rows, self.cols))

    def scalar(self, scale):
        """
        Multiplies the matrix by a scalar
        """
        elements = self.elements * scale
        return Sparse(SparseRep(elements, self.indices, self.rows, self.cols))

    def __mul__(self, other):
        # add multiplication with scalar
        """
        Matrix multiplication of two sparse matrices
        """

        assert (self.cols == other.rows), "Can only multiply 2 matrices of dimensions (m,n) and (n,p)"
        multiply = []
        m_indices = []
        # loop through all the elements and mutiply the ones that are in the same row and column
        for i in range(len(self.elements)):
            for j in range(len(other.elements)):
                if self.indices[i,1] == other.indices[j,0]:
                    multiply.append(self.elements[i] * other.elements[j])
                    m_indices.append([self.indices[i,0],other.indices[j,1]])
        # this will give an array that has repeated entries of the same row and column
        # so we need to remove the duplicates and sum the elements
        # lists to store unique sets of indices
        elements = []
        unique = []
        for k in range(len(multiply)):

            if m_indices[k] not in unique:
                unique.append(m_indices[k])
                elements.append(multiply[k])
            else:
                ind = unique.index(m_indices[k])
                elements[ind] += multiply[k]
        elements = np.array(elements)
        unique = np.array(unique)
        return Sparse(SparseRep(elements, unique, self.rows, other.cols))

    def __add__(self, other):
        """
        Add two sparse matrices
        """
        assert (self.rows == other.rows and self.cols == other.cols), "Matrix dimensions do not match"
        add = []
        add.extend(self.elements.tolist())
        add.extend(other.elements.tolist())
        a_indices = []
        a_indices.extend(self.indices.tolist())
        a_indices.extend(other.indices.tolist())
        elements = []
        unique = []
        for k in range(len(add)):
            if a_indices[k] not in unique:
                unique.append(a_indices[k])
                elements.append(add[k])
            else:
                ind = unique.index(a_indices[k])
                elements[ind] += add[k]
        elements = np.array(elements)
        unique = np.array(unique)
        return Sparse(SparseRep(elements, unique, self.rows, self.cols))

    def __sub__(self, other):
        """
        Subtract two sparse matrices
        """
        assert (self.rows == other.rows and self.cols == other.cols), "Matrix dimensions do not match"
        elements = self.elements.tolist()
        unique = self.indices.tolist()
        other_indices = other.indices.tolist()
        
        for i in range(len(other.elements)):
            if other_indices[i] in unique:
                ind = unique.index(other_indices[i])
                elements[ind] -= other.elements[i]
            else:
                elements.append(-other.elements[i])
                unique.append(other_indices[i])
        elements = np.array(elements)
        unique = np.array(unique)
        return Sparse(SparseRep(elements, unique, self.rows, self.cols))

    def __mod__(self, other):
        """
        Kronecker product of two sparse matrices
        """
        kronecker = []
        indices = []

        for i in range(len(self.elements)):
            for j in range(len(other.elements)):
                kronecker.append(self.elements[i]*other.elements[j])
                indices.append([self.indices[i,0]*other.rows+other.indices[j,0],
                                self.indices[i,1]*other.cols+other.indices[j,1]])
                
        kronecker = np.array(kronecker)
        indices = np.array(indices)
        return Sparse(SparseRep(kronecker, indices, self.rows * other.rows,self.cols * other.cols))

    def transpose(self):
        """
        Transpose of a sparse matrix
        """
        indices = []
        
        for i in self.indices:
            indices.append([i[1],i[0]])
        indices = np.array(indices)
        return Sparse(SparseRep(self.elements, indices, self.cols, self.rows))

    def __str__(self):
        matrix = np.zeros((self.rows, self.cols), dtype = complex)
        for i in range(len(self.elements)):
            matrix[self.indices[i,0],self.indices[i,1]] = self.elements[i]
        return str(matrix)



#class programmer(object):

#testing
"""
a = np.array([[3, 2, 1], 
      [0, 0, 0], 
      [0, 3, 4]])
b = np.array([[5, 1, 2], 
      [7, 4, 7],
      [1, 2, 3]])
a = Sparse(a)
b = Sparse(b)
print(a+b)


c = Sparse(np.array([[3,2]]))
d = Sparse(np.array([[5,1],[7,4]]))
print(d % c) 

print(a.scalar(2)) # not working 
"""
"""
We can use something similar to this code, to test the performance of the sparse matrix class against the dense matrix class. 
Could be useful to include in the report and something to talk about during the presentation. For 16x16 matrices, the performance of the sparse matrix class is around 26x faster than the dense matrix class. We would have to test this on different size matrices and see how much the performance difference is. Also have to test the kronecker product.

c = scipy.sparse.random(16,16,density=0.1).toarray()
d = scipy.sparse.random(16,16,density=0.1).toarray()
csparse = Sparse(c)
dsparse = Sparse(d)
t1 = time.time()
test = csparse*dsparse
t2 = time.time()
t3 = t2-t1
print("sparse", t3)

cdense = Matrix(c)
ddense = Matrix(d)
t4 = time.time()
test = cdense*ddense
t5 = time.time()
t6 = t5-t4
print("dense", t6)

print("Sparse is", t6/t3, "times faster")

"""
