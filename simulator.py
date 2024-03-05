#use simulator to implement classes of matrices etc. -> simulator interface should be completely separate

import numpy as np


class Matrix(object):

    def __init__(self, array):
        self.matrix = array
        self.rows, self.cols = self.matrix.shape

    def __mul__(self, other):
        # matrix multiplication
        assert (self.rows == other.cols) and (self.cols == other.rows), "Matrix order"
        multiply = np.zeros((self.rows, other.cols),dtype=complex)
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(other.rows):
                    multiply[i, j] += self.matrix[i, k] * other.matrix[k, j]
        return Matrix(multiply)

    def __mod__(self, other):
        # tbh this is more of a kronecker product rather than tensor product but well
        tensor = np.zeros((self.rows * other.rows, self.cols * other.cols), dtype=complex)

        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(other.rows):
                    for l in range(other.cols):
                        tensor[i * other.rows + k, j * other.cols + l] = self.matrix[i, j] * other.matrix[k, l]

        return Matrix(tensor)

    def __str__(self):
        return str(self.matrix)


class SparseMatrix(object):

    def __init__(self, array):
        """
        SparseMatrix(np.ndarray) -> Creates a SparseMatrix from a numpy array and stores it as a SparseRep
        """
        assert isinstance(array, np.ndarray) or isinstance(
            array, SparseRep), "Argument is not an array or Sparse Representation"

        if isinstance(array, np.ndarray):

            self.rows, self.cols = array.shape
            elements = []
            for i in range(self.rows):
                for j in range(self.cols):
                    if array[i, j] != 0:
                        elements.append([array[i, j], i, j])
            self.elements = SparseRep(elements, self.rows, self.cols)
        else:
            self.elements = array.elements
            self.rows = array.maxrows
            self.cols = array.maxcols

    def __str__(self):
        self.elements = elements
        self.rows = max([i[1] for i in elements]) + 1
        self.cols = max([i[2] for i in elements]) + 1
        self.matrix = np.zeros((self.rows, self.cols))
        for i in elements:
            self.matrix[i[1], i[2]] = i[0]


class SparseRep(object):

    def __init__(self, elements, maxrows, maxcols):
        self.elements = elements
        self.maxrows = maxrows
        self.maxcols = maxcols


#testing
a = np.array([[3, 2]])
b = np.array([[5, 1], [7, 4]])
a = Matrix(a)
b = Matrix(b)
c = b % a
print(c)
print(a % b)

i = Matrix(np.array([[1, 0], [0, 1]]))
print(i)
t = i % i % i  #multiple not working -> should be 8x8 identity, associativity not working
print(t)

#zero = Matrix(np.array([1, 0]))
#one = Matrix(np.array([0, 1]))
#zerozero = zero % zero
#print(zerozero)
