#use simulator to implement classes of matrices etc. -> simulator interface should be completely separate

# MATRIX abstract base class with DENSE and SPARSE child classes, SPARSEREP is a helper class 

from abc import ABC, abstractmethod
import numpy as np
import math
import time # to check performance, no point using sparse if its slower than normal matrix
import scipy # for generating random sparse matrices, testing purposes

class Matrix(ABC):
    @abstractmethod
    def __add__(self, other):
        pass
    
    @abstractmethod
    def __sub__(self, other):
        pass
    
    @abstractmethod
    def __mul__(self, other):
        pass
    
    @abstractmethod
    def __mod__(self, other):
        pass
    
    @abstractmethod
    def __str__(self):
        pass
    
    @abstractmethod
    def transpose(self):
        pass
    
    @abstractmethod
    def scalar(self, scale):
        pass

class Dense(Matrix):
    """
    Class for initial testing of the quantum computer simulator. 
    Also used for testing the performance of the sparse matrix class
    """

    def __init__(self, array, id=""):
        """
        Converts a 2D np.ndarray into a matrix object.
        """
        self.matrix = array
        self.rows, self.cols = array.shape
        self.id = id

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
        return Dense(multiply)

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

        return Dense(tensor)

    def __add__(self, other):
        """
        Add two matrices. This is needed to ensure that matrices added together are of the same dimension.
        """
        assert (self.rows == other.rows
                and self.cols == other.cols), "Matrix dimensions do not match"
        return Dense(self.matrix + other.matrix)

    def __sub__(self, other):
        """
        Subtract two matrices. This is needed to ensure that matrices added together are of the same dimension.
        """
        assert (self.rows == other.rows
                and self.cols == other.cols), "Matrix dimensions do not match"
        return Dense(self.matrix - other.matrix)

    def transpose(self):
        """
        Transpose a matrix
        """
        zero = np.zeros((self.cols, self.rows), dtype=complex)
        for i in range(self.cols):
            for j in range(self.rows):
                zero[i, j] = self.matrix[j, i]
        return Dense(zero)

    def scalar(self, scale):
        """
        Multiply a matrix by a scalar
        """
        return Dense(self.matrix * scale)

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


class Sparse(Matrix):
    """
    2D sparse matrix class with entries of type complex
    """

    def __init__(self, array, id=""):
        """
        Sparse(np.ndarray) -> Converts the array into a sparse matrix. 
        Elements of the matrix are stored in an array whose elements are in the following format [Matrix element] [row, column]. 
        Row and column are zero indexed.

        Sparse(SparseRep) -> Converts the sparse representation into a sparse matrix. 
        This is used to distinguish between converting an array containing a matrix and an array containing a sparse representation of a matrix [Matrix element, row, column).

        Sparse(Matrix) -> Converts a dense matrix into a sparse matrix.
        The rows and columns attributes refer to the maximum rows and columns of the matrix. 

        id argument is the string identifier of the matrix. It is used to print circuit diagrams using the programmer class. It is a string of max length 3
        """
        assert isinstance(array, (np.ndarray, SparseRep, Dense)), "Can only convert an array, SparseRep or Matrix classes"

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

        # converts Dense to sparse matrix
        elif isinstance(array, Dense):
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
        return Sparse(SparseRep(kronecker, indices, self.rows * other.rows, self.cols * other.cols))

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



class programmer(object):
    # might have to redesign to be able to implement cnot and cv
    def __init__(self, register):
        self.initial = register
        # finds number of qubits based on rows of the register
        rows = register.rows
        self.qubit_count = 0
        while True:
            rows = rows/2
            self.qubit_count += 1
            # stops if it reaches 1
            if rows == 1:
                break
            # raises exception to invalid register
            if rows < 1:
                raise Exception("Invalid register size, must have 2^n rows")
        self.steps = []
        # dictionary to know how to print each gate
        self.gates_dict = {H:"H ",I:"I ",X:"X ",Z:"Z ",CNOT:"C0",CZ:"CZ"}
        
    def add_step(self, gates):
        assert len(gates) == self.qubit_count, "Number of gates does not match number of qubits"
        self.steps.append(gates)
        print(self)
        return self
        
    def remove_step(self, step_number):
        self.steps.remove(self.steps[step_number])
        print(self)
        return self

    def __str__(self):
        assert len(self.steps)>0,"Must have at least one step"
        print(len(self.steps))
        buffer = ""
        for i in range(self.qubit_count):
            buffer = buffer + "q"+str(i)+" -> "
            for j in range(len(self.steps)):
                buffer = buffer + self.gates_dict[self.steps[j][i]] + " "
            buffer = buffer + "\n"
        return buffer

def state(n, m):
    # fixed so it properly initializes the register and uses Sparse
    """
        Nx1 Matrix state of the quantum register initialised to state corresponding to qubit m
    """
    assert isinstance(m,int), "The number of qubits n inputted must be an integer"
    assert isinstance(n,int), "The qubit to which quantum register is intialised m must be an integer"
    assert (m >= 0 and m < 2**n), "m must be between 0 and 2^n"

    # initialise register to all zeros
    state = np.zeros((2**n, 1), dtype=complex)

    # initialize to state |m>
    state[m] = 1

    return Sparse(state)

# global gates
I = Sparse(np.array([[1, 0], [0, 1]]),id = "I")  #Identity
H = Sparse((1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]), id = "H")  #Hadamard
X = Sparse(np.array([[0, 1], [1, 0]]), id = "X")  #Pauli X 
Y = Sparse(np.array([[0, -1j], [1j, 0]]), id = "Y") #Pauli Y
Z = Sparse(np.array([[1, 0], [0, -1]]), id = "Z") #Pauli Z 

def CNOT(qubit_count, control_list, target_list):
    """
    Returns an appropriate CNOT-Type gate for the given qubit count, control and target qubits as a Sparse matrix.
    This gate flips all the target qubits, if all the control qubits are |1>
    Control and Target qubits are zero-indexed and are to be inputted as lists of integers.

    Example:
    To construct a Toffoli gate for a 3 qubit register, we require 2 control qubits and 1 target qubit.
    Toffoli = CNOT(3, [0,1], [2])
    """
    assert qubit_count >= (len(control_list) + len(target_list)), "Number of qubits must be greater than or equal to the number of control and target qubits"
    assert isinstance(qubit_count, int), "Qubit count must be an integer"
    assert isinstance(control_list, list), "Control qubits must be provided as a list"
    assert isinstance(target_list, list), "Target qubits must be provided as a list"
    for control in control_list:
        assert isinstance(control, int), "One or more control qubits is not an integer"
    for target in target_list:
        assert isinstance(target, int), "One or more target qubits is not an integer

    # initializes a zero array
    gate = np.zeros((2**qubit_count, 2**qubit_count), dtype=complex)

    # initializes a list to store the indices of the rows
    rows = np.arange(0,2**qubit_count,1)
    
    # converts the indices to binary
    bin_rows = []
    for row in rows:
        bin_rows.append(bin(row)[2:].zfill(qubit_count))

    # initializes a list to store the swapped binary indices
    bin_swapped_rows = []

    # flips bits in the binary representation of the row indices according to the required CNOT-Type Gate
    for binary in bin_rows:
        counter = 0
        for i in range(len(control_list)):
            if binary[qubit_count - control_list[i] - 1] == "1":
                counter += 1
        if counter == len(control_list):
            for j in range(len(target_list)):
                if binary[qubit_count - target_list[j] - 1] == "1":    
                    buffer = list(binary)
                    buffer[qubit_count - target_list[j] - 1] = "0"
                    binary = "".join(buffer)
                else:
                    buffer = list(binary)
                    buffer[qubit_count - target_list[j] - 1] = "1"
                    binary = "".join(buffer)  
        bin_swapped_rows.append(binary)

    # initializes a list to store the swapped integer indices
    swapped_rows = []
    
    # converts the binary representation of the row indices back to decima
    for binary in bin_swapped_rows:
        swapped_rows.append(int(binary,2))

    # sets the appropriate ones in the gate
    for i in range(len(swapped_rows)):
        gate[rows[i], swapped_rows[i]] = 1
        
    return Sparse(gate)
#print(CNOT(3,1,0))
#print(CNOT(4,3,1))
#print(H%I%H)

wikipedia_toff = TOFF(4,[0,2],[1])
print(wikipedia_toff)
q = programmer(state(4,2))
q.add_step([H,H,H,H])
q.add_step([H,Z,X,I])
