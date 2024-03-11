import numpy as np
from simulator import Matrix, Sparse


class State(object):
  """
		A class to represent a quantum register.

		...
	
		Attributes
		------------
		n : int
				number of qubits 
		N : int 
				number of states 
		state : list
				current state of the register 
	
		Methods
		-----------
		apply_hadamard
		apply_phase_shift
		apply_cnot
		apply_controlled_v
		measure 
	
		"""

  def __init__(self, n):
    """
				construct the quantum register 
				"""
    # we need to add a method to show a the state of a specific qubit (Jordi)
    self.n = n  # number of qubits
    self.N = 2**n  # number of basis states
    self.state = np.zeros(
        (self.N, 1),
        dtype=complex)  # state vector of size 2^n, initialized to |0>
    self.state[0] = 1  # initial state is |0>
    self.state = Matrix(
        self.state)  # needs to be matrix object to use tensor product

  def apply_gate(self, gate, qubit):  #working on this (alex)
    #apply the gate to the given qubit and update the state

    #count how many qubits either side of one to apply gate to
    nl = qubit  #make identity matrix of this dimension
    Il = [[0] * nl for _ in range(nl)]
    for i in range(nl):
      Il[i][i] = 1
    Il = Matrix(np.array(Il))  #won't work with empty array

    nr = self.n - qubit  #make identity matrix of this dimension
    Ir = [[0] * nr for _ in range(nr)]
    for i in range(nr):
      Ir[i][i] = 1
    Ir = Matrix(np.array(Ir))

    print(Il)
    print(Ir)
    t = Il % gate % Ir

    self.state = t * self.state  #matrix multiplication

  # Apply hadamard gates to all qubits
  def apply_hadamard_all(self):
    H = Matrix((1 / np.sqrt(2)) * np.array([[1., 1.], [1., -1.]]))

    t = H
    for i in range(0, self.n - 1):
      t = t % H

    self.state = t * self.state

  # Apply Hadamard gate to a qubit
  def apply_hadamard(self, qubit):
    H = Matrix((1 / np.sqrt(2)) * np.array([[1., 1.], [1., -1.]]))
    self.apply_gate(H, qubit)

  # Apply phase shift to given qubit
  def apply_phase_shift(self, phi, qubit):
    phase = Matrix(np.array([1, 0], [0, np.exp(1j * phi)]))
    self.state = phase.tensor()
    return phase_shift

  # Apply CNOT to given qubit
  def apply_cnot(self, control, target):
    target = control  # target become copy of control
    cnot = Matrix(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
    return control, target  #  return both target and control

  # Apply controlled_v to given qubit
  def apply_controlled_v(self, qubit):
    V = Matrix(np.array([[1, 0], [0, 1j]]))
    controlled_v = np.matmul(V, qubit)
    return controlled_v

  # Not_gate takes one state to the other
  def not_gate(self, qubit):
    not_gate = np.array([[0, 1], [1, 0]])
    not_qubit = np.matmul(not_gate, qubit)
    return not_qubit

  # EPR pair
  def epr_pair(self, qubit1, qubit2):
    H_qubit = self.apply_hadamard(qubit1)
    control, target = self.apply_cnot(H_qubit, qubit2)
    '''insert this line qunatum state consists of control and target'''
    return state

  #print method - quickly see current state without measurement
  def __str__(self):
    return (
        f'There are {self.n} qubits, the state of the register is {self.state}'
    )

  #def measure(self):


#new structure 

def qubit(state):
  if state==0:
    return Sparse([1,0])

  elif state==1:
    return Sparse([0,1])
      
  else:
    print('Error: "state" must be 0 or 1') #error message 
      

class QCircuit(object):
  
  def __init__(self, n):

    self.n = n #number of qubits 
    self.N = 2**n #size of state space 

    #set up initial state of quantum register (all qubits |0>)
    self.state = qubit(0) 
    for i in range(n-1):
      self.state = self.state % qubit(0)

  
  #def apply_H(self, *args):
    """
    apply hadamard gate to qubits given in argument 
    """

  
    


    

#global variables 

#gates
I = Sparse(np.array([[1, 0], [0, 1]]))  #Identity
H = Sparse((1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]))  #Hadamard
X = Sparse(np.array([[0, 1], [1, 0]]))  #Pauli X 
Z = Sparse(np.array([[1, 0], [0, -1]])) #Pauli Z 
CNOT = Sparse(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])) #Controlled-NOT 
CZ = Sparse(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])) #Controlled-Z