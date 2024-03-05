import numpy as np
from simulator import Matrix, SparseMatrix


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
    state of the register 
  
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
    self.n = n  # number of qubits
    self.N = 2**n  # number of states
    self.state = np.zeros(
        2**self.n,
        dtype=complex)  # state vector of size 2^n, initialized to |0>
    self.state[0] = 1  # initial state is |0>

  def apply_gate(self, gate, qubit):
    #apply the gate to the given qubit and update the state

    #count how many identities needed then tensor 2x2s
    #Il = np.eye(2**qubit)
    #Ir = np.eye(2**(self.n - qubit - gate.shape[0]**0.5))
    identity = Matrix(np.array([[1, 0], [0, 1]]))

    Il = identity
    for i in range(2**qubit):
      Il = Il % identity  #working on1 this (alex)

    Ir = identity
    for i in range(2**(self.n - qubit - int(gate.rows**0.5))):
      Ir = Ir % identity

    t = Il % gate % Ir
    self.state = t % self.state

  # Apply hadamard gates to all qubits
  def apply_hadamard_all(self):
    x = 1 / np.sqrt(2)
    H = Matrix(np.array([[x * 1, x * 1], [x * 1, -x * 1]]))

    t = H
    for i in range(0, self.n):
      t = t % H
    print(t)

    self.state = t % self.state
    print(self.state)

  # Apply Hadamard gate to a qubit
  def apply_hadamard(self, qubit):
    x = 1 / np.sqrt(2)
    H = Matrix(np.array([[x * 1, x * 1], [x * 1, -x * 1]]))
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
