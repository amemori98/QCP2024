import numpy as np


class Gates(object):

  def __init__(self, n):
    self.n = n
    self.state = np.zeros(2**self.n, dtype=complex)
    self.state[0] = 1  #not sure why this is set to 1?

  # Apply Harmard gate to given qubit
  def apply_hadamard(self, qubit):
    H = np.array([[1, 1], [1, -1]])
    x = 1 / np.sqrt(2)
    hada = np.multiply(H, x)
    hadamard = np.multiply(hada, qubit)
    return hadamard

  # Apply phase shift to given qubit
  def apply_phase_shift(self, phi, qubit):
    phase = np.array([1, 0], [0, np.exp(1j * phi)])
    phase_shift = np.multiply(phase, qubit)
    return phase_shift

  # Apply CNOT to given qubit
  def apply_cnot(self, control, target):
    target = control  # target become copy of control
    return control, target  #  return both target and control

  # Apply controlled_v to given qubit
  def apply_controlled_v(self, qubit):
    V = np.array([[1, 0], [0, 1j]])
    controlled_v = np.multiply(V, qubit)
    return controlled_v

  # Not_gate takes one state to the other
  def not_gate(self, qubit):
    not_gate = np.array([[0, 1], [1, 0]])
    not_qubit = np.multiply(not_gate, qubit)
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

  #def apply_gate(self, ): #reduced repetition in gate functions

  #def measure(self):
