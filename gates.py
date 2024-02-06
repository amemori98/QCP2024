import numpy as np

class gates(object):
  def __init__(self, qubit):
    self.qubit = qubit
    
  def apply_hadamard(self, qubit):
    H = np.array([[1, 1], [1, -1]])
    x = 1 / np.sqrt(2)
    hada = np.multiply(H, x)
    hadamard = np.multiply(hada, qubit)
    return hadamard
    
  def apply_phase_shift(self, phi, qubit):
    phase = np.array([1, 0],[0, np.exp(1j*phi)])
    phase_shift = np.multiply(phase, qubit)
    return phase_shift

  def apply_cnot(self, control, target):
    if control[0] == 0 and control[1] == 1:
      target[0] = target[1]
      target[1] = target[0]  # flipping the target qubit
    elif control[0] == 1 and control[1] == 0:
      pass  # do nothing
    else:
      pass  # do nothing
    return control, target
  def apply_controlled_v(self, qubit):
    V = np.array([[1, 0], [0, 1j]])
    controlled_v = np.multiply(V, qubit)
    return controlled_v
  def and_gate(self, qubit):
    
    
    
    
    
      
    
    
