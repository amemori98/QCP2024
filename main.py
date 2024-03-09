#main is for calling simulator and implementing the algorithms and visualisation (if any)
#need to import other files so can call classes etc.

from gates import State
from simulator import Sparse
import numpy as np


#Grover's algorithm
def grovers_algorithm(n_qubits):

  #set up intial state
  psi = State(n_qubits)

  #apply hadamard gate to each qubit simultaneously
  psi.apply_hadamard_all()

  #apply the oracle

  #apply the diffusion

  #measure


def grovers_algorithm_2(): #general version will have input for n qubits and selecting target 
  """
  2 qubit Grover's algorithm, need 3rd qubit (measurement qubit)?
  Target set to |11> = |3>
  This skips using the state class (for now) 
  """

  #basis states
  zero = Sparse(np.array([[1], [0]]))
  one = Sparse(np.array([[0], [1]]))

  #gates
  I = Sparse(np.array([[1, 0], [0, 1]]))  #Identity
  H = Sparse((1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]))  #Hadamard
  X = Sparse(np.array([[0, 1], [1, 0]]))  #Pauli X 
  Z = Sparse(np.array([[1, 0], [0, -1]])) #Pauli Z 
  CNOT = Sparse(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])) #Controlled-NOT 
  CZ = Sparse(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])) #Controlled-Z

  
  #set up initial state
  state = zero % zero 
  
  #apply hadamard gate to all qubits 
  state = (H % H) * state  
    
  #apply oracle and diffusion sqrt(N) times
  for i in range(int(np.sqrt(2**2))): #int or round?

    i_state = state #story initial state 
    print(i_state.cols)
    print(i_state.elements)
    print(i_state)
    O = CZ  #Oracle O = 1 - 2|11><11| = CZ (need to change so general)
    state = O * state 
    
    D = I - (i_state*(i_state.transpose())) - (i_state*(i_state.transpose()))#Diffuser D = 1 - 2|initial state><initial state|, need Sparse subtraction method, outer product of vectors? -> can possibly build this from basic gates but more confusing 
    
    state = D * state 
    
  #measure qubits -> should now be in |11> state 
  print(state) 


#testing 
#grovers_algorithm(2)
grovers_algorithm_2()
