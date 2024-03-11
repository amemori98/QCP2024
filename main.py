#main is for calling simulator and implementing the algorithms and visualisation (if any)
#need to import other files so can call classes etc.

from gates import State
from simulator import Sparse, Matrix
import numpy as np
import random
from matplotlib import pyplot as plt


def grovers_algorithm_2():  #general version will have input for n qubits and selecting target
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
    H = Sparse((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]))  #Hadamard
    X = Sparse(np.array([[0, 1], [1, 0]]))  #Pauli X
    Z = Sparse(np.array([[1, 0], [0, -1]]))  #Pauli Z
    CNOT = Sparse(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                  [0, 0, 1, 0]]))  #Controlled-NOT
    CZ = Sparse(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                  [0, 0, 0, -1]]))  #Controlled-Z

    #set up initial state
    state = zero % zero

    #set target to |11>
    target = one % one

    #apply hadamard gate to all qubits
    state = (H % H) * state

    #apply oracle and diffusion sqrt(N) times
    for i in range(int(np.sqrt(2**2))):  #int or round?

        i_state = state  #story initial state
        print(i_state.cols)
        print(i_state.elements)
        print(i_state)
        #O = CZ  #Oracle O = 1 - 2|11><11| = CZ
        O = (I % I) - (target * (target.transpose())) - (target *
                                                         (target.transpose()))
        state = O * state

        D = (I % I) - (i_state * (i_state.transpose())) - (
            i_state * (i_state.transpose())
        )  #Diffuser D = 1 - 2|initial state><initial state|

        state = D * state

    #measure qubits -> should now be in |11> state
    print(state)


#testing
#grovers_algorithm_2()


def grovers_algorithm_3():
    """
		2 qubit Grover's algorithm, need 3rd qubit (measurement qubit)?
		Target set to |11> = |3>
		This skips using the state class (for now) 
		"""

    # basis states
    zero = Matrix(np.array([[1], [0]]))  # MATRIX class
    one = Sparse(np.array([[0], [1]]))

    # gates
    # I = Sparse(np.array([[1, 0], [0, 1]]))  # Identity
    # H = Sparse((1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]))  # Hadamard

    I = Matrix(np.array([[1, 0], [0, 1]]))  # Identity MATRIX class
    H = Matrix((1 / np.sqrt(2)) *
               np.array([[1, 1], [1, -1]]))  # Hadamard MATRIX class

    # not used gates
    X = Sparse(np.array([[0, 1], [1, 0]]))  # Pauli X
    Z = Sparse(np.array([[1, 0], [0, -1]]))  # Pauli Z
    CNOT = Sparse(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                  [0, 0, 1, 0]]))  # Controlled-NOT

    CZ = Matrix(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                  [0, 0, 0, -1]]))  # Controlled-Z

    # set up initial state
    state = zero % zero
    print("-------------------")
    print("1: create 2-qubit state")
    print(state)

    # apply hadamard gate to all qubits
    state = (H % H) * state
    print("2: put the state into a superposition of all basis states")
    print(state)
    print("-------------------")

    # apply oracle and diffusion sqrt(N) times
    for i in range(int(np.sqrt(2**2))):  # int
        print("3: index of loop", i)

        i_state = state  # store initial state
        print("4: store initial state")
        print(i_state)

        O = CZ  # Oracle O = 1 - 2|11><11| = CZ (need to change so general)
        print("5: oracle = CZ")
        print(O)

        state = O * state
        print("6: apply the oracle to state")
        print(state)

        D = (I % I) - ((i_state * (i_state.transpose()))).scalar(
            2
        )  # Diffuser D = 1 - 2|initial state><initial state|, need Sparse subtraction method, outer product of vectors? -> can possibly build this from basic gates but more confusing
        print("7: diffuser")
        print(D)

        state = D * state
        print("8: apply diffuser to the state")
        print(state)
        print("-------------------")

    # measure qubits -> should now be in |11> state --> it IS!
    print("9: final state |11>")
    print(state)


#grovers_algorithm_3() # works for 2 qubit system (hardcoded without state class) and using MATRIX class not Sparse

# -------------------------------------------------------------------------------------

# general Grover's algorithm for n qubit quantum register
# uses the functions defined below (a state function, nxn matrix gate function, and a apply gate function) NOT the state class

# have also noticed that for a 3-qubit system it doesn't seem to work well... with the final state having less amplitude in the target state than all the rest so maybe we should increase the no of iterations the oracle and diffuser are applied to the 3-qubit system!


def state(n, m):
    """
		nx1 Matrix state of the quantum register initialised to state corresponding to qubit m
		"""
    assert (
        type(n) == int), "The number of qubits n inputted must be an integer"
    assert (
        type(m) == int
    ), "The qubit to which quantum register is intialised m must be an integer"

    N = 2**n
    state = np.zeros((2**n, 1), dtype=complex)
    state[m] = 1  # initialise state to |m>

    return Matrix(state)


def nxn_matrix_gate(matrix_gate, n):
    """
		NxN Matrix gate given a Matrix gate and the number of qubits n
		"""
    assert (type(matrix_gate) == Matrix
            ), "The gate inputted must be a Matrix object"
    assert (
        type(n) == int), "The number of qubits n inputted must be an integer"

    gate_n = matrix_gate

    for i in range(n - 1):
        gate_n = gate_n % matrix_gate

    return gate_n


def identity(N):
    """
		NxN identity Matrix object
		"""
    I = []
    for i in range(N):
        row = [0] * N
        row[i] = 1
        I.append(row)
    return Matrix(np.array(I))


def apply_nxn_gate_to_qr(nxn_gate, matrix_state):
    """
		Applies an NxN gate to a Matrix state
    """
    assert (type(matrix_state) == Matrix
            ), "The state inputted must be a Matrix object"
    assert (type(nxn_gate) == Matrix
            ), "The gate inputted must be an NxN Matrix object"

    matrix_state = nxn_gate * matrix_state

    return matrix_state


def measure(state):
    """
    Does a single measurement of the quantum state (collapses state) and plots result  
    """
    #Convert state to a list
    state = list(state.matrix) 

    # Calculate the probabilities of each outcome based on the quantum state
    probabilities = [abs(coeff)**2 for coeff in state]

    # Perform the measurement probabilistically
    outcome_index = random.choices(range(len(state)), probabilities)[0]

    # Prepare and return the measurement outcome
    result = [0] * len(state)
    result[outcome_index] = 1

    figsize = (8, 4)

    eigenvectors = [('Ψ' + (('0' * (int(
        (np.log2(4))) - len(str(bin(i))[2:]))) + str(bin(i))[2:]))
                    for i in range(len(result))]
    #eigenvectors = [('Ψ' + str(i)) for i in range(len(result))] #labels as decimal

    plt.figure(figsize=figsize)
    plt.grid()
    plt.ylim(0, 1)
    plt.bar(eigenvectors, result)
    plt.xlabel("States")
    plt.ylabel("Amplitude")
    plt.title('Measurement Result')
    plt.show()


# grovers algorithm for n-qubit quantum register
def general_grovers_algorithm():

    H = Matrix((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]))

    n = int(input("Enter integer number of qubits for quantum register: "))
    N = 2**n  # no of basis states
    N_list = [i for i in range(N)
              ]  # list of possible "state to find" given n qubit register

    target_state_int = int(
        input(
            f"Choose the target state you want Grovers algorithm to find (integer in the range {N_list[0]}-{N_list[-1]}): "
        ))

    n_q_state = state(n, 0)  # initialised to zero
    target_state = state(n, target_state_int)  # initialised to target state

    H_n = nxn_matrix_gate(H, n)  # NxN Hadamard gate

    # apply H to all qubits in quantum register (puts the n qubit state into a superposition of all its basis states)
    n_q_state = apply_nxn_gate_to_qr(H_n, n_q_state)

    I_n = identity(N)  # NxN identity Matrix object

    # N = n**2 usually it takes sqrt(N) operations to find target state however for the 3-qubit system it
    # takes N operations to find the target state (or to have a larger final amplitude in the final states)
    # also notice that for the 1-qubit system it always return equal amplitude for both possible states :/

    if n <= 4:
        no_of_iterations = int(N)
    else:
        no_of_iterations = int(np.sqrt(N))
    # print(no_of_iterations)

    # apply oracle and diffusion no_of_iterations times
    for i in range(no_of_iterations):
        initial_state = n_q_state  # store initial n qubit state

        # oracle O = I_n - 2|target_state><target_state|
        O = I_n - (target_state * (target_state.transpose())).scalar(2)

        # apply the oracle to the state (reflecting the state about the axis perpendicular to target state)
        n_q_state = O * n_q_state

        # diffuser D = 1 - 2|initial state><initial state|
        D = I_n - (initial_state * (initial_state.transpose())).scalar(2)

        # apply the diffuser to the state (reflecting the state about the initial state)
        n_q_state = D * n_q_state

    print(f"Final state of the quantum register: \n {n_q_state}")
  
    measure(n_q_state)  


general_grovers_algorithm()
