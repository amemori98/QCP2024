#main is for calling simulator and implementing the algorithms and visualisation (if any)
#need to import other files so can call classes etc.

from simulator import Matrix, Dense, Sparse
import numpy as np
import random
from matplotlib import pyplot as plt


# -------------------------------------------------------------------------------------

# general Grover's algorithm for n qubit quantum register
# uses the functions defined below (a state function, nxn matrix gate function, and a apply gate function) NOT the state class

# have also noticed that for a 3-qubit system it doesn't seem to work well... with the final state having less amplitude in the target state than all the rest so maybe we should increase the no of iterations the oracle and diffuser are applied to the 3-qubit system!

H = Sparse((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]))



def state(n, m):
    # fixed so it properly initializes the register and uses Sparse
    """
		Nx1 Matrix state of the quantum register initialised to state corresponding to qubit m
	"""
    assert (type(n) == int), "The number of qubits n inputted must be an integer"
    assert (type(m) == int), "The qubit to which quantum register is intialised m must be an integer"
    assert (m >= 0 and m < 2**n), "m must be between 0 and 2^n"

    # initialise register to all zeros
    state = np.zeros((2**n, 1), dtype=complex)
    
    # initialize to state |m>
    state[m] = 1
        
    return Sparse(state)


def nxn_matrix_gate(matrix_gate, n):
    """
        NxN Matrix gate given a Matrix gate and the number of qubits n
    """
    assert (type(matrix_gate) == Dense or type(matrix_gate) == Sparse
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
    return Dense(np.array(I))


def apply_nxn_gate_to_qr(nxn_gate, matrix_state):
    # what is the point of this? pretty sure if the types are not right, it will throw an error regardlessww
    """
		Applies an NxN gate to a Matrix state
    """
    assert (type(matrix_state) == Dense or type(matrix_state) == Sparse
            ), "The state inputted must be a Matrix object"
    assert (type(nxn_gate) == Dense or type(nxn_gate) == Sparse
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

    n = int(input("Enter integer number of qubits for quantum register: "))
    N = 2**n  # no of basis states
    N_list = [i for i in range(N)
              ]  # list of possible "state to find" given n qubit register

    target_state_int = int(
        input(
            f"Choose the target state you want Grovers algorithm to find (integer in the range {N_list[0]}-{N_list[-1]}): "
        ))
    assert (
        target_state_int in N_list)

    n_q_state = state(n, 0)  # initialised to zero
    target_state = state(n, target_state_int)  # initialised to target state

    H_n = nxn_matrix_gate(H, n)  # NxN Hadamard gate

    # apply H to all qubits in quantum register (puts the n qubit state into a superposition of all its basis states)
    n_q_state = apply_nxn_gate_to_qr(H_n, n_q_state)

    I_n = identity(N)  # NxN identity Matrix object

    # N = n**2 usually it takes sqrt(N) operations to find target state however for the 3-qubit system it
    # takes N operations to find the target state (or to have a larger final amplitude in the final states)
    # also notice that for the 1-qubit system it always return equal amplitude for both possible states :/

    no_of_iterations = int(N) if n == 3 else int(np.sqrt(N))
    # oracle O = I_n - 2|target_state><target_state|
    O = I_n - (target_state * (target_state.transpose())).scalar(2)
    
    # apply oracle and diffusion no_of_iterations times
    for i in range(no_of_iterations):
        initial_state = n_q_state  # store initial n qubit state

        # apply the oracle to state: reflects state about axis perp. to target state
        n_q_state = O * n_q_state

        # diffuser D = 1 - 2|initial state><initial state|
        D = I_n - (initial_state * (initial_state.transpose())).scalar(2)

        # apply the diffuser to the state (reflecting the state about the initial state)
        n_q_state = D * n_q_state

    print(f"Final state of the quantum register: \n {n_q_state}")
  
    measure(n_q_state)

general_grovers_algorithm()

#print(state(4,2))
#print(H)