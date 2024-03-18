from simulator import Sparse, SparseRep, Dense, state, H

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
    assert (target_state_int in N_list)
    t1 = time.time()
    n_q_state = state(n, 0)  # initialised to zero
    target_state = state(n, target_state_int)  # initialised to target state

    H_n = nxn_matrix_gate(H, n)  # NxN Hadamard gate

    # apply H to all qubits in quantum register (puts the n qubit state into a superposition of all its basis states)
    n_q_state = apply_nxn_gate_to_qr(H_n, n_q_state)

    I_n = identity(N)  # NxN identity Matrix object

    no_of_iterations = int(np.pi / 4 * np.sqrt(N))

    initial_state = n_q_state  # store initial n qubit state

    # oracle O = I_n - 2|target_state><target_state|
    O = I_n - (target_state * (target_state.transpose())).scalar(2)

    # apply oracle and diffusion no_of_iterations times
    for _ in range(no_of_iterations):

        # apply the oracle to state: reflects state about axis perp. to target state
        n_q_state = O * n_q_state

        # diffuser D = 1 - 2|initial state><initial state|
        D = I_n - (initial_state * (initial_state.transpose())).scalar(2)

        # apply the diffuser to the state (reflecting the state about the initial state)
        n_q_state = D * n_q_state

    print(f"Final state of the quantum register: \n {n_q_state}")
    t2 = time.time()
    print(t2 - t1)
    measure(n_q_state)

def nxn_matrix_gate(matrix_gate, n):
    """
                NxN Matrix gate given a Matrix gate and the number of qubits n
        """
    assert (type(matrix_gate) == Dense or type(matrix_gate)
            == Sparse), "The gate inputted must be a Matrix object"
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
    assert (type(matrix_state) == Dense or type(matrix_state)
            == Sparse), "The state inputted must be a Matrix object"
    assert (type(nxn_gate) == Dense or type(nxn_gate)
            == Sparse), "The gate inputted must be an NxN Matrix object"

    matrix_state = nxn_gate * matrix_state

    return matrix_state