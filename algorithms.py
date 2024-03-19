from quantum_simulator import *

def grovers_algorithm():
    """
    Uses quantum simulator program to implement Grover's algorithm on n qubits for a given target state
    """

    n = int(input("Enter integer number of qubits for quantum register: "))
    N = 2**n  # no of basis states

    # list of possible "state to find" given n qubit register
    N_list = [i for i in range(N)]

    target_state_int = int(
        input(f"Choose the target state you want Grovers algorithm to find (integer in the range {N_list[0]}-{N_list[-1]}): "))
    assert (target_state_int in N_list)

    n_q_state = state(n, 0)  # initialised to zero

    q = Programmer(n_q_state, name=f"Grovers {n} Qubits")

    q.add_step([H]*n)
    q.compile()
    q.run()
    initial_state = q.output

    target_state = state(n, target_state_int)  # initialised to target state
    I_n = identity(N)  # NxN identity Matrix object

    Oracle = I_n  - (target_state * (target_state.transpose()).scalar(2))
    Oracle.id = "Oracle"

    Diffuser = (initial_state * (initial_state.transpose()).scalar(2)) - I_n
    Diffuser.id = "Diffuser"

    no_of_iterations = int(np.pi / 4 * np.sqrt(N))
    for i in range(no_of_iterations):
        q.add_step([Oracle])
        q.add_step([Diffuser])

    q.compile()
    q.run()

    
    q.measure()
    print(q)
    print(f"Final state of the quantum register: \n {q.output}")

# the function below attempts the implementation of Shor's algorithm. However at the moment it DOES NOT WORK.
def shors_algorithm(number):
    comp_qubit_count = len(bin(number-1)[2:])
    peri_qubit_count = 2*comp_qubit_count
    
    register = state(peri_qubit_count,0)
    shors = Programmer(register, name = "Shors algorithm")
    shors.add_step([H]*peri_qubit_count)
    print(shors)
    shors.compile()
    shors.run()
    shors.measure()
