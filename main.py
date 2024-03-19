from quantum_simulator import *

def grovers_algorithm():

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

def shors_algorithm(number):
    # https://arxiv.org/pdf/1804.03719.pdf
    # https://arxiv.org/pdf/1903.00768.pdf
    # https://www.science.org/doi/epdf/10.1126/science.aad9480
    # https://arxiv.org/pdf/1207.0511.pdf
    # https://arxiv.org/pdf/1706.08061.pdf
    # https://arxiv.org/pdf/1611.07995.pdf
    # https://arxiv.org/pdf/quant-ph/0205095.pdf
    # https://cs.stackexchange.com/questions/72439/matrix-for-modular-exponentiation-in-shors-algorithm
    # https://bobbycorpus.wordpress.com/2019/10/28/constructing-the-quantum-fourier-circuit/
    # https://github.com/mett29/Shor-s-Algorithm/blob/master/Shor.ipynb
    # https://www.science.org/doi/10.1126/science.aad9480
    # there is a method using basically an oracle gate, but this is incredibly stupid because its basically
    # calculating the modular exponentiation first then converting it into a gate, which literally defeats
    # the whole purpose of shors algorithm because by calculating said oracle, u would already calculate
    # the period that shors would normally calculate. Its circular reasoning
    comp_qubit_count = len(bin(number-1)[2:])
    peri_qubit_count = 2*comp_qubit_count
    
    register = state(peri_qubit_count,0)
    shors = Programmer(register, name = "Shors algorithm")
    shors.add_step([H]*peri_qubit_count)
    print(shors)
    shors.compile()
    shors.run()
    shors.measure()

    
#shors_algorithm(15)
