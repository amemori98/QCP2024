

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
    zero = Dense(np.array([[1], [0]]))  # MATRIX class
    one = Dense(np.array([[0], [1]]))

    # gates
    # I = Sparse(np.array([[1, 0], [0, 1]]))  # Identity
    # H = Sparse((1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]))  # Hadamard

    I = Sparse(np.array([[1, 0], [0, 1]]))  # Identity MATRIX class
    H = Dense((1 / np.sqrt(2)) *
               np.array([[1, 1], [1, -1]]))  # Hadamard MATRIX class

    # not used gates
    X = Sparse(np.array([[0, 1], [1, 0]]))  # Pauli X
    Z = Sparse(np.array([[1, 0], [0, -1]]))  # Pauli Z
    CNOT = Sparse(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                  [0, 0, 1, 0]]))  # Controlled-NOT

    CZ = Sparse(
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