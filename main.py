#main is for calling simulator and implementing the algorithms and visualisation (if any)
#need to import other files so can call classes etc.

from gates import State


#Grover's algorithm
def grovers_algorithm(n_qubits):

    #set up intial state
    psi = State(n_qubits)

    #apply hadamard gate to each qubit simultaneously
    psi.apply_hadamard_all()

    #apply the oracle

    #apply the diffusion

    #measure


#test
#psi = State(2)  #initialise 2 qubit register
#print(psi)  #print initial state
#grovers_algorithm(2)

psi = State(2)
print(psi)
psi.apply_hadamard_all()
print(psi.state)
