#main is for calling simulator and implementing the algorithms and visualisation (if any)
#need to import other files so can call classes etc.
#use for testing

from gates import Gates

#testing

psi = Gates(2)  #initialise 2 qubit register
print(psi)  #print initial state

h = psi.apply_hadamard(
    psi.state[0])  #psi.state[0] is qubit 0, can make simpler
print(psi)
print(h)
#NOTE should update so psi=h when gate applied, easier to apply next gate
hh = psi.apply_hadamard(psi.state[0]).apply_hadamard(psi.state[0])
#applying twice doesnt work since apply_hadamard returns an array not the object

#Grover's algorithm
#Printing and visualisations
