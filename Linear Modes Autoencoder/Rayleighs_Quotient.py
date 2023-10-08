import numpy as np

def rayleigh_quotient(A, V):

    # Calculate the Rayleigh quotient
    eigenvalue_estimate = (V.T @ A @ V) / (V.T @ V)

    return eigenvalue_estimate

A = np.array([[2, -1],
              [-1, 2]])

V = np.array([[0.602, 0.602],
              [0.600, -0.601]])

for i in range(int(V.shape[0])):

    print("Eigenvalue",i,":", rayleigh_quotient(A, V[:, i]))
    print("Eigenfrequency",i,":", np.sqrt(rayleigh_quotient(A, V[:, i])))
    print("\n")

