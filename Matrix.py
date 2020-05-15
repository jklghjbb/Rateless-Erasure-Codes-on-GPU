import numpy as np
def identity(size):
    I = np.zeros(shape = (size,size),dtype=np.uint32)
    for i in range(size):
        I[i][i] = 1
    return I

#inverse a matrix on GF(2)
def inverse(A,L):
    #phase 1:
    V = A
    I = identity(L)
    for i range(L):
        V[i] = V[i] +  

    
    return 

def isFull(A):
    return

print(identity(10))