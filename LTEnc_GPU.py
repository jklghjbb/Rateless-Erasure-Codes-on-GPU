#generate a L greater than K
#L: number of intermidiate symbols
#K; number of source symbols
#pre-code relationship
from PRNG import findPrime,Choose,Trip
from V_table import V0, V1,Fj,Dj,Q,Jk
from math import ceil,floor
import numpy as np
from time import time
from numba import cuda
from V_table import PACKET_SIZE

#Compute S and H
def computeSH(K):
    #find X
    X = 0
    while X >=0:
        if X*(X-1) >= 2*K:
            break
        X = X + 1
    #find S
    S = findPrime(ceil(0.01*K)+X)
    #find H
    H = 1
    while H >= 1:
        if Choose(H,ceil(H/2)) >= K + S:
            break
        H = H + 1
    return S,H
    #find S

#count ones in a given integer
def count_ones(num):
    count = 0
    mask = 1
    for i in range(16):
        if (mask << i) & num:
            count = count + 1
    return count

#generate sequence of gray sequence
def gray(num):
    return num ^ floor(num/2)

#generate subsequence of gray sequence 这里用了list 注意
def Mk(H_prime):
    Mk_list = []
    i = 0
    for i in range(65536):
        num = gray(i)
        if count_ones(num) == H_prime:
            Mk_list.append(num)
    return Mk_list

@cuda.jit
def G_LT_GPU(A_device,triples,K,L,S,H):
    Idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if Idx < K:
        d = triples[Idx][0]
        a = triples[Idx][1]
        b = triples[Idx][2]
        L_prime = triples[Idx][3]
        #print(d)
        while b>=L:
            b = (b + a)%L_prime
        A_device[S+H+Idx][b] = 1
        for j in range(1,min(d,L)):
            b = (b + a)%L_prime
            while b>=L:
                b = (b + a)%L_prime
            #result = result ^ C[b]
            A_device[S+H+Idx][b] = 1

# @cuda.jit
# def G_LT_GPU_1(Jk_device,V0_device,V1_device,Fj_device,A_device,K,L,S,H):
#     Idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
#     d,a,b,L_prime = Trip_GPU(Jk_device,V0_device,V1_device,Fj_device,K,Idx,L)
#     if Idx < K:
#         #print(d)
#         while b>=L:
#             b = (b + a)%L_prime
#         A_device[S+H+Idx][b] = 1
#         for j in range(1,min(d,L)):
#             b = (b + a)%L_prime
#             while b>=L:
#                 b = (b + a)%L_prime
#             #result = result ^ C[b]
#             A_device[S+H+Idx][b] = 1

#Generate Generator Matrix A
def GeneratorMatrix_GPU(K):
    S,H = computeSH(K)
    H_prime = ceil(H/2)
    L = K + S + H
    #Output generator array
    A = np.zeros(shape = (K+S+H,K+S+H),dtype=np.int32)
    
    #G_LDPC
    for i in range(K):
        a = 1 + (floor(i/S) % (S-1))
        b = i % S
       #C[K + b] = np.bitwise_xor(C[K + b],C[i])
        A[b][i] = 1
        b = (b + a) % S
        #C[K + b] = np.bitwise_xor(C[K + b],C[i])
        A[b][i] = 1
        b = (b + a) % S
        #C[K + b] = np.bitwise_xor(C[K + b],C[i])
        A[b][i] = 1   
    #I_S
    for i in range(S):
        A[i][K+i] = 1
    #construct G_Half
    M = Mk(H_prime)
    mask = 1
    for h in range(H):
        for j in range(K+S):
            #if bit h of 
            if (mask << h) & M[j]:
                #C[h+K+S]=np.bitwise_xor(C[h+K+S],C[j])
                A[S+h][j] = 1
    #compute I_H
    for i in range(H):
        A[S+i][K+S+i]=1

  


   
    threads_per_block = 128
    blocks_per_grid = ceil(K/threads_per_block)
    
    start_all = time()
    #pre-compute the triples
    triples = np.zeros(shape =(K,4),dtype=np.int32)
    for i in range(K):
        d,a,b,L_prime = Trip(K,i,L)
        triples[i][0] = d
        triples[i][1] = a
        triples[i][2] = b
        triples[i][3] = L_prime

    #CPU precompute the Trip array
    start_transfer = time()
    triples_device = cuda.to_device(triples)
    A_device = cuda.to_device(A) #transfer A to GPU memory
    # V0_device = cuda.to_device(V0)
    # V1_device = cuda.to_device(V1)
    # Jk_device = cuda.to_device(Jk)
    # Fj_device = cuda.to_device(Fj)
    print("Transfer time " + str(time() - start_transfer))


    #compute G_LT LT encoding
    start_compute = time()
    #G_LT_GPU[blocks_per_grid,threads_per_block](A_device,triples_device,K,L,S,H)
    G_LT_GPU[blocks_per_grid,threads_per_block](A_device,triples_device,K,L,S,H)
    cuda.synchronize()
    print("Compute time " + str(time() - start_compute))
    #back
    start_back = time()
    A = A_device.copy_to_host()
    print("Transfer back time " + str(time() - start_back))
    print("G_LT time " + str(time() - start_all))
    return A,L





#Compute th