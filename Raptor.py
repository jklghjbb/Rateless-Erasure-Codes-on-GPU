from V_table import PACKET_SIZE
import numpy as np
import os
from math import ceil
from PRNG import Trip
from LTEnc import GeneratorMatrix
from LTEnc_GPU import GeneratorMatrix_GPU
from time import time
from Decoder import r_min,row_in_graph,decoding_schedule,choose_ones
from Decoder_GPU import decoding_schedule_GPU_1


isGPU_LT = False
isGPU = True
def read_source(file, filesize):

    K = ceil(filesize / PACKET_SIZE)
    C_prime  = np.zeros(shape = (K,int(PACKET_SIZE/4)),dtype=np.uint32)
    print(K)
    # Read data by blocks of size core.PACKET_SIZE
    for i in range(K):
            
        data = bytearray(file.read(PACKET_SIZE))
        if not data:
            raise "stop"

        if len(data) != PACKET_SIZE:
            data = data + bytearray(PACKET_SIZE - len(data))
        # Paquets are condensed in the right array type
        C_prime[i] = np.frombuffer(data, dtype=np.uint32)
    #K is the number of source symbols
    return C_prime,K

filename="cw1.zip"
with open(filename, "rb") as file:

        filesize = os.path.getsize(filename)
        C_prime,K = read_source(file, filesize)
        start = time()
        if  isGPU_LT == False:
            A,L = GeneratorMatrix(K)
        else:
            A,L = GeneratorMatrix_GPU(K)
        print("time " + str(time() - start))
        print("number of symbols: "+str(K))
        print("number of rows: " + str(L))
        D = np.zeros(shape = (L,int(PACKET_SIZE/4)),dtype=np.uint32) #the constraint matrix
        for i in range(L):
            if i < L-K:
                D[i] = np.zeros(int(PACKET_SIZE/4))
            else:
                D[i] = C_prime[i-L+K]

        #print(np.linalg.matrix_rank(A))
        
        start = time()
        if isGPU == False:
            c,d = decoding_schedule(A,L,D)
        else:
            #c,d = decoding_schedule_GPU_1(A,L,D)
            c,d = decoding_schedule_GPU_1(A,L,D)
        print("Decode time " + str(time() - start))
       # print(A)


        C_inter = np.zeros(shape = (L,int(PACKET_SIZE/4)),dtype=np.uint32)
        for i in range(L):
            C_inter[c[i]] = D[d[i]]
 
        C_source = np.zeros(shape = (K,int(PACKET_SIZE/4)),dtype=np.uint32)
        #construct the encoding symbols
        A,L = GeneratorMatrix(K)
        G_LT = A[L-K:]
        for i in range(K):
            d,a,b,L_prime= Trip(K,i,L)
            while b>=L:
                b = (b + a)%L_prime
            C_source[i] = C_source[i] ^ C_inter[b]
            for j in range(1,min(d,L)):
                b = (b + a)%L_prime
                while b>=L:
                    b = (b + a)%L_prime
            #result = result ^ C[b]
                C_source[i] = C_source[i] ^ C_inter[b]

        #determine if decoded succussfully
        print((C_source == C_prime))
        

        #recover the soure symbol emulate

        #print(np.linalg.matrix_rank(A))
        # A1 = A[380:416,380:416]
        # print(A1)
        # print(A[0:380,0:380])
        # print(np.linalg.matrix_rank(A1))


        #use encoding 

       # print(D)

        