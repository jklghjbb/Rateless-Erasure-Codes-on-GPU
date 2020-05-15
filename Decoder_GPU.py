import numpy as np
import networkx
from math import ceil
from numba import cuda
#must Know symbol size T and numnber K of symbols
from V_table import PACKET_SIZE #symbol size T
from time import time

#def exchange rows: can be 
@cuda.jit
def exchange_rows_GPU(A,i,j,L):
    Idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if Idx < L:
        swap = A[i][Idx]
        A[i][Idx] = A[j][Idx]
        A[j][Idx] = swap
@cuda.jit
def exchange_rows_GPU_1(A,i,row_device,L,m):
    j = row_device[0]
    Idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if Idx < L and j<m and j>i:
        swap = A[i][Idx]
        A[i][Idx] = A[j][Idx]
        A[j][Idx] = swap 



@cuda.jit
def xor_GPU(A_device,i,L,indices,num_threads):
    Idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if Idx < num_threads:
        for k in range(i,L):
            if A_device[i][k] != 0:
                A_device[indices[Idx]][k] = A_device[indices[Idx]][k] ^ A_device[i][k]

@cuda.jit
def xor_GPU_D_1(A_device,d_device,i,L,indices,num_threads):
    Idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    
    if Idx < num_threads:
        for k in range(i,L):
            if A_device[d_device[i]][k] != 0:
                A_device[d_device[indices[Idx]]][k] = A_device[d_device[indices[Idx]]][k] ^ A_device[d_device[i]][k]
@cuda.jit
def xor_GPU_D(A_device,i,L,indices,num_threads):
    Idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if Idx < num_threads:
        for k in range(0,L):
            if A_device[i][k] != 0:
                A_device[indices[Idx]][k] = A_device[indices[Idx]][k] ^ A_device[i][k]




#one dimensional array element swap:
def swap_cd(d,i,j):
    d[[i,j]] = d[[j,i]] 

@cuda.jit
def swap_cd_GPU(d,i,j_device):
    j = j_device[0]
    swap = d[i]
    d[i] = d[j]
    d[j] = swap



@cuda.jit
def collect_GPU(A_device,index,col,result,length):
    Idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if Idx < length:
        result[Idx] = A_device[Idx+index+1][col]

@cuda.jit
def collect_GPU_back(A_device,index,col,result,length):
    Idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if Idx < length:
        result[Idx] = A_device[Idx][col]

@cuda.jit
def find_nearest(A_device,i,m,row_device):
    if A_device[i][i] == 0:
        for col in range(i+1,m):
            if A_device[col][i] == 1:
                row_device[0] = col               
                break
    else:
        row_device[0] = m + 1

# @cuda.jit
# def find_one(A_device,)

#first phase
def decoding_schedule_GPU_1(A,L,D):
    A_device = cuda.to_device(A)
    D_device = cuda.to_device(D)
    m = len(A) #n rows #m clomns
    c = np.array(range(L))
    d = np.array(range(m))
    for i in range(0,L): #n
        j=i
        #decide whether to perform row exchanging
        
        row_device = cuda.device_array(1,dtype=np.int32)
        find_nearest[1,1](A_device,i,m,row_device)
     
        #swap rows
        threads_per_block_L = 256
        blocks_per_grid_L = ceil(L/threads_per_block_L)
        exchange_rows_GPU_1[blocks_per_grid_L,threads_per_block_L](A_device,i,row_device,L,m)
        #swap_cd(d,i,row)
        threads_per_block_m = 256
        blocks_per_grid_m = ceil((PACKET_SIZE/4)/threads_per_block_m)
        exchange_rows_GPU_1[blocks_per_grid_m,threads_per_block_m](D_device,i,row_device,PACKET_SIZE/4,m)
            
        
   
        
        #collect elements from
        collected = np.zeros(m-i-1) #each ent-ry is the number of a entry in j
        collected_device = cuda.to_device(collected)
        threads_per_block_i = 1024
        if m-i-1 !=0:
            blocks_per_grid_i= ceil((m-i-1)/threads_per_block_i)
        else:
            blocks_per_grid_i = 1
        collect_GPU[blocks_per_grid_i,threads_per_block_i](A_device,i,i,collected_device,m-i-1)
        collected_device.copy_to_host(collected)


      
        #calulate which row for forward reduction
        indices = []
        for row in range(len(collected)):
            if collected[row] == 1:
                indices.append(row+i+1)
        indices_array = np.array(indices,dtype=np.int32)

        #transfer indices to GPU memory
        indices_device = cuda.to_device(indices_array)      
        threads_per_block_th = 1024
        num_threads = len(indices)
        if len(indices_array) !=0:
            blocks_per_grid_th= ceil(len(indices_array)/threads_per_block_i)
        else:
            blocks_per_grid_th = 1

        #d_device = cuda.to_device(d)
        xor_GPU[blocks_per_grid_th,threads_per_block_th](A_device,i,L,indices_device,num_threads)
        xor_GPU_D[blocks_per_grid_th,threads_per_block_th](D_device,i,len(D[0]),indices_device,num_threads)
        #A_device.copy_to_host(A)

   
    #backward subsitution
    for i in range(L-1,0,-1):
        #collect element
        collected = np.zeros(i) #each ent-ry is the number of a entry in j      
        collected_device = cuda.to_device(collected)
        
        threads_per_block_i = 256
        if i !=0:
            blocks_per_grid_i= ceil(i/threads_per_block_i)
        else:
            blocks_per_grid_i = 1
        collect_GPU_back[blocks_per_grid_i,threads_per_block_i](A_device,0,i,collected_device,i)
        collected_device.copy_to_host(collected)

        
        #collect elements
        indices = []
        for row in range(len(collected)):
            if collected[row] == 1:
                indices.append(row)       
        indices_array = np.array(indices)
 
        #back ward subsistution
        indices_device = cuda.to_device(indices_array)      
        threads_per_block_th = 1024
        num_threads = len(indices)
        if len(indices_array) !=0:
            blocks_per_grid_th= ceil(len(indices_array)/threads_per_block_i)
            xor_GPU[blocks_per_grid_th,threads_per_block_th](A_device,i,L,indices_device,num_threads)
            xor_GPU_D[blocks_per_grid_th,threads_per_block_th](D_device,i,len(D[0]),indices_device,num_threads)
            #A_device.copy_to_host(A)
        else:
            blocks_per_grid_th = 1

    D_device.copy_to_host(D)
    A_device.copy_to_host(A)
    print(A)
    return c,d


def decoding_schedule_GPU_2(A,L,D):
    A_device = cuda.to_device(A)
    D_device = cuda.to_device(D)
    m = len(A) #n rows #m clomns
    c = np.array(range(L))
    d = np.array(range(m))
    d_device = cuda.to_device(d)
    for i in range(0,L): #n
        j=i
        #decide whether to perform row exchanging
        
        row_device = cuda.device_array(1,dtype=np.int32)
        #row = 0
        
        # if A[i][i] == 0:
        #     for col in range(j+1,m):
        #         if A[col][i] == 1:
        #             row = col
        #             break
        find_nearest[1,1](A_device,i,m,row_device)
           
        
       
        #swap rows
       # print(i)
        # if  i<row and row < m:
        #     #excahnge_rows(A,i,row)
        #     threads_per_block_L = 256
        #     blocks_per_grid_L = ceil(L/threads_per_block_L)
        #     exchange_rows_GPU[blocks_per_grid_L,threads_per_block_L](A_device,i,row,L)
        #     swap_cd(d,i,row)

        threads_per_block_L = 256
        blocks_per_grid_L = ceil(L/threads_per_block_L)
        exchange_rows_GPU_1[blocks_per_grid_L,threads_per_block_L](A_device,i,row_device,L,m)
        #swap_cd(d,i,row)
        threads_per_block_m = 256
        blocks_per_grid_m = ceil((PACKET_SIZE/4)/threads_per_block_m)
        #exchange_rows_GPU_1[blocks_per_grid_m,threads_per_block_m](D_device,i,row_device,PACKET_SIZE/4,m)
        swap_cd_GPU(d_device,i,row_device)
            
        
        # A_device.copy_to_host(A)

        # if A[i][j] !=1 :
        #     raise  Exception("Canot decode")
        
        #collect elements from
        collected = np.zeros(m-i-1) #each ent-ry is the number of a entry in j
        collected_device = cuda.to_device(collected)
        threads_per_block_i = 1024
        if m-i-1 !=0:
            blocks_per_grid_i= ceil((m-i-1)/threads_per_block_i)
        else:
            blocks_per_grid_i = 1
        collect_GPU[blocks_per_grid_i,threads_per_block_i](A_device,i,i,collected_device,m-i-1)
        collected_device.copy_to_host(collected)


        #print(collected)
        #calulate which row for forward reduction
        indices = []
        for row in range(len(collected)):
            if collected[row] == 1:
                indices.append(row+i+1)
        indices_array = np.array(indices,dtype=np.int32)

        #transfer indices to GPU memory
        indices_device = cuda.to_device(indices_array)      
        threads_per_block_th = 1024
        num_threads = len(indices)
        if len(indices_array) !=0:
            blocks_per_grid_th= ceil(len(indices_array)/threads_per_block_i)
        else:
            blocks_per_grid_th = 1
        
        #perform xor
        #d_device = cuda.to_device(d)
        xor_GPU[blocks_per_grid_th,threads_per_block_th](A_device,i,L,indices_device,num_threads)
        xor_GPU_D_1[blocks_per_grid_th,threads_per_block_th](D_device,d_device,i,len(D[0]),indices_device,num_threads)
        #A_device.copy_to_host(A)


    #backward subsitution

    for i in range(L-1,0,-1):
        #collect element
        collected = np.zeros(i) #each ent-ry is the number of a entry in j      
        collected_device = cuda.to_device(collected)
        
        threads_per_block_i = 256
        if i !=0:
            blocks_per_grid_i= ceil(i/threads_per_block_i)
        else:
            blocks_per_grid_i = 1
        collect_GPU_back[blocks_per_grid_i,threads_per_block_i](A_device,0,i,collected_device,i)
        collected_device.copy_to_host(collected)

        
        # for row in range (0, i): #row is the idex         
        #     collected[row] = A[row][i]
        # print(collected)

        indices = []
        for row in range(len(collected)):
            if collected[row] == 1:
                indices.append(row)       
        indices_array = np.array(indices)
        # indices_device = 
        # if indices:
        #     for row in indices_array:
        #         xor_rows(A,i,row)
        #         xor_rows(D,d[i],d[row])

        # A_device.copy_to_host(A)

        indices_device = cuda.to_device(indices_array)      
        threads_per_block_th = 1024
        num_threads = len(indices)
        if len(indices_array) !=0:
            blocks_per_grid_th= ceil(len(indices_array)/threads_per_block_i)
            xor_GPU[blocks_per_grid_th,threads_per_block_th](A_device,i,L,indices_device,num_threads)
            xor_GPU_D_1[blocks_per_grid_th,threads_per_block_th](D_device,d_device,i,len(D[0]),indices_device,num_threads)
            #A_device.copy_to_host(A)
        else:
            blocks_per_grid_th = 1

    D_device.copy_to_host(D)
    A_device.copy_to_host(A)
    d_device.copy_to_host(d)
    print(A)
    return c,d




            





