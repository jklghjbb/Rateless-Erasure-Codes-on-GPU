import numpy as np
import networkx
#must Know symbol size T and numnber K of symbols
from V_table import PACKET_SIZE #symbol size T
from time import time

#def exchange rows: can be 
def excahnge_rows(A,i,j):
    for k in range(len(A[0])):
        swap = A[i][k]
        A[i][k] = A[j][k]
        A[j][k] = swap

def exchange_columns(A,i,j):
    for k in range(len(A)):
        swap = A[k][i]
        A[k][i] = A[k][j]
        A[k][j] = swap

#def exchange degree array for each
def exchange_degrees(degree,i,j):
    temp = degree[i]
    degree[i] = degree[j]
    degree[j] = temp
#then choose any row with exactly 2 ones in V that is part of a maximum size component in the graph defined by V
def row_in_graph(A,i,u,rows,L): #use networkx to construct graph
    #calculate maximal subgraph in 
    graph = networkx.Graph()
    for row in rows:
        vertices = []
        for vertice in range(i,L-u):
            if A[row][vertice] == 1:
                vertices.append(vertice)
        v1,v2 = tuple(vertices)
        graph.add_edge(v1,v2,row_index = row)
    
    #Componets in this graph: full collected
    components = list(networkx.connected_components(graph))
   
    # Find the max component
    maximal_component = max(components,key=len)
   # print(maximal_component)
    for row in rows:
        vertices = []
        for vertice in range(i,L-u):
            if A[row][vertice] == 1:
                vertices.append(vertice)
        if set(vertices).issubset(maximal_component):
            return row
   
#Choose a row with r ones in V with minimum original degree
def min_degree_row(m,degree,rows):
    min_d = m+1
    min_r = None
    for row in rows:
        if degree[row] < min_d:
            min_d = degree[row]
            min_r = row
    return min_r 


#min r One row of A has exactly r ones in V: avoid array creation: avoid extra complexity
def r_min(A,i,u,L):
    min_r = None
    rows = []   #need to be edited in GPU solution
    for k in range(i,len(A)): #O(n) #control row 
        # the number of 1s in this row 
        r = np.sum(A[k][i:L-u] == 1) #in kernal mode, this could be implemented   
        if r!=0:
            if min_r == None or r < min_r:
                min_r = r
                rows = [k]
            elif r==min_r:
                rows.append(k)
    return min_r,rows

#select indexes of 1 in a given row i
def choose_ones(A,i,u,ones,L):
    ones_index = 0
    for column in range(i, L - u):
        if A[i][column] == 1:
            ones[ones_index] = column
            ones_index = ones_index + 1

#xor rwo rows: xor i row into j
def xor_rows(A,i,j):
    A[j] = A[j] ^ A[i]

#one dimensional array element swap:
def swap_cd(d,i,j):
    d[[i,j]] = d[[j,i]] 


#first phase
def decoding_schedule(A,L,D):
    i = 0
    u = 0
    #initial V is A
    V = A
    m = len(A)
    #initial c, d
    c = np.array(range(L))
    d = np.array(range(m))

    degree = np.zeros(m,dtype =np.int32)
    #initial degree array
    for j in range(m):
        degree[j] = np.sum(A[i] == 1)
    
    #choose which row swap
    while i+u<L:
        #choose rows to swap
        r,rows = r_min(A,i,u,L)
        #print(r)
        if r==2:
            #choose maximun size component in graph
            #row = row_in_graph(A,i,u,rows,L)
            row = min_degree_row(m,degree,rows)
        else:
            #choose rows with r ones with 
            row = min_degree_row(m,degree,rows)
        
        #exchange rows & exchange degrees
        excahnge_rows(A,i,row)  #schedule
        exchange_degrees(degree,i,row)
        swap_cd(d,i,row)

        #reorder one column in the first Tips: the chosen row is the first row of V:i

        #find which column to exchange
        ones = np.zeros(r,dtype = np.int32) #从左到右
        choose_ones(A,i,u,ones,L)
        
        #rearrange columns
        for j in range(len(ones)):
            if j == 0:               
                exchange_columns(A,ones[0],i)
                swap_cd(c,ones[0],i)    #schedule       
            else:               
                exchange_columns(A,L-u-j,ones[len(ones)-j])
                swap_cd(c,L-u-j,ones[len(ones)-j])  #schedule
    
        #xor the chosen rows into other rows
        for row in range(i + 1, m):
                if A[row][i]:
                    xor_rows(A, i, row) 
                    #schedule
                    xor_rows(D,d[i],d[row])
        #update i and u
        i += 1
        u += r - 1
        

   # Divide U into U_upper with i*u matirix, and (m-i) * u lower
   # U_lower i -  I+u is Id
    for col in range(L-u, L):
        if A[col][col] != 1:
            print(col)
            # find a row to swap
            for row in range(col + 1, m):
                if A[row][col] == 1:
                    # swap rows
                    excahnge_rows(A, col, row)
                    exchange_degrees(degree,col,row) #schedule for D
                    swap_cd(d,col,row)
                    break
                #to determine  if row exhanged 
           
            
            if  A[col][col] != 1:
                 raise Exception("Canot decode")
                
                
                #xor col to each row below colomun
        for row in range(col + 1, m):
                if A[row][col]:
                    xor_rows(A, col, row)
                    #schedule
                    xor_rows(D,d[col],d[row])

    #construct U_upper
    for column in range(L - 1, L-u-1, -1):
        for row in range(i, column):
            if A[row][column] == 1:
                xor_rows(A, column, row)
                #schedule
                xor_rows(D,d[column],d[row])

    
    #construt L x L matrix
    A = A[:L]

    for row in range(i):
        for column in range(L-u, L):
            if A[row][column]:
                    xor_rows(A, column, row)
                    xor_rows(D,d[column],d[row])
     
    return c,d

now = time()
random_array = np.zeros((30000,30000))
excahnge_rows(random_array,0,18,) 
print(time()-now)
#Calculate L = K + S + H

#Triple is also passed to the decoder

#Determine the Rank of 