#This file 

from V_table import V0, V1,Fj,Dj,Q,Jk
from math import floor,sqrt
from numba import cuda

# @cuda.jit(device=True)
# def Rand_GPU(V0_device,V1_deviceX, i, m):
#     return (V0_device[(X + i) % 256] ^ V1_device[(floor(X/256)+ i) % 256]) % m

def Rand(X, i, m):
    return (V0[(X + i) % 256] ^ V1[(floor(X/256)+ i) % 256]) % m

# @cuda.jit(device=True)
# def Deg_GPU(Fj_device,v):
#     for j in range(1,8):
#         if v >=Fj[j-1] and v < Fj[j]:
#             return Dj[j-1] 
def Deg(v):
    for j in range(1,8):
        if v >=Fj[j-1] and v < Fj[j]:
            return Dj[j-1] 
#Factorial
def Choose(N,K):
    if N==K:
        return 1
    elif K==1:
        return N
    else:
        return Choose(N-1,K-1)+Choose(N-1,K)

#Determine if the number is a prime
# @cuda.jit(device=True)
# def isPrime_GPU(num): 
#     if num == 2 or num == 3:  
#         return True
#     if num % 6 != 1 and num % 6 != 5:  
#         return False
#     tmp = int(sqrt(num))
#     for i in range(5, tmp+1):  
#         if num % i == 0 or num % (i+2) == 0:
#             return False
#     return True

def isPrime(num): 
    if num == 2 or num == 3:  
        return True
    if num % 6 != 1 and num % 6 != 5:  
        return False
    tmp = int(sqrt(num))
    for i in range(5, tmp+1):  
        if num % i == 0 or num % (i+2) == 0:
            return False
    return True  

#Find the Smallest prime >= L
# @cuda.jit(device=True)
# def findPrime_GPU(L):
#     i = L
#     while i>=L:
#         if isPrime_GPU(i) == True:
#             return i
#         i = i + 1

def findPrime(L):
    i = L
    while i>=L:
        if isPrime(i) == True:
            return i
        i = i + 1
        
# #generate Trip for encoding Symbols
# @cuda.jit(device=True)
# def Trip_GPU(Jk_device,V0_device,V1_device,Fj_device, K,X,L):
#     A = (53591 + Jk_device[K-4]*997) % Q
#     B = 10267*(Jk_device[K-4]+1) % Q
#     L_prime = findPrime_GPU(L)
#     Y = (B + X*A) % Q
#     v = Rand_GPU(V0_device,V1_device,Y, 0, 2**20)
#     d = Deg_GPU(Fj_device,v)
#     a = 1 + Rand_GPU(V0_device,V1_device,Y, 1, L_prime-1)
#     b = Rand_GPU(V0_device,V1_device,Y, 2, L_prime)
#     return d,a,b,L_prime

def Trip(K,X,L):
    A = (53591 + Jk[K-4]*997) % Q
    B = 10267*(Jk[K-4]+1) % Q
    L_prime = findPrime(L)
    Y = (B + X*A) % Q
    v = Rand(Y, 0, 2**20)
    d = Deg(v)
    a = 1 + Rand(Y, 1, L_prime-1)
    b = Rand(Y, 2, L_prime)
    return d,a,b,L_prime

print(Trip(10,8,5200))
print(findPrime(13))

