#################################################################
# Name:     SolveLinear.py                                      #
# Authors:  Michael Battaglia                                   #                                        #
# Function: Program contains routines for solving linear        #
#           systems of equations.                               #
#################################################################

#essential imports
import numpy as np
from time import process_time
import matplotlib.pyplot as plt

#function: Implement Gaussian Elimination
def GaussElim(A_in,v_in):
    #np.copy A and v to temporary variables using np.copy command
    A = np.copy(A_in)
    v = np.copy(v_in)
    N = len(v)

    #for each mth element in diagonal
    for m in range(N):
        # Divide by the diagonal element
        div = A[m,m]
        A[m,:] /= div
        v[m] /= div
    
        # Now subtract from the lower rows
        for i in range(m+1,N):
            mult = A[i,m]
            A[i,:] -= mult*A[m,:]
            v[i] -= mult*v[m]
    
    # Backsubstitution
    #create an array of the same type as the input array
    x = np.empty(N,dtype=v.dtype)
    for m in range(N-1,-1,-1):
        x[m] = v[m]
        for i in range(m+1,N):
            x[m] -= A[m,i]*x[i] 
    return x

#function: Implement Partial Pivot
def PartialPivot(A_in,v_in):
    #np.copy A and v to temporary variables using np.copy command
    A = np.copy(A_in)
    v = np.copy(v_in)
    N = len(v)

    #for each mth element in diagonal
    for m in range(N):
        # Partial pivot on mth column from A[m,m]
        col = A[:,m]
        row = m + np.argmax(np.square(col[m:]))
        # Swap row m with optimal row
        A[m,:], A[row,:] = np.copy(A[row,:]), np.copy(A[m,:])
        v[m], v[row] = v[row], v[m]
        
        # Divide by the diagonal element
        div = A[m,m]
        A[m,:] /= div
        v[m] /= div
    
        # Now subtract from the lower rows
        for i in range(m+1,N):
            mult = A[i,m]
            A[i,:] -= mult*A[m,:]
            v[i] -= mult*v[m]
    
    # Backsubstitution
    #create an array of the same type as the input array
    x = np.empty(N,dtype=v.dtype)
    for m in range(N-1,-1,-1):
        x[m] = v[m]
        for i in range(m+1,N):
            x[m] -= A[m,i]*x[i] 
    return x

#function: Implement LU Decomposition
def LUdecomp(A_in):
    #Length of vector inputs
    N = A.shape[1]
    #A will eventually become U
    U = np.copy(A_in)
    #initiate L array
    L = np.zeros(U.shape)
    #initiate list to record swaps
    swaps = []

    #for each mth element in diagonal
    for m in range(N):
        #record state in L array
        L[:,m][m:] = U[:,m][m:]
        
        # Partial pivot on mth column from A[m,m]
        col = U[:,m]
        row = m + np.argmax(np.square(col[m:]))
        swaps.append(row)
        # Swap row m with optimal row
        U[m,:], U[row,:] = np.copy(U[row,:]), np.copy(U[m,:])
        L[m,:], L[row,:] = np.copy(L[row,:]), np.copy(L[m,:])
        
        # Divide by the diagonal element
        div = U[m,m]
        U[m,:] /= div
    
        # Now subtract from the lower rows
        for i in range(m+1,N):
            mult = U[i,m]
            U[i,:] -= mult*U[m,:]
    #Conlusion of LU decomposition
    return L, U, swaps

#function: Apply permutation operation on vector
def permute(op, v_in):
    #input vector to be permuted
    v = np.copy(v_in)
    N = len(v)
    #permute elements of v according to operator
    for m in range(N):
        v[m], v[op[m]] = v[op[m]], v[m]
    return v

#function: Implement LU Backsubstitution
def LUbacksub(L_in, U_in, v_in):
    #input vector (permuted accordingly)
    v = np.copy(v_in)
    N = len(v)
    #LU decomposition matrices
    L = np.copy(L_in)
    U = np.copy(U_in)
            
    # Backsubstitution for Ly = v
    #create an array of the same type as the input array
    y = np.empty(N,dtype=v.dtype)
    for m in range(N):
        y[m] = v[m]/L[m,m]
        for i in range(m):
            y[m] -= L[m,i]*y[i]/L[m,m]
            
    # Backsubstitution for Ux = y
    #create an array of the same type as the input array
    x = np.empty(N,dtype=y.dtype)
    for m in range(N-1,-1,-1):
        x[m] = y[m]/U[m,m]
        for i in range(m+1,N):
            x[m] -= U[m,i]*x[i]/U[m,m]
    return x

#function: main
if __name__ == '__main__':
    #simple diagnostic of linear equation solver
    A = np.array([[ 2, 1, 4, 1],
                  [ 3, 4,-1,-1],
                  [ 1,-4, 1, 5],
                  [ 2,-2, 1, 3]], float)
    v = np.array([-4, 3, 9, 7], float)
    print("matrix A")
    print(A)
    print("vector v")
    print(v)
    print("Solve Ax = v")
    #solve using gaussian elimination
    x_gaus = GaussElim(A,v)
    print("By gaussian elimination")
    print("x =",x_gaus)
    #solve using partial pivot
    x_ppiv = PartialPivot(A,v)
    print("By gaussian elimination with partial pivot")
    print("x =",x_ppiv)
    #solve using LU decomposition
    L, U, op = LUdecomp(A)
    x_LUdc = LUbacksub(L,U,permute(op,v))
    print("By LU decomposition with partial pivot")
    print("x =",x_LUdc)
    #check diagnostic success
    x_sol = np.array([2, -1, -2, 1], float)
    print("In reality")
    print("x =",x_sol)

    
    #advanced diagnostic of linear equation solver
    #trial N, sizes of matrices
    N_trials = np.arange(10,1000,10)
    #error as a function of N
    e_gaus = np.zeros(len(N_trials))
    e_ppiv = np.zeros(len(N_trials))
    e_LUdc = np.zeros(len(N_trials))
    #timing as a function of N
    t_gaus = np.zeros(len(N_trials))
    t_ppiv = np.zeros(len(N_trials))
    t_LUdc = np.zeros(len(N_trials))

    print("Solving "+str(len(N_trials))+" linear systems")
    #for each trial size N
    for i, N in enumerate(N_trials):
        #generate a random matrix
        A = np.random.rand(N,N)
        #perform QR factorization
        Q, R = np.linalg.qr(A)
        #random orthogonal (nonsingular) matrix
        A = Q
        #generate a random vector
        v = np.random.rand(N)

        print("Solving "+str(i+1)+"/"+str(len(N_trials)))
        #solve using gaussian elimination
        start = process_time()
        x_gaus = GaussElim(A,v)
        end = process_time()
        e_gaus[i] = np.mean(abs(v - np.dot(A,x_gaus)))
        t_gaus[i] = end - start
        #solve using partial pivot
        start = process_time()
        x_ppiv = PartialPivot(A,v)
        end = process_time()
        e_ppiv[i] = np.mean(abs(v - np.dot(A,x_ppiv)))
        t_ppiv[i] = end - start
        #solve using LU decomposition
        start = process_time()
        L, U, op = LUdecomp(A)
        x_LUdc = LUbacksub(L,U,permute(op,v))
        end = process_time()
        e_LUdc[i] = np.mean(abs(v - np.dot(A,x_LUdc)))
        t_LUdc[i] = end - start

    #plot dependence of error on N
    plt.plot(N_trials,e_gaus,label="gaus")
    plt.plot(N_trials,e_ppiv,label="ppiv")
    plt.plot(N_trials,e_LUdc,label="LUdc")
    plt.title("Error vs Matrix size")
    plt.ylabel("error")
    plt.xlabel("N")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()

    #plot dependence of timing on N
    plt.plot(N_trials,t_gaus,label="gaus")
    plt.plot(N_trials,t_ppiv,label="ppiv")
    plt.plot(N_trials,t_LUdc,label="LUdc")
    plt.title("Execution time vs Matrix size")
    plt.ylabel("exec time [s]")
    plt.xlabel("N")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()
        
