#################################################################
# Name:     Gauss-Seidel.py                                     #
# Authors:  Michael Battaglia                                   #
# Function: Program calculates PDE solution using Gauss-Seidel  #
#           relaxation method.                                  #
#################################################################

#essential imports
import numpy as np
import matplotlib.pyplot as plt

#function: Gauss-Seidel relaxation method
def GaussSeidel(image, delta, relax, flag=-99, plot=False):
    #assume image full of flags except at boundaries
    im1 = np.copy(image)
    #calculate boundary mean
    bdy = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if abs(image[i][j]-flag) > 1e-10:
                #point is on boundary
                bdy.append(image[i][j])
    bdy = np.mean(bdy)
    #change flags to boundary mean
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if abs(image[i][j]-flag) < 1e-10:
                #point is in interior
                im1[i][j] = bdy
    #plot initial state if parameter give
    if plot:
        plt.imshow(im1.T, interpolation='nearest', cmap='gray')
        plt.title("Gauss-Seidel Solving PDE")
        plt.colorbar()
        plt.draw()
        plt.pause(0.01)
            
    #initialize error
    error = 2*delta
    #relax until error within tolerance
    while error > delta:
        im2 = np.copy(im1)
        #for each row
        for i in range(image.shape[0]):
            #for each column
            for j in range(image.shape[1]):
                #leave boundary untouched (mark non-bnd by -99)
                if abs(image[i][j]-flag) < 1e-10:
                    #Gauss Seidel relaxation
                    im2[i][j] = 0.25*(1+relax)*(im2[i+1,j]+im2[i-1,j]
                                    +im2[i,j+1]+im2[i,j-1])-relax*im2[i][j]
        #calculate error
        error = np.amax(np.absolute(im2-im1))
        im1 = np.copy(im2)
        #plot if parameter give
        if plot:
            plt.cla()
            plt.imshow(im1.T, interpolation='nearest', cmap='gray')
            plt.draw()
            plt.pause(0.01)
    #close plots
    if plot:
        plt.close()
    return im1, error
            

#function: main
if __name__ == '__main__':
    #construct grid 8cm x 20cm at 0.1cm resolution
    length = 8.0
    width = 20.0
    dx = 0.1
    T = -99*np.ones([int(width/dx),int(length/dx)],float)
    #initialize boundary values
    T[:,0].fill(10.0)  #top row
    T[0,:] = np.linspace(10.0,0,T.shape[1],endpoint=True) #left column
    T[-1,:] = np.linspace(10.0,0,T.shape[1],endpoint=True) #right column
    #bottom row with indent
    T[0:int(5.0/dx),-1] = np.linspace(0,5.0,int(5.0/dx),endpoint=True) #lrow
    T[T.shape[0]-int(5.0/dx):T.shape[0],-1] = np.linspace(5.0,0,int(5.0/dx),endpoint=True) #rrow
    T[int(5.0/dx)-1,T.shape[1]-int(3.0/dx):T.shape[1]] = np.linspace(7.0,5.0,int(3.0/dx),endpoint=True) #left indent
    T[T.shape[0]-int(5.0/dx),T.shape[1]-int(3.0/dx):T.shape[1]] = np.linspace(7.0,5.0,int(3.0/dx),endpoint=True) #right indent
    T[int(5.0/dx)-1:T.shape[0]-int(5.0/dx)+1,T.shape[1]-int(3.0/dx)].fill(7.0) #top indent
    #fill out indent area
    T[int(5.0/dx):T.shape[0]-int(5.0/dx),T.shape[1]-int(3.0/dx)+1:T.shape[1]].fill(0.0)
    
    #PDE solution accuracy
    delta = 1.0e-6
    #Gauss-Seidel relaxation parameter w
    relax = 0.9
    #solve PDE using Gauss-Seidel relaxation
    T,error = GaussSeidel(T, delta=delta, relax=relax, flag=-99, plot=True)
    
    #plot final solution
    plt.title("Gauss-Seidel Relaxation for Laplace's Eqn")
    plt.imshow(T.T, interpolation='nearest', cmap='gray')
    plt.colorbar()
    plt.show()
