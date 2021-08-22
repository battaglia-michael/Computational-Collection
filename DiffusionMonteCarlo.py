#################################################################
# Name:     randDLA.py                                          #
# Authors:  Michael Battaglia                                   #
# Function: Program simulates diffusion limited aggregation     #
#           using Monte Carlo Methods.                          #
#################################################################

#essential modules
import numpy as np
import matplotlib.pyplot as plt

#function: 2D diffusion limited random walk
def randDLA(lims, sink=False, source=False, periodic=True, N=False):
    """
    sink: position vector for aggregation sink
          if False, then boundary is sink
    source: position vector for particle source
            if False, then particles randomly appear on Free spaces
    lims: vector of dimension lengths
          if False, then periodic boundaries
    N: number of participating particles
       if False, then spawn particles until source or boundary is taken
    """
    if sink is False:
        if periodic:
            #there will be no aggregate
            print("No aggregate can form")
            return float("NaN")
    if N is False:
        if source is False:
            if not periodic:
                #aggregate will never end
                print("No end condition for aggregate")
                return float("NaN")
    
    #initialize list of occupied positions
    occupied_pos = []
    anchored=np.zeros(lims,dtype=int)
    #generate particles
    generate = True
    while generate:
        if source:
            #specified source
            pos = pos_0
        else:
            #random source particle
            pos = np.array([np.random.randint(0,lim) for lim in lims],dtype=int)
        if not anchored[pos[0]][pos[1]]:
            #take each step if position is not in a stuck position
            while not isStuck(pos, lims, sink, periodic, anchored):
                #take a random step in a random direction with a random orientation
                step = np.zeros(len(pos))
                step[np.random.randint(0,len(pos))] = 1-2*np.random.randint(0,2)
                pos = pos + step
                #impose periodic boundary
                pos = np.mod(pos,lims).astype(int)
                if len(occupied_pos)==0:
                    print("Position:", pos)
            occupied_pos.append(pos)
            anchored[pos[0],pos[1]] = 1
            print("Anchored:",len(occupied_pos))
            print("Anchor pos:",pos)
            if N:
                #generate until N particles
                if len(occupied_pos) == N:
                    #generated N particles
                    generate = False
            else:
                #generate until
                if source:
                    #source covered triggers end
                    if all(pos == source):
                        #occupied source
                        generate = False
                if periodic:
                    #boundary covered triggers end
                    if any(pos==lims-1) or any(pos==0):
                        #occupied boundary
                        generate = False
    #return list of occupied positions
    return anchored, np.array(occupied_pos)

#function: check if particle is stuck (to edge, or other particle)
def isStuck(pos, lims, sink, periodic, anchored):
    xp = pos[0]
    yp = pos[1]
    if not periodic:
        #not periodic, gets stuck on wall
        if any(pos==lims-1) or any(pos==0):
            #if the particle has reached a wall
            return True
    if all(pos==sink):
        #if particle hits sink
        return True
    if anchored[xp-1:xp+2,yp-1:yp+2].any():
        #if particle is adjacent to an anchored particle
        return True
    else:
        #particle is free
        return False

#function: animated plot of 2D random walk
def D2plot(pos, animate=0.01):
    if animate:
        for i in range(len(pos)):
            plt.cla()
            plt.title('diffusion limited aggregation')
            plt.scatter(pos.T[0][:i+1],pos.T[1][:i+1])
            plt.xlabel('x position')
            plt.ylabel('y position')
            plt.draw()
            plt.pause(animate)
    else:
        plt.title('diffusion limited aggregation')
        plt.scatter(pos.T[0],pos.T[1])
        plt.xlabel('x position')
        plt.ylabel('y position')
    plt.show()

#function: evaluate fractal dimension
def fracDim(image):
    print(image.shape)
    cen = (np.array([image.shape[0],image.shape[1]])-1)/2
    r = range(1, min(cen)+1)
    m = np.zeros(len(r))
    for i in range(len(r)):
        subimage = image[cen[0]-r[i]:cen[0]+r[i]+1,cen[1]-r[i]:cen[1]+r[i]+1]
        m[i] = subimage.sum()
    plt.title("fractal dimension")
    plt.plot(r,m)
    plt.ylabel('mass')
    plt.xlabel('radius')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

#function: main
if __name__ == '__main__':
    #size of box
    L = np.array([201,201])
    #central position
    central = (L-1)/2
    #take a random walk until aggregation reaches edge
    image, pos = randDLA(L, central)
    #plot path
    D2plot(pos, animate=True)
    #plot fractal dimension
    fracDim(image)
