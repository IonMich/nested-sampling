# Based on code  translated to Python by Issac Trotts in 2007
#  Lighthouse at (x,y,z) emitted n flashes observed at positions on coast.
# Inputs:
#  Prior(u)    is uniform (=1) over (0,1), mapped to x = 4*u - 2; and
#
#  Prior(w)    is uniform (=1) over (0,1), mapped to z = 2*w; so that
#  Position    is 2-dimensional -2 < x < 2, 0 < z < 2 with flat prior
#  Likelihood  is L(x,z) = PRODUCT[k] (z/pi) / ((D[k] - x)^2 + z^2)
# Outputs:
#  Evidence    is Z = INTEGRAL L(x,z) Prior(x,z) dxdz
#  Posterior   is P(x,z) = L(x,z) / Z estimating lighthouse position
#  Information is H = INTEGRAL P(x,z) log(P(x,z)/Prior(x,z)) dxdz

from mininest import nested_sampling
import matplotlib.pyplot as plt
import numpy as np

def generatePositions(lightHCoords, samples=64):
    """
    
    """
    x=lightHCoords[0]
    z=lightHCoords[-1]
    if dim==2:
        thetaArray = np.random.uniform(-np.pi/2,np.pi/2,samples)
        flashesPositionsX, flashesPositionsY = z * np.tan(thetaArray) + x ,\
                                                    np.zeros(samples)
    elif dim==3:
        y=lightHCoords[1]
        thetaArray = np.random.uniform(0,np.pi/2,samples)
        flashesPositionsX = z * np.tan(thetaArray)
        flashesPositionsY = np.zeros(samples)
        
        phiArray = np.random.uniform(0,2*np.pi,samples)
        flashesPositionsX , flashesPositionsY = x + np.cos(phiArray)*(flashesPositionsX) - np.sin(phiArray)*(flashesPositionsY),\
                                                y + np.sin(phiArray)*(flashesPositionsX) + np.cos(phiArray)*(flashesPositionsY)
    
    return flashesPositionsX , flashesPositionsY

n = 100              # number of objects
max_iter = 2000     # number of iterations
dim = 3
transverseDim = dim - 1 

assert(dim==2 or dim==3)

# Number of flashes
N = 4000;
#np.random.seed(0)
positions = generatePositions([1.25,1.10,0.70], samples=N)

#map of unit domain to the spatial domain
transverse = lambda unit : 4.0 * unit - 2.0 
depth = lambda unit : 2.0 * unit

plt.figure('Flashes (Data)')
if dim==2:
    plt.hist(positions[0],50,range = (-10, 10))
if dim==3:
    plt.plot(positions[0],positions[1],'.')
    plt.xlim([-10,10])
    plt.ylim([-10,10])
