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
    
    
class LightHouse:
    """
    
    """
    def __init__(self,unitArray):
        assert(len(unitArray)==dim)
        self.update(unitArray)
        self.logWt=None     # log(Weight), adding to SUM(Wt) = Evidence Z
        
    def update(self,unitArray):
        """
        
        """
        assert(len(unitArray)==dim)
        self.unitCoords = np.zeros(dim)
        for i , unitSample in enumerate(unitArray):
            self.unitCoords[i] = unitSample  # Uniform-prior controlling parameter for position
        self.mapUnitToXYZ()
        self.assignlogL()
        
    def mapUnitToXYZ(self):
        """
        go from unit coordinates to lighthouse position 
        """
        self.Coords = np.zeros(dim)
        for i in range(transverseDim):
            self.Coords[i] = transverse(self.unitCoords[i])
        self.Coords[-1] = depth(self.unitCoords[-1])
            
    def assignlogL(self):
        """
        assign the attribute
        # logLikelihood = ln Prob(data | position)
        """
        self.logL = logLhoodLHouse(self.Coords)
    
    def copy(self):
        """
        
        """
        return LightHouse(self.unitCoords)


def logLhoodLHouse(lightHCoords):    
    """     
    logLikelihood function
     Easterly position
     Northerly position
    """
    x = lightHCoords[0]
    z = lightHCoords[-1]
    DX = positions[0]
    
    if dim ==2:
        logL = np.sum( np.log( (z / np.pi) / ((DX - x)*(DX - x) + z*z) ) )
    elif dim==3:
        y = lightHCoords[1]
        DY = positions[1]
        logL = np.sum( np.log( (z / np.pi**2) / ((DX - x)*(DX - x) + (DY - y)*(DY - y) + z*z) / np.sqrt((DX - x)*(DX - x) + (DY - y)*(DY - y)) ) )
    return logL

def sample_from_prior():
    """
    
    
    """
    unitCoords = np.random.uniform(size=dim)
    Obj = LightHouse(unitCoords)
    return Obj