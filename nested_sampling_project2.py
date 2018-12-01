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
flashesPositions = generatePositions([1.25,1.10,0.70], samples=N)

#map of unit domain to the spatial domain
transverse = lambda unit : 4.0 * unit - 2.0
depth = lambda unit : 2.0 * unit

plt.figure('Flashes (Data)')
if dim==2:
    plt.hist(flashesPositions[0],50,range = (-10, 10))
if dim==3:
    plt.plot(flashesPositions[0],flashesPositions[1],'.')
    plt.xlim([-10,10])
    plt.ylim([-10,10])


class LHouses():
    """
    TESTS:
    put the following at the begining of the if name==main statement

    myLH = LightHouse(np.array([0.2,0.3,0.4]) )
    myLHpair = LHouses(np.array([0.2,0.3,0.4]) )
    myLHpair2 = LHouses(np.array([[0.2,0.3,0.4],[0.5,0.6,0.8]]) )
    print(myLH.__dict__)
    print(myLHpair.__dict__)
    print(myLHpair2.__dict__)
    """
    def __init__(self,unitArray):
        """
        for 2 lighthouses in 3d, unitArray should be a (2,3) array
        """
        configDim = np.size(unitArray)
        assert(configDim%dim==0 and configDim>1)
        self.update(unitArray)
        self.logWt=None     # log(Weight), adding to SUM(Wt) = Evidence Z

    def update(self,unitArray):
        """

        """
        configDim = np.size(unitArray)
        assert(configDim % dim == 0 and configDim>1)
        self.unitCoords = np.zeros(unitArray.shape)
        for indexTuple , unitSample in np.ndenumerate(unitArray):
            self.unitCoords[indexTuple] = unitSample  # Uniform-prior controlling parameter for position
        self.mapUnitToXYZ()
        self.assignlogL()

    def mapUnitToXYZ(self):
        """
        go from unit coordinates to lighthouse position(s)
        """
        self.Coords = np.zeros(self.unitCoords.shape)
        for indexTuple , unitSample in np.ndenumerate(self.unitCoords):
            if indexTuple[-1] != dim-1:
                self.Coords[indexTuple] = transverse(self.unitCoords[indexTuple])
            else:
                self.Coords[indexTuple] = depth(self.unitCoords[indexTuple])

    def assignlogL(self):
        """
        assign the attribute
        # logLikelihood = ln Prob(data | position)
        """
#        assert(False) ## waiting for new LH function
        self.logL = logLhoodLHouse(self.Coords)

    def copy(self):
        """

        """
        return LHouses(self.unitCoords)


def logLhoodLHouse(lightHCoords):
    """
    logLikelihood function
     Easterly position
     Northerly position
    """

    i = len(lightHCoords.shape)
    if i == 1:
        x = lightHCoords[0]
        z = lightHCoords[-1]
        DX = flashesPositions[0]

        if dim ==2:
            logL = np.sum( np.log( (z / np.pi) / ((DX - x)*(DX - x) + z*z) ) )
        elif dim==3:
            y = lightHCoords[1]
            DY = flashesPositions[1]
            logL = np.sum( np.log( (z / np.pi**2) / ((DX - x)*(DX - x) + (DY - y)*(DY - y) + z*z) / np.sqrt((DX - x)*(DX - x) + (DY - y)*(DY - y)) ) )
    if i == 2:
        x1 = lightHCoords[0,0]
        z1 = lightHCoords[0,2]
        x2 = lightHCoords[1,0]
        z2 = lightHCoords[1,2]
        DX = flashesPositions[0]

        if dim ==2:
            logL = np.sum( np.log( (z1 / np.pi) / ((DX - x1)*(DX - x1) + z1*z1) + (z2 / np.pi) / ((DX - x2)*(DX - x2) + z2*z2g) ) )
        elif dim==3:
            y1 = lightHCoords[0,1]
            y2 = lightHCoords[0,1]
            DY = flashesPositions[1]
            logL = np.sum( np.log(  1/2 * (z1 / np.pi**2) / ((DX - x1)*(DX - x1) + (DY - y1)*(DY - y1) + z1*z1) / np.sqrt((DX - x1)*(DX - x1) + (DY - y1)*(DY - y1)) + \
             1/2 * (z2 / np.pi**2) / ((DX - x2)*(DX - x2) + (DY - y2)*(DY - y2) + z2*z2) / np.sqrt((DX - x2)*(DX - x2) + (DY - y2)*(DY - y2)) ) )
    return logL

def sample_from_prior():
    """


    """
    unitCoords = np.random.uniform(size=dim)
    Obj = LHouses(unitCoords)
    return Obj

def explore(Obj,logLstar):
    """
    # Evolve object within likelihood constraint
    # Object being evolved
    # Likelihood constraint L > Lstar
    """
    ret =  Obj.copy()
    step = 0.1;   # Initial guess suitable step-size in (0,1)
    accept = 0;   # # MCMC acceptances
    reject = 0;   # # MCMC rejections
    a = 1.0;
    Try = Obj.copy()          # Trial object
    for m in range(20):  # pre-judged number of steps

        # Trial object u-w step
        unitCoords_New = ret.unitCoords + step * (2.0*np.random.uniform(size=ret.unitCoords.shape) - 1.0);  # |move| < step
        unitCoords_New -= np.floor(unitCoords_New);      # wraparound to stay within (0,1)
        Try.update(unitCoords_New)

        # Accept if and only if within hard likelihood constraint
        if Try.logL > logLstar:
            ret = Try.copy()
            accept+=1
        else:
            reject+=1

        # Refine step-size to let acceptance ratio converge around 50%
        if( accept > reject ):
            step *= np.exp(a / accept);
            a /= 1.5
        if( accept < reject ):
            step /= np.exp(a / reject);
            a *= 1.5
#    print(logLstar, accept)
    return ret

def cornerplots(posteriors):
    """

    """
    transverseDomain = (-2,2)
    depthDomain = (0,2)
    domains = sum(((transverseDomain,)*transverseDim,(depthDomain,)),())
    plt.figure('posteriors')
    for i in range(dim):
        plt.subplot(dim,dim,i*dim+i+1)
        plt.hist(posteriors[i],500,range = domains[i])

        # joint posteriors
        for j in range(i):
            subPltIndex = i*dim + 1 + j
            plt.subplot(dim,dim,subPltIndex)
            plt.plot(posteriors[j],posteriors[i],'.')
    plt.show()

def process_results(results):
    """


    """
    avgCoords = [0.0,]*dim # first moments of coordinates
    sqrCoords = [0.0,]*dim # second moments of coordinates
    ni = results['num_iterations']
    samples = results['samples']
    logZ = results['logZ']
    posteriors = [ [] for i in range(dim) ]
    for j in range(dim):
        for i in range(ni):
            w = np.exp(samples[i].logWt - logZ); # Proportional weight
            posteriors[j].append(samples[i].Coords[j])
            avgCoords[j] += w * samples[i].Coords[j];
            sqrCoords[j] += w * samples[i].Coords[j] * samples[i].Coords[j];
    cornerplots(posteriors)

    logZ_sdev = results['logZ_sdev']
#    H = results['info_nats']
#    H_sdev = results['info_sdev']
    print("# iterates: %i"%ni)
    print("Evidence: ln(Z) = %g +- %g"%(logZ,logZ_sdev))
#    print("Information: H  = %g nats = %g bits"%(H,H/log(2.0)))
    print("mean(x) = {:9.4f}, stddev(x) = {:9.4f}".format(avgCoords[0], np.sqrt(sqrCoords[0]-avgCoords[0]*avgCoords[0])));
    if dim ==3: print("mean(y) = {:9.4f}, stddev(y) = {:9.4f}".format(avgCoords[1], np.sqrt(sqrCoords[1]-avgCoords[1]*avgCoords[1])));
    print("mean(z) = {:9.4f}, stddev(z) = {:9.4f}".format(avgCoords[-1], np.sqrt(sqrCoords[-1]-avgCoords[-1]*avgCoords[-1])));

if __name__ == "__main__":
    results = nested_sampling(n, max_iter, sample_from_prior, explore)
    process_results(results)
