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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')
from sklearn.cluster import KMeans
from mininest import nested_sampling
from KDEpy import FFTKDE

def generatePositions(lightHCoords, samples_for_eachLH):
    """
    #lightHCoords=([[1st],[2nd]]) for 2LH
    #lightHCoords=([[1st]]) for 1LH
    """
    X=[]
    Y=[]
    for i in range(len(lightHCoords)):
        x=lightHCoords[i][0]
        z=lightHCoords[i][-1]
        if dim==2:
            thetaArray = np.random.uniform(-np.pi/2,np.pi/2,samples_for_eachLH)
            flashesPositionsX, flashesPositionsY = z * np.tan(thetaArray) + x ,\
            np.zeros(samples_for_eachLH)
        elif dim==3:
            y=lightHCoords[i][1]
            thetaArray = np.random.uniform(0,np.pi/2,samples_for_eachLH)

            flashesPositionsX, flashesPositionsY = z * np.tan(thetaArray), np.zeros(samples_for_eachLH)

            phiArray = np.random.uniform(0,2*np.pi,samples_for_eachLH)
            flashesPositionsX, flashesPositionsY = x + np.cos(phiArray)*(flashesPositionsX) - np.sin(phiArray)*(flashesPositionsY),\
                                                   y + np.sin(phiArray)*(flashesPositionsX) + np.cos(phiArray)*(flashesPositionsY)
        X,Y=np.append(X,[flashesPositionsX]),np.append(Y,[flashesPositionsY])


    return X,Y

n = 100            # number of objects
max_iter = 50000    # number of iterations
dim = 3
transverseDim = dim - 1
model_num_LH = 2

assert(dim==2 or dim==3)

# Number of flashes
N = 1000
#np.random.seed(0)

LHactualCoords=([[1.50,1.20,0.80],[-1.50,1.20,0.80]]) #Actual Coordinates of Light Houses
# LHactualCoords=([[1.50,1.10,0.70]]) #Actual Coordinates of Light Houses
actual = np.array(LHactualCoords)

flashesPositions = generatePositions(LHactualCoords, N)

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
                self.Coords[indexTuple] = transverse(unitSample)
            else:
                self.Coords[indexTuple] = depth(unitSample)

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
    x = np.array( lightHCoords[...,0])
    z = np.array(lightHCoords[...,-1])
    DX = flashesPositions[0]
    sumLikelihoodLH = 0

    if dim ==2:
        if np.sum(x.shape) == 0:
            sumLikelihoodLH = (z / np.pi) / ((DX - x)*(DX - x) + z*z)
        else:
            for e in range(model_num_LH):
                sumLikelihoodLH += (1/model_num_LH)* (z[e] / np.pi) / ((DX - x[e])*(DX - x[e]) + z[e]*z[e])


    elif dim==3:
        y = np.array(lightHCoords[...,1])
        DY = flashesPositions[1]
        if np.sum(x.shape) == 0:
            sumLikelihoodLH = (z / np.pi**2) / ((DX - x)*(DX - x) + (DY - y)*(DY - y) + z*z) / np.sqrt((DX - x)*(DX - x) + (DY - y)*(DY - y))
        else:
            for e in range(model_num_LH):
                sumLikelihoodLH += (1/model_num_LH)* (z[e] / np.pi**2) / ((DX - x[e])*(DX - x[e]) + (DY - y[e])*(DY - y[e]) + z[e]*z[e]) / np.sqrt((DX - x[e])*(DX - x[e]) + (DY - y[e])*(DY - y[e]))

    logL = np.sum( np.log(sumLikelihoodLH ))
    return logL

def sample_from_prior():
    """


    """
    unitCoords = np.random.uniform(size=(model_num_LH,dim))
    unitCoords = np.squeeze(unitCoords) # if (1,dim) squeeze to (dim,)
    Obj = LHouses(unitCoords)
    return Obj

def explore(Obj,logLstar):
    """
    # Evolve object within likelihood constraint
    # Object being evolved
    # Likelihood constraint L > Lstar
    """
    ret =  Obj.copy()
    step = 0.1   # Initial guess suitable step-size in (0,1)
    accept = 0   # # MCMC acceptances
    reject = 0   # # MCMC rejections
    a = 1.0
    Try = Obj.copy()          # Trial object
    for _ in range(20):  # pre-judged number of steps

        # Trial object u-w step
        unitCoords_New = ret.unitCoords + step * (2.0*np.random.uniform(size=ret.unitCoords.shape) - 1.0)  # |move| < step
        unitCoords_New -= np.floor(unitCoords_New)      # wraparound to stay within (0,1)
        Try.update(unitCoords_New)

        # Accept if and only if within hard likelihood constraint
        if Try.logL > logLstar:
            ret = Try.copy()
            accept+=1
        else:
            reject+=1

        # Refine step-size to let acceptance ratio converge around 50%
        if( accept > reject ):
            step *= np.exp(a / accept)
            a /= 1.5
        if( accept < reject ):
            step /= np.exp(a / reject)
            a *= 1.5
#    print(logLstar, accept)
    return ret

def cornerplots(posteriors,weights=None):
    """
    note bandwidth bw=0.01
    """
    pSize = posteriors[...,0].size # total number of posterior coordinates (3 for a single lhouse)
    numLhouses = pSize//dim
    transverseDomain = (-2,2)
    depthDomain = (0,2)
    domains = sum( ((transverseDomain,)*transverseDim,(depthDomain,))*numLhouses, () )
    plt.figure("Posterior plots")
    for i in range(pSize):
        plt.subplot(pSize,pSize,i*pSize+i+1)
        samples = posteriors[i]
        x = np.linspace(*domains[i],2000)
        estimator = FFTKDE(kernel='gaussian', bw=0.01)
        y = estimator.fit(samples, weights=weights).evaluate(x)
        plt.plot(x, y)
        plt.hist(samples,bins=50,range = domains[i],weights=weights,density=True) 
        # joint posteriors
        for j in range(i):
            subPltIndex = i*pSize + 1 + j
            plt.subplot(pSize,pSize,subPltIndex)
            xp, yp = posteriors[j],posteriors[i]
            xy = np.vstack([xp,yp]).T
            kde = FFTKDE(kernel='gaussian', norm=2,bw=0.05)
            grid, points = kde.fit(xy,weights).evaluate(2**7)
            # The grid is of shape (obs, dims), points are of shape (obs, 1)
            x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
            z = points.reshape(2**7, 2**7).T
            # Plot the kernel density estimate
            ax = plt.gca()
            ax.contourf(x, y, z, 1000, cmap="hot")
            plt.xlim(domains[j])
            plt.ylim(domains[i])
    plt.tight_layout()
    
#        plt.subplots_adjust(wspace=0.5, hspace=0.5)
#        plt.hist(posteriors[i], 100, range=domains[i], weights=wt)
#        if i==0:
#            plt.title("X Posterior Data")
#            plt.axvline(x=LHactualCoords[0][0], color='r', linestyle='dashed')
#            if model_num_LH==2: plt.axvline(x=LHactualCoords[1][0], color='r', linestyle='dashed')
#        elif i==1:
#            plt.title("Y Posterior Data")
#            plt.axvline(x=LHactualCoords[0][1], color='r', linestyle='dashed', linewidth=2)
#            if model_num_LH==2: plt.axvline(x=LHactualCoords[1][1], color='r', linestyle='dashed')
#        else:
#            plt.title("Z Posterior Data")
#            plt.axvline(x=LHactualCoords[0][2], color='r', linestyle='dashed', linewidth=2)
#            if model_num_LH==2: plt.axvline(x=LHactualCoords[1][2], color='r', linestyle='dashed')
#        
#        # joint posteriors
#        for j in range(i):
#            subPltIndex = i*pSize + 1 + j
#            ax = fig.add_subplot(pSize,pSize,subPltIndex)
#            plt.subplots_adjust(wspace=0.5, hspace=0.5)
#            ax.scatter(x=posteriors[j], y=posteriors[i],s=0.2)
#            plt.xlim(domains[j])
#            plt.ylim(domains[i])
#            if i==1:
#                plt.ylabel('y')
#                plt.plot(LHactualCoords[0][0],LHactualCoords[0][1], marker='*', color='r')
#                if model_num_LH==2: plt.plot(LHactualCoords[1][0],LHactualCoords[1][1], marker='*', color='r')
#            else:
#                if j==0:
#                    plt.xlabel('x')
#                    plt.ylabel('z')
#                    plt.plot(LHactualCoords[0][0],LHactualCoords[0][2], marker='*', color='r')
#                    if model_num_LH==2: plt.plot(LHactualCoords[1][0],LHactualCoords[1][2], marker='*', color='r')
#                else:
#                    plt.xlabel('y')
#                    plt.plot(LHactualCoords[0][1],LHactualCoords[0][2], marker='*', color='r')
#                    if model_num_LH==2: plt.plot(LHactualCoords[1][1],LHactualCoords[1][2], marker='*', color='r')
            
    
def plot_weights(weights):
    plt.figure('weights')
    plt.plot(weights[:len(weights)//model_num_LH])

def threeDimPlot(posteriors,weights=None):
    """
    assumes that posteriors is (dim,totalSamples) shaped numpy array
    """
    fig = plt.figure('{}-d plot'.format(dim))
    ax = fig.add_subplot(111, projection='3d')
    xp, yp, zp = posteriors[0,:],posteriors[1,:],posteriors[2,:]
    xyz = np.vstack([xp,yp,zp])
    color = gaussian_kde_weights(xyz,weights=weights)(xyz)
    idx = color.argsort()
    xps,yps,zps,cs,ws = xp[idx],yp[idx],zp[idx],color[idx],weights[idx]
    scatter = ax.scatter(xs=xps,ys=yps,zs=zps,s=2*ws/np.max(ws),c=cs, alpha=0.5,cmap='hot',label='Weighted Posterior')
    plt.colorbar(scatter)
    ax.scatter(xs=actual[...,0],ys=actual[...,1],zs=actual[...,2],marker = '*',color='red',s=200,depthshade=False,label='Actual LH')
    x , y , z = [] , [] , []
    for i in range(model_num_LH):
        x.append(clusterCenterPositions[i][0])
        y.append(clusterCenterPositions[i][1])
        z.append(clusterCenterPositions[i][-1])
    ax.scatter(x,y,z,marker = '*',color='green',s=200,depthshade=False,label='Cluster Estimate')
    ax.set_xlim(-2,2),ax.set_ylim(-2,2),ax.set_zlim(0,2)
    ax.set_xlabel('X axis'),ax.set_ylabel('Y axis'),ax.set_zlabel('Z axis')
    ax.set_title('A 3D-Plot of posterior points',weight='bold',size=12)
    plt.legend()
    plt.tight_layout()

def clustering(posteriors,weights=None,extraClusters=6):
    posteriorPoints = posteriors.T
    kmeans = KMeans(n_clusters=model_num_LH+extraClusters,max_iter=1000,tol=1E-7,n_init=100).fit(posteriorPoints,weights)
    clusterCenterPositions = kmeans.cluster_centers_[:model_num_LH]
    print(clusterCenterPositions)
    print(kmeans.inertia_)
    return clusterCenterPositions , kmeans
    
def get_posteriors(results):
    ni = results['num_iterations']
    samples = results['samples']
    shape =  samples[0].Coords.shape
    posteriors = np.zeros(sum( ( shape, (ni,) ), () ) )
    for i in range(ni):
        coords = samples[i].Coords
        posteriors[...,i] = coords
    posteriors = np.swapaxes(posteriors, 0, -2)
    posteriors = posteriors.reshape((dim,model_num_LH*ni))
    return posteriors

def get_weights(results):
    ni = results['num_iterations']
    samples = results['samples']
    logZ = results['logZ']
    weights = [0]*ni
    for i in range(ni):
        weights[i] = np.exp(samples[i].logWt - logZ)
    weights = weights * model_num_LH
    weights = np.array(weights)
    print(weights)

    return weights

def get_statistics(results,weights=None):   
    ni = results['num_iterations']
    samples = results['samples']
    shape =  samples[0].Coords.shape
    avgCoords = np.zeros(shape) # first moments of coordinates
    sqrCoords = np.zeros(shape) # second moments of coordinates
    logZ = results['logZ']
    for i in range(ni):
        coords = samples[i].Coords
        avgCoords += weights[i] * coords
        sqrCoords += weights[i] * coords * coords
    
    print("Num of Iterations: %i" %ni)    
    print("avgCoords[0]", avgCoords[0])
    
    meanX, sigmaX = avgCoords[0], np.sqrt(sqrCoords[0]-avgCoords[0]*avgCoords[0])
    print("mean(x) = %f, stddev(x) = %f" %(meanX, sigmaX))

    if dim==3:
        meanY, sigmaY = avgCoords[1], np.sqrt(sqrCoords[1]-avgCoords[1]*avgCoords[1])
        print("mean(y) = %f, stddev(y) = %f" %(meanY, sigmaY))

    meanZ, sigmaZ = avgCoords[-1], np.sqrt(sqrCoords[-1]-avgCoords[-1]*avgCoords[-1])
    print("mean(z) = %f, stddev(z) = %f" %(meanZ, sigmaZ))
    logZ_sdev = results['logZ_sdev']
    print("Evidence: ln(Z) = %g +- %g"%(logZ,logZ_sdev))

    # Analyze the changes in x,y,z and evidence for different z values
    statData = []
    statData.append((meanX, sigmaX))
    statData.append((meanY, sigmaY))
    statData.append((meanZ, sigmaZ))
    statData.append((logZ, logZ_sdev))
    return statData

def process_results(results):
    """
    return posterior numpy array in shape (numlhouses,dim,ni)
    """
    posteriors = get_posteriors(results)
    weights = get_weights(results)
    clusterCenterPositions , kmeans = clustering(posteriors,weights)
    if model_num_LH==1:
        statData = get_statistics(results,weights)
    else:
        statData = None
    return posteriors, weights, clusterCenterPositions, kmeans, statData

def do_plots(posteriors,weights):
    plot_weights(weights)
#    if dim==3: threeDimPlot(posteriors,weights)
    cornerplots(posteriors, weights)

if __name__ == "__main__":
    results = nested_sampling(n, max_iter, sample_from_prior, explore)

    posteriors, weights, clusterCenterPositions, kmeans, statData = process_results(results)
    do_plots(posteriors,weights)
    plt.show()
