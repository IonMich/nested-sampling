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
from KDEpy import TreeKDE

def generatePositions(lightHCoords, samples_for_eachLH):
    
    """
    Args:
        lightHCoords: A numpy array containing LH coordinates in 2D/3D.
        samples_for_eachLH: The number of flashes.
    Returns:
        (X, Y): The position of flashes observed at the shore.
    Description:
        Randomly generates a 'theta' and 'phi' as numpy arrays.
        Note: lightHCoords=([[1st],[2nd]]) for 2LH.
        Note: lightHCoords=([[1st]]) for 1LH.
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


    return (X,Y)

n = 100             # number of objects
max_iter = 50000    # number of iterations
dim = 3
transverseDim = dim - 1
model_num_LH = 2

assert(dim==2 or dim==3)

# Number of flashes
N = 1000

# LHactualCoords=([[1.50,0.70]]) # One 2D LightHouse - Actual Coordinates
#LHactualCoords=([[1.50,1.10,0.25]]) # One LightHouse - Actual Coordinates
# LHactualCoords=([[1.50,1.20,0.80],[-1.50,-1.20,0.60]]) #Two LightHouses - Actual Coordinates
#LHactualCoords=([[1.50,1.20,0.80],[-0.20,0.30,0.20],[-1.50,-1.20,0.60]]) #Three LightHouses - Actual Coordinates

############# or generate random actual Lhouse positions
actual_num_LH = 2
LHactualCoords_transv=np.random.uniform(-2, 2, size=(actual_num_LH, transverseDim))
LHactualCoords_depth=np.random.uniform(0, 2, size=(actual_num_LH, 1))
LHactualCoords = np.hstack([LHactualCoords_transv,LHactualCoords_depth]).tolist() #list of actual coordinates
#########################################################################################################

actual = np.array(LHactualCoords)
print("Actual lighthouse coordinates:\n",np.array(LHactualCoords))

flashesPositions = generatePositions(LHactualCoords, N)

#map of unit domain to the spatial domain
transverse = lambda unit : 4.0 * unit - 2.0
depth = lambda unit : 2.0 * unit

plt.figure('Flashes (Data)')
plt.title("Distribution of flashes")

if dim==2:
    plt.hist(flashesPositions[0],50,range = (-10, 10))
if dim==3:
    plt.plot(flashesPositions[0],flashesPositions[1],'.')
    plt.xlim([-10,10])
    plt.ylim([-10,10])
plt.xlabel('x')
plt.ylabel('y')

class LHouses():

    """
    Class definition for collection of lighthouses.
    """
    
    def __init__(self,unitArray):
        """
        Initializes the class with the following attributes.
        Note: For 2 lighthouses in 3D, unitArray should be a (2,3) array.
        """
        configDim = np.size(unitArray)
        assert(configDim%dim==0 and configDim>1)
        self.update(unitArray)
        self.logWt=None     # log(Weight), adding to SUM(Wt) = Evidence Z

    def update(self,unitArray):
        """
        Creates a new instance of the coordinate value.
        Computes the loglikehood of the coordinate.
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
        Converts from unit coordinates to lighthouse position(s)
        """
        self.Coords = np.zeros(self.unitCoords.shape)
        for indexTuple , unitSample in np.ndenumerate(self.unitCoords):
            if indexTuple[-1] != dim-1:
                self.Coords[indexTuple] = transverse(unitSample)
            else:
                self.Coords[indexTuple] = depth(unitSample)

    def assignlogL(self):
        """
        Assigns the attribute logLikelihood = ln Prob(data | position)
        """
        self.logL = logLhoodLHouse(self.Coords)

    def copy(self):
        """
        Returns the copy of the instance
        """
        return LHouses(self.unitCoords)


def logLhoodLHouse(lightHCoords):
    
    """
    Args:
        lightHCoords: Contains the coordinates of a lighthouse.
    Returns:
        logL: The log likelihood value for the given argument.
    Description:
        Uses specific formula for 2D and 3D case to calculate likelihood.
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
    Args:
        None.
    Returns:
        Obj: An object of the LHouses class.
    Description:
        Generates a 2D/3D coordinate and creates an object.
    """
    
    unitCoords = np.random.uniform(size=(model_num_LH,dim))
    unitCoords = np.squeeze(unitCoords) # if (1,dim) squeeze to (dim,)
    Obj = LHouses(unitCoords)
    return Obj

def explore(Obj,logLstar):
   
    """
    Args:
        Obj: An instance of the LHouses class.
        logLstar: The least likelihood value used in sampling.
    Returns:
        ret: A modified version of Obj.
    Description:        
        Performs Markov Chain Monte Carlo (MCMC) to modify the original object.  
        Object is evolved with likelihood constraint L > Lstar.
    """
    
    ret =  Obj.copy() 
    step = 0.1   # Initial guess suitable step-size in (0,1)
    accept = 0   # # MCMC acceptances
    reject = 0   # # MCMC rejections
    a = 1.0
    Try = Obj.copy()     # Trial object
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
    return ret

def cornerplots(posteriors,weights=None):

    """
    Args:
        posteriors: A numpy array containing posterior coordinates.
        weights: A numpy array containing weight distribution for posterior data samples.
    Returns:
        None.
    Description:
        Plots individually the posterior data for x, y and z.
        Creates a histogram plot and a scatter plot estimating the LH coordinates.
    """
    
    pSize = posteriors[...,0].size # total number of posterior coordinates (3 for a single lhouse)
    numLhouses = pSize//dim
    transverseDomain = (-2,2)
    depthDomain = (0,2)
    domains = sum( ((transverseDomain,)*transverseDim,(depthDomain,))*numLhouses, () )
    plt.figure("Posterior plots")
    plt.title("Posterior distribution of lighthouse(s)")
    for i in range(pSize):
        plt.subplot(pSize,pSize,i*pSize+i+1)
        samples = posteriors[i]
        x = np.linspace(*domains[i],2000)
        estimator = TreeKDE(kernel='gaussian', bw=0.01)
        y = estimator.fit(samples, weights=weights).evaluate(x)
        plt.plot(x, y)
        try:
            plt.hist(samples,bins=50,range = domains[i],weights=weights,density=True)
        except AttributeError:
            plt.hist(samples,bins=50,range = domains[i],weights=weights,normed=True)
        if i==0:
            plt.title("X Posterior Data")
            plt.axvline(x=LHactualCoords[0][0], color='r', linestyle='dashed')
            for k in range(len(LHactualCoords)):
                plt.axvline(x=LHactualCoords[k][0], color='r', linestyle='dashed')
        elif i==1 and dim==3:
            plt.title("Y Posterior Data")
            plt.axvline(x=LHactualCoords[0][1], color='r', linestyle='dashed')
            for k in range(len(LHactualCoords)):
                plt.axvline(x=LHactualCoords[k][1], color='r', linestyle='dashed')
        else:
            plt.title("Z Posterior Data")
            plt.axvline(x=LHactualCoords[0][2], color='r', linestyle='dashed')
            for k in range(len(LHactualCoords)):
                plt.axvline(x=LHactualCoords[k][2], color='r', linestyle='dashed') 
        
        # Joint posteriors
        for j in range(i):
            subPltIndex = i*pSize + 1 + j
            plt.subplot(pSize,pSize,subPltIndex)
            xp, yp = posteriors[j],posteriors[i]
            xy = np.vstack([xp,yp]).T
            kde = TreeKDE(kernel='gaussian', norm=2,bw=0.05)
            grid, points = kde.fit(xy,weights).evaluate(2**8)
            
            # The grid is of shape (obs, dims), points are of shape (obs, 1)
            x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
            z = points.reshape(2**8, 2**8).T
            
            # Plot the kernel density estimate
            ax = plt.gca()
            ax.contourf(x, y, z, 1000, cmap="hot")
            plt.xlim(domains[j])
            plt.ylim(domains[i])
            if i==1:
                plt.ylabel('y')
            else:
                if j==0:
                    plt.xlabel('x')
                    plt.ylabel('z')
                else:
                    plt.xlabel('y')
    plt.tight_layout()
    
def plot_weights(weights):
        
    """
    Args:
        weights: A numpy array containing weight distribution for posterior data samples.
    Returns:
        None.
    Description:
        Plots the weight distribution vs number of iteration.
    """
    
    plt.figure('Weights')
    plt.title("Weights distribution")
    plt.xlabel('Number of iterations')
    plt.ylabel('Weights')

    plt.plot(weights[:len(weights)//model_num_LH])

def threeDimPlot(posteriors,weights=None):
    
    """
    Args:
        posteriors: A numpy array containing posterior coordinates.
        weights: A numpy array containing weight distribution for posterior data samples.
    Returns:
        None.
    Description:
        Plots the actual LH coordinate and the estimated LH coordinates in a 3D box. 
    """
    
    fig = plt.figure('{}-D plot'.format(dim))
    ax = fig.add_subplot(111, projection='3d')
    xp, yp, zp = posteriors[0,:],posteriors[1,:],posteriors[2,:]
    xyz = np.vstack([xp,yp,zp]).T
    kde = TreeKDE(kernel='gaussian', norm=2,bw=0.05)
    color = kde.fit(xyz,weights).evaluate(xyz)
    ax = plt.gca()
    scatter = ax.scatter(xs=xp, ys=yp, zs=zp, c=color,cmap="hot")
    plt.colorbar(scatter)
    
    for i in range(len(LHactualCoords)):
        for j in range(3):
            if j!=0:
                xA = [LHactualCoords[i][0], LHactualCoords[i][0]]
                xC = [clusterCenterPositions[i][0],clusterCenterPositions[i][0]]
            else:
                xA = xC = [-2,2]
            if j!=1:
                yA = [LHactualCoords[i][1], LHactualCoords[i][1]]
                yC = [clusterCenterPositions[i][1],clusterCenterPositions[i][1]]
            else:
                yA = yC = [-2,2]
            if j!=2:
                zA = [LHactualCoords[i][2], LHactualCoords[i][2]]
                zC = [clusterCenterPositions[i][2],clusterCenterPositions[i][2]]
            else:
                zA = zC = [0, 2]
            ax.plot(xA,yA,zA,'r--',alpha=0.8, linewidth=3)
            ax.plot(xC,yC,zC,'g--',alpha=0.8, linewidth=3)
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

def clustering(posteriors,weights=None,extraClusters=20):
    
    """
    Args:
        posteriors: A numpy array containing posterior coordinates.
        weights: A numpy array containing weight distribution for posterior data samples.
        extraClusters: An average calculator used for higher accuracy.
    Returns:
        clusterCenterPositions: The mean value of the estimated LH coordinate.
        kmeans: An object instance that fits the data according to the cluster.
    Description:
        Required for multiple lighthouses.
        Determines LH positions by finely differentiating the posterior values.
        Performs clustering 20 times to achieve better estimate.   
    """
    
    posteriorPoints = posteriors.T
    kmeans = KMeans(n_clusters=model_num_LH,max_iter=1000,tol=1E-7,n_init=100).fit(posteriorPoints,weights)
    clusterCenterPositions = kmeans.cluster_centers_
    kmeans2 = KMeans(n_clusters=model_num_LH+extraClusters,max_iter=1000,tol=1E-7,n_init=100).fit(posteriorPoints,weights)
    clusterCenterPositions2 = kmeans2.cluster_centers_
    # print(clusterCenterPositions2)
    for i in range(len(clusterCenterPositions[...,0])):
        idx = np.argmin(np.sum(np.abs(clusterCenterPositions[i] - clusterCenterPositions2),axis=1))
        clusterCenterPositions[i] = clusterCenterPositions2[idx]

    print("Cluster positions:")
    print(clusterCenterPositions)
    return clusterCenterPositions , kmeans
    
def get_posteriors(results):
    
    """
    Args:
        results: A dictionary data returned by the mininest function.
    Returns:
        posteriors: A numpy array containing x,y,z coordinates.
    Description:
        Determines the dimension of the array required for posterioirs.
        Extracts coordinate from results and appends them to posteriors.
    """
    
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
    
    """
    Args:
        results: A dictionary data returned by the mininest function.
    Returns:
        weights: A numpy array containing weight distribution in the posterior data samples.
    Description:
        Extracts the evidence values from results.
    """
    
    ni = results['num_iterations']
    samples = results['samples']
    logZ = results['logZ']
    weights = [0]*ni
    for i in range(ni):
        weights[i] = np.exp(samples[i].logWt - logZ)
    weights = weights * model_num_LH
    weights = np.array(weights)

    return weights

def get_statistics(results,weights=None):
    
    """
    Args:
        results: A dictionary data returned by the mininest function.
        weights: A numpy array containing weight distribution for posterior data samples.
    Returns:
        statData: A list of tuples containing statistical data.
    Description:
        Extracts the mean and standard deviation from results.
        Prints the extracted data.
    """
    
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
        
    meanX, sigmaX = avgCoords[0], np.sqrt(sqrCoords[0]-avgCoords[0]*avgCoords[0])
    print("\nmean(x) = %f, stddev(x) = %f" %(meanX, sigmaX))

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
    if dim==3: statData.append((meanY, sigmaY))
    statData.append((meanZ, sigmaZ))
    statData.append((logZ, logZ_sdev))
    return statData

def z_test():
    
    """
    Args:
        None
    Returns:
        None.
    Description:
        Determine the standard deviation for different z coordinates of lighthouse.
        Vary the z coord to analyze changes in the uncertainty of evidence.
    """
    
    print("\n***TEST: Change in posterior data while varying z between 0 to 2***")
    zVals = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] 
    global flashesPositions
    initPositions = flashesPositions

    x, y = LHactualCoords[0][0], LHactualCoords[0][1]
    dataStat = [[] for i in range(4)]
    for z in zVals:
        print("\nLighthouse at a height(z) = %f" %z)
        flashesPositions = generatePositions([[x, y, z]], N)
        print("\nActual lighthouse coordinates:\n",[x,y,z])
        results = nested_sampling(n, max_iter, sample_from_prior, explore)
        processData = process_results(results)[4]
        for k in range(4):
            dataStat[k].append(processData[k])
    sigmaZ, sigmaEvid = [], [] 
    for i in range (len(zVals)):
        sigmaZ.append(dataStat[2][i][1])
        sigmaEvid.append(dataStat[3][i][1])
    
    plt.figure("Uncertainty change for z")
    plt.title("Uncertainty in measured z")
    plt.plot(zVals, sigmaZ, 'b')
    plt.xlabel("z")
    plt.ylabel("sigmaZ")
    plt.show()
    
    plt.figure("Uncertainty change for evidence")
    plt.title("Uncertainty in measured evidence")
    plt.plot(zVals, sigmaEvid, 'r')
    plt.xlabel("z")
    plt.ylabel("sigmaEvidence")
    plt.show()

    flashesPositions = initPositions
    
def process_results(results):
    
    """
    Args:
        results: A dictionary data returned by the mininest function.
    Returns:
        posteriors: A numpy array containing posterior coordinates.
        weights: A numpy array containing weight distribution for posterior data samples.
        clusterCenterPositions: The mean value of the estimated LH coordinate.
        kmeans: An object instance that fits the data according to the cluster.
        statData: A list of tuples containing statistical data.
    Description:
        Serves as a hub for the main function.
    """
    
    posteriors = get_posteriors(results)
    weights = get_weights(results)
    clusterCenterPositions , kmeans = clustering(posteriors,weights)
    if len(LHactualCoords)==1:
        statData = get_statistics(results,weights)
    else:
        statData = None
    return posteriors, weights, clusterCenterPositions, kmeans, statData

def do_plots(posteriors, weights):
    
    """
    Args:
        posteriors: A numpy array containing posterior coordinates.
        weights: A numpy array containing weight distribution for posterior data samples.
    Returns:
        None
    Description:
        Plot the weight distribution.
        Plot the 3D graph for posterior for 3D case.
        Plot the cornerplot to show posterior data.
    """
    
    print("\nGenerating Plots. This might take some time...")
    plot_weights(weights)
    if dim==3: threeDimPlot(posteriors,weights)
    cornerplots(posteriors, weights)
    
def compare_models1LH_2LH():
    
    """
    Args:
        None
    Returns:
        logZvalues1LH: Evidences for 1LH model
        logZvalues2LH: Evidences for 2LH model
    Description:
        Z_vs_x_2LH.pdf: Plots showing comparision of 1LH and 2Lh model
    """
    
    xvalues=np.linspace(0,0.1,11).tolist()
    logZvalues1LH = []
    logZvalues2LH = []
    global LHactualCoords,model_num_LH,flashesPositions
    initLHact=LHactualCoords
    initmodel_num=model_num_LH
    initflashes=flashesPositions
    for i,values in enumerate(xvalues):
        print("Current separation: {}".format(2*values))
        LHactualCoords[0][0]=values
        LHactualCoords[1][0]=-values
        flashesPositions = generatePositions(LHactualCoords, N)
        print(LHactualCoords)
        model_num_LH=1
        results = nested_sampling(n, max_iter, sample_from_prior, explore)
#        posteriors, kmeans, statData = process_results(results)
        logZ = results['logZ']
        logZ_sdev = results['logZ_sdev']
        print("logZ for {} Lhouse: {} +- {}".format(model_num_LH,logZ,logZ_sdev))
        logZvalues1LH.append(logZ)
        model_num_LH=2
        results = nested_sampling(n, max_iter, sample_from_prior, explore)
#        posteriors, kmeans, statData = process_results(results)
        logZ = results['logZ']
        logZ_sdev = results['logZ_sdev']
        print("logZ for {} Lhouse: {} +- {}".format(model_num_LH,logZ,logZ_sdev))
        logZvalues2LH.append(logZ)
    LHactualCoords=initLHact
    model_num_LH=initmodel_num
    flashesPositions=initflashes
    plt.figure()
    plt.plot(2*xvalues,logZvalues2LH-logZvalues1LH,'r-.')
    plt.xlabel('Separation between sources')
    plt.ylabel('log(Ratio of Evidences)')
    plt.suptitle('Comparision of Evidences for 1LH and 2LH models')
    plt.savefig('Z_vs_x_2LH.pdf')
    return logZvalues1LH, logZvalues2LH

if __name__ == "__main__":
    results = nested_sampling(n, max_iter, sample_from_prior, explore)
    posteriors, weights, clusterCenterPositions, kmeans, statData = process_results(results)

    do_plots(posteriors,weights)
    plt.show()
    
    if len(LHactualCoords)==1: z_test()
