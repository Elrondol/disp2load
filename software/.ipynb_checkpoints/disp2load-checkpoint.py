import numpy as np
from scipy.integrate import quad
from numpy import arctan, tan, cos, sin, arccos, arcsin, sqrt, log, log10, pi
from scipy.optimize import minimize
import time
import json
import utils

def compute_I1(r, z):
    """
    Computes I1 from the paper.

    Parameters
    ----------
    r : numpy.ndarray
        Input array of shape (nv, 2) representing vertices, where nv is the number of vertices. 
    z : float
        Depth.

    Returns
    -------
    numpy.ndarray
        Computed I1.
    """
    nv = r.shape[0]
    rot = np.array([[0, 1], [-1, 0]])
    I1 = np.array([0, 0])

    for i in range(nv):
        ri = r[i, :]

        if i == nv - 1:
            rj = r[0, :]
        else:
            rj = r[i + 1, :]

        rjio = np.dot(rot, (rj - ri))
        a = np.dot(rj-ri,rj-ri)
        b = np.dot(ri,rj-ri)
        c = np.dot(ri,ri)
        d = c + z**2
        e = d - b**2 / a
        t0 = b / a
        t1 = 1 + b / a

        if np.linalg.norm(rjio) != 0:
            S1i = compute_S1i(a, e, z, t0, t1)
            I1 = I1 + np.dot(S1i, rjio)

    return I1

def compute_I2(r, z):
    """
    Computes I2 from the paper.

    Parameters
    ----------
    r : numpy.ndarray
        Input array of shape (nv, 2) representing vertices, where nv is the number of vertices. 
    z : float
        Depth.

    Returns
    -------
    numpy.ndarray
        Computed I2.
    """
    nv = r.shape[0]
    rot = np.array([[0, 1], [-1, 0]])
    I2 = 0

    for i in range(nv):
        ri = r[i, :]

        if i == nv - 1:
            rj = r[0, :]
        else:
            rj = r[i + 1, :]

        rjio = np.dot(rot, (rj - ri))
        a = np.dot(rj - ri, rj - ri)
        b = np.dot(ri, rj - ri)
        c = np.dot(ri, ri)
        d = c + z**2
        e = d - b**2 / a
        t0 = b / a
        t1 = 1 + b / a

        if np.linalg.norm(rjio) != 0:
            S2i = compute_S2i(a, e, t0, t1)
            I2 = I2 + np.dot(rjio, S2i)

    return I2

def compute_I3(r, z):
    """
    Computes I3 from the paper.

    Parameters
    ----------
    r : numpy.ndarray
        Input array of shape (nv, 2) representing vertices, where nv is the number of vertices. 
    z : float
        Depth.

    Returns
    -------
    numpy.ndarray
        Computed I3.
    """
    nv = r.shape[0]
    rot = np.array([[0, 1], [-1, 0]])
    I3 = 0

    for i in range(nv):
        ri = r[i, :]

        if i == nv - 1:
            rj = r[0, :]
        else:
            rj = r[i + 1, :]

        rjo = np.dot(ri, np.dot(rot, rj))
        a = np.dot(rj - ri, rj - ri)
        b = np.dot(ri, rj - ri)
        c = np.dot(ri, ri)
        d = c + z**2
        e = d - b**2 / a
        t0 = b / a
        t1 = 1 + b / a

        if rjo != 0:
            S2i = compute_S2i(a, e, t0, t1)
            I3 = I3 + rjo * S2i

    return I3

def compute_I4(r, z):
    """
    Computes I4 from the paper.

    Parameters
    ----------
    r : numpy.ndarray
        Input array of shape (nv, 2) representing vertices, where nv is the number of vertices. 
    z : float
        Depth.

    Returns
    -------
    numpy.ndarray
        Computed I4.
    """
    nv = r.shape[0]
    rot = np.array([[0, 1], [-1, 0]])
    I4 = 0

    for i in range(nv):
        ri = r[i, :]

        if i == nv - 1:
            rj = r[0, :]
        else:
            rj = r[i + 1, :]

        rjo = np.dot(ri, np.dot(rot, rj))
        a = np.dot(rj - ri, rj - ri)
        b = np.dot(ri, rj - ri)
        c = np.dot(ri, ri)
        d = c + z**2
        A = c / a - b**2 / a**2
        B = d / a - b**2 / a**2
        t0 = b / a
        t1 = 1 + b / a

        if rjo != 0:
            S3i = compute_S3i(a, A, B, t0, t1)
            I4 = I4 + rjo * S3i

    return I4

def compute_S1i(a, e, z, t0, t1):
    """
    Computes S1i from d'Urso and Marmo 2013.

    Parameters
    ----------
    a : float
        Product between position vectors, see a_i in the paper for the formulation.
    e : float
        Computed as : d-b**2/a, see e_i in the appendix A of the paper.
    z : float
        Depth.
    t0 : float
        Computed as : b/a, see t_0i in the appendix A of the paper.
    t1 : float
        Computed as : 1+b/a, see t_1i in the appendix A of the paper.
    Returns
    -------
    float
        Computed S1i.
    """
    S1i = sqrt(a) * t0 - sqrt(a) * t1

    if e - z**2 != 0:
        S1i = S1i - sqrt(e - z**2) * arctan(sqrt(a) * t0 / sqrt(e - z**2)) + \
              sqrt(e - z**2) * arctan(sqrt(a) * t1 / sqrt(e - z**2)) + \
              sqrt(e - z**2) * arctan(sqrt(a) * t0 * z / (sqrt(e + a * t0**2) * sqrt(e - z**2))) - \
              sqrt(e - z**2) * arctan(sqrt(a) * t1 * z / (sqrt(e + a * t1**2) * sqrt(e - z**2)))

    if z != 0:
        S1i = S1i - z * np.log(a * t0 + np.sqrt(a) * np.sqrt(e + a * t0**2)) + \
              z * log(a * t1 + sqrt(a) * sqrt(e + a * t1**2))

    if t0 != 0:
        S1i = S1i - sqrt(a) * t0 * log(sqrt(e + a * t0**2) + z)

    if t1 != 0:
        S1i = S1i + sqrt(a) * t1 * log(sqrt(e + a * t1**2) + z)

    return S1i / sqrt(a)

def compute_S2i(a, e, t0, t1):
    """
    Computes S2i from d'Urso and Marmo 2013.

    Parameters
    ----------
    a : float
        Product between position vectors, see a_i in the paper for the formulation.
    e : float
        Computed as : d-b**2/a, see e_i in the appendix A of the paper.
    t0 : float
        Computed as : b/a, see t_0i in the appendix A of the paper.
    t1 : float
        Computed as : 1+b/a, see t_1i in the appendix A of the paper.
    Returns
    -------
    float
        Computed S2i.
    """
    S2i = log((a * t1 + np.sqrt(a * (a * t1**2 + e))) / (a * t0 + np.sqrt(a * (a * t0**2 + e)))) / sqrt(a)
    return S2i

def compute_S3i(a, A, B, t0, t1):
    """
    Computes S3i from d'Urso and Marmo 2013.

    Parameters
    ----------
    a : float
        Product between position vectors, see a_i in the paper for the formulation.
    A : float
        Computed as : c / a - b**2 / a**2, not expressed explicitly in the paper.
    B : float
        Computed as : d / a - b**2 / a**2 , not expressed explicitly in the paper.
    t0 : float
        Computed as : b/a, see t_0i in the appendix A of the paper.
    t1 : float
        Computed as : 1+b/a, see t_1i in the appendix A of the paper.
    Returns
    -------
    float
        Computed S3i.
    """
    S3i = arctan(t1 * sqrt((B - A) / (A * (B + t1**2)))) - arctan(t0 * sqrt((B - A) / (A * (B + t0**2))))
    S3i = S3i * sqrt((B - A) / A) + log((t1 + sqrt(B + t1**2)) / (t0 + sqrt(B + t0**2))) / sqrt(a)
    return S3i


def alpha(r):
    """
    Computes alpha from the paper.

    Parameters
    ----------
    r : numpy.ndarray
        Input array of shape (nv, 2) representing the coordinates of the vertices.

    Returns
    -------
    float
        Computed alpha.
    """
    a = 0
    for i in range(len(r)):
        ri = r[i, :]
        if i < len(r) - 1:
            rj = r[i + 1, :]
        else:
            rj = r[0, :]
        if np.linalg.norm(rj) > 0 and np.linalg.norm(ri) > 0:
            aj = np.arctan2(rj[1], rj[0])
            ai = np.arctan2(ri[1], ri[0])
            da = aj - ai
            if da > np.pi:
                da = da - 2 * np.pi
            if da < -np.pi:
                da = da + 2 * np.pi
            if np.abs(da) == np.pi:
                da = 0
            a = a + da
    return a

############################  FUNCTIONS TO BUILD IMPORTANT MATRICES FOR THE INVERSION ####### 

def build_G(rs,xs,ys,l,m):
    """
    Function to build the matrix G, which relates pressures to displacements (model).

    Parameters
    ----------
    rs : ndarray
        Source mesh obtained with create_source_mesh from utils module. See the function for details.
    xs : ndarray
        x locations of the stations.
    ys : ndarray
        y locations of the stations.
    l : float
        Lame parameter.
    m : float
        Lame parameter.

    Returns
    -------
    ndarray
        Matrix G.
    """
    source_number = len(rs[:,0])*len(rs[0,:]) 
    data_number = len(xs)*3
    G = np.zeros((data_number,source_number))
    rs_formatted = rs.reshape(source_number,4,2) 
    for i in range(source_number): # we iterate on the sources = parameters to invet for       
        r = rs_formatted[i] 
        for j in range(len(xs)): #we also loop on the location of the stations, each station has 3 data (x, y and z displacement)
            xyz = [xs[j],ys[j],0] #we assume z = 0 because we record the data at the surface 
            # Translating the coordinates of the vertices so that the considered point is located at the middle
            r_translated = np.zeros(r.shape)
            r_translated[:,0] = -r[:,0] + xyz[0] 
            r_translated[:,1] = -r[:,1] + xyz[1]   
            z = xyz[2]

            # Integrals
            I1 = compute_I1(r_translated, z)
            I3 = compute_I3(r_translated, z)
            I4 = compute_I4(r_translated, z)

            # Displacements
            tmp = -I1/(4*np.pi*(l+m)) #coefficient for source i and station j for horizontal displacements (corresponds to equation 1 in the mathematical background section)
            alpha = tmp[0]
            beta  = tmp[1]
            gamma = I3/(4*np.pi*m) + I4/(4*np.pi*(l + m)) #coefficent for source  i  and station j for vertical displacement (corresponds to eq 2 in the mathematical background section)
            
            starting_idx = j*3 # 3 data per station
            G[starting_idx,i] = alpha #x coeff
            G[starting_idx+1,i] = beta #y coeff
            G[starting_idx+2,i] = gamma #z coeff
    
    return G


def kfromij(i,j,nx):
    """
    Function to find the global indices of sources to build the Laplacian/Gaussian covariance matrices.

    Parameters
    ----------
    i : int
        Index along the x-axis.
    j : int
        Index along the y-axis.
    nx : int
        Number of grid points along the x-axis.

    Returns
    -------
    int
        Global index of the source.
    """
    return j*nx+i # Global numbering to create the Laplacian


def build_laplacian(rs):    
    """
    Function that computes the Laplacian with finite difference. The Laplacian serves as the inverse covariance matrix to regularize the inversion.

    Parameters
    ----------
    rs : ndarray
        Mesh created with the function create_source_mesh from utils module. Details of the mesh are described in the documentation of that function.

    Returns
    -------
    ndarray
        Laplacian matrix.
    """
    shape = rs[:,:,0,0].shape
    dx = (rs[-1,-1,2,0]-rs[0,0,0,0])/len(rs[0,:,0,0])
    dy = (rs[-1,-1,2,1]-rs[0,0,0,1])/len(rs[:,0,0,0])
    source_number = shape[0]*shape[1]
    nx, ny = shape[1], shape[0]
    laplacien = np.zeros((source_number,source_number))
    for i in np.arange(0,nx):
        for j in np.arange(0,ny):
                k=kfromij(i,j,nx)
                laplacien[k,k]=-2/dx**2-2/dy**2 #central term
                #left term
                if i >0 :
                    kl=kfromij(i-1,j,nx)
                    laplacien[k,kl]=1/dx**2
                #right term
                if i < nx-1 :
                    kr=kfromij(i+1,j,nx)
                    laplacien[k,kr]=1/dx**2
                #bottom term
                if j >0 :
                    kb=kfromij(i,j-1,nx)
                    laplacien[k,kb]=1/dy**2
                #top term
                if j < ny-1 :
                    kt=kfromij(i,j+1,nx)
                    laplacien[k,kt]=1/dy**2
    return laplacien     


def build_gaussian_inv(rs, sigma, normalized=False):
    """"
    Builds a Gaussian matrix for regularization and returns its inverse to remain consistent with the Laplacian implementation.

    Parameters
    ----------
    rs : ndarray
        Mesh created with the function create_source_mesh from utils module. Details of the mesh are described in the documentation of that function.
    sigma : float
        Standard deviation for the Gaussian matrix.
    normalized : bool, optional
        If True, normalize the volume under the Gaussians.

    Returns
    -------
    ndarray
        Inverse of the Gaussian matrix.
    """
    shape = rs[:,:,0,0].shape
    nx, ny = shape[1], shape[0]
    source_number = nx * ny
    gaussian = np.zeros((source_number, source_number))
    for i in range(nx):
        for j in range(ny):
            k = kfromij(i, j, nx)
            for u in range(nx):
                for v in range(ny):
                    kl = kfromij(u, v, nx)
                    dx = i - u
                    dy = j - v
                    gaussian[k, kl] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
    
    if normalized==True: #normalizing the volume under the gaussians so that no matter the value of sigma the optimum value of lambda remains the same
        row_sums = gaussian.sum(axis=1)
        gaussian = gaussian / row_sums[:, np.newaxis]
        
    return np.linalg.inv(gaussian)

##########################################################

#######################################################################################################
############ FUNCTION FOR LINEAR INVERSION WITH INVERSE COVARIANCE MATRIX REGULARIZATION ############## 
#######################################################################################################

def inversion_linear(G,Us_formatted,lamb,Cm1,rs,sigma):
    """
    Perform linear inversion.

    Parameters
    ----------
    G : ndarray
        Model matrix with shape [data_number, source_number].
    Us_formatted : ndarray
        Formatted observed data with shape [data_number,].
    lamb : float
        Regularization parameter.
    Cm1 : ndarray or str
        Inverse covariance matrix; if array, use the array as the inverse covariance matrix.
        If string, determine the type of regularization: 'laplacian' or 'gaussian'.
    rs : ndarray
        Mesh of sources. Refer to function create_source_mesh from utils module for further details on the dimensions.
    sigma : float
        Standard deviation for Gaussian regularization.

    Returns
    -------
    ndarray
        Inverted load distribution.
    """
    if isinstance(Cm1, np.ndarray)==False: #checking if the covariance matrix was already provided
        if Cm1=='laplacian' or Cm1=='Laplacian': #if not provided it will create the covariance matrix according the string
            Cm1 =  build_laplacian(rs)
        elif Cm1=='Gaussian' or Cm1=='gaussian':
            Cm1 = build_gaussian_inv(rs,sigma)
    GTG = G.T@G
    GTD = G.T@Us_formatted
    ps = np.linalg.solve(GTG+lamb*Cm1.T@Cm1,GTD) #linear inversion, lamb is the lambda in the formulation and Cm1 is the inverse covariance matrix
    return ps
    
################################################################################
#########################" FUNCTIONS FOR NON LINEAR INVERSION ################## 
#####################################################################

#"################ FUNCTIONS FOR THE MISFIT AND ITS GRADIENT ####
def compute_cost(ps,Us_obs,G,lamb_cov_inv):
    """
    Compute the cost/misfit function for the inversion.

    Parameters
    ----------
    ps : ndarray
        Load distribution.
    Us_obs : ndarray
        Observed displacements.
    G : ndarray
        Model matrix.
    lamb_cov_inv : ndarray
        Precomputed term involving regularization parameter and inverse covariance matrix (lambda * Cm1.T@Cm1).

    Returns
    -------
    float
        Cost value.
    """
    return np.sum((Us_obs-np.matmul(G,ps))**2)

def compute_grad_misfit(ps,Us_obs,G,lamb_cov_inv):
    """
    Compute the gradient of the misfit for the inversion.

    Parameters
    ----------
    ps : ndarray
        Load distribution.
    Us_obs : ndarray
        Observed displacements.
    G : ndarray
        Model matrix.
    lamb_cov_inv : ndarray
        Precomputed term involving regularization parameter and inverse covariance matrix (lambda * Cm1.T@Cm1).

    Returns
    -------
    ndarray
        Gradient of the misfit.
    """
    return np.matmul(G.T,(np.matmul(G,ps)-Us_obs)) + np.matmul(lamb_cov_inv,ps)
#########################################################################


################## LINESEARCH FUNCTIONS ########################

def linesearch(alpha,Us_obs,G,ps,delta_m,lamb_cov_inv,constraint_idx):
    """
    Perform linesearch to find an appropriate step size (alpha) for optimization.

    Parameters
    ----------
    alpha : float
        Initial guess for the step size.
    Us_obs : ndarray
        Observed data with shape [data_number,].
    G : ndarray
        Model matrix with shape [data_number, source_number].
    ps : ndarray
        Current estimated load distribution.
    delta_m : ndarray
        Descent direction to update ps.
    lamb_cov_inv : ndarray
        Precomputed term lambda * Cm1.T@Cm1.

    Returns
    -------
    float or None
        Found step size (alpha) if conditions are met, otherwise None.
    """
    c1, c2 = 1e-4, 0.9
    cond2 = False
    grad  = compute_grad_misfit(ps,Us_obs,G,lamb_cov_inv)
    it = 0
    if isinstance(constraint_idx,np.ndarray)==True:
        grad[constraint_idx] = 0 #if we provided bounds then we make sure the gradient is null for parameters with bounds, because lower bound = upper bound
    while cond2==False and it<20:
        it +=1
        if compute_cost(ps+alpha*delta_m,Us_obs,G,lamb_cov_inv) <= compute_cost(ps,Us_obs,G,lamb_cov_inv) + c1*alpha*np.dot(grad,delta_m):
            if np.dot(compute_grad_misfit(ps=ps+alpha*delta_m,Us_obs=Us_obs,G=G,lamb_cov_inv=lamb_cov_inv),delta_m) >= c2*np.dot(grad,delta_m):
                return alpha #both conditions fulfilled, we found a good alpha
            else:
                alpha=alpha*10 #curvature condition was not met, our alpha is too small
        else:
            alpha=alpha/2 #sufficient decrease conditon was not met, our alpha is too big
    return None #did not succeed to find a good value for alpha within the provided maximum number of iterations


def linesearch_armijo(alpha,Us_obs,G,ps,delta_m,lamb_cov_inv,constraint_idx):
    """
    Perform linesearch with Armijo condition to find an appropriate step size (alpha) for optimization if the linesearch with both Wolfe conditions failed.

    Parameters
    ----------
    alpha : float
        Initial guess for the step size.
    Us_obs : ndarray
        Observed data with shape [data_number,].
    G : ndarray
        Model matrix with shape [data_number, source_number].
    ps : ndarray
        Current estimated load distribution.
    delta_m : ndarray
        Descent direction to update ps.
    lamb_cov_inv : ndarray
        Precomputed term lambda * Cm1.T@Cm1.

    Returns
    -------
    float or None
        Found step size (alpha) if the Armijo condition is met, otherwise None.
    """
    c1 = 1e-4
    cond = False
    grad  = compute_grad_misfit(ps,Us_obs,G,lamb_cov_inv)
    it = 0
    if isinstance(constraint_idx,np.ndarray)==True:
        grad[constraint_idx] = 0 #if we provided bounds then we make sure the gradient is null for parameters with bounds, because lower bound = upper bound
    while cond==False:
        it +=1
        if compute_cost(ps+alpha*delta_m,Us_obs,G,lamb_cov_inv) <= compute_cost(ps,Us_obs,G,lamb_cov_inv) + c1*alpha*np.dot(grad,delta_m):
                return alpha 
        else:
            alpha=alpha/2 #armijo = sufficient decrease not met, our alpha is too big
    return None #should not happend

####################################################################


######################## FUNCTIONS FOR THE INVERSION WITH DIFFERENT ALGORTHMS ##### 

def inversion_TV(G,Us_formatted,lamb,gamma_coeff,maxit,source_number,epsilon):
    """
    Perform inversion using Total Variation (TV) regularization.

    Parameters
    ----------
    G : ndarray
        Model matrix with shape [data_number, source_number].
    Us_formatted : ndarray
        Formatted observed data with shape [data_number,1].
    lamb : float
        Regularization parameter for TV.
    gamma_coeff : float
        Coefficient for the step size (gamma) in the optimization.
    maxit : int
        Maximum number of iterations for optimization.
    source_number : int
        Number of sources.
    epsilon : float
        Convergence criterion.

    Returns
    -------
    ndarray
        Inverted load distribution.
    int
        Number of iterations performed.
    """
    Gstar = G #renaming the variable to be more consistant with the algorithm from Mallat (in book G = \phi*) -> in algo renamed G
    G = Gstar.T # equivalent to G^T in algo  
    ps = np.zeros((source_number,1)) #x in the algo
    b = G@Us_formatted #equivalent to G^T d_obs in algo 
    T = lamb #threshold for the soft thresholding -> the higher the threshold the lower the smoothing 
    hessian = G@Gstar #equivalent to G^T G
    gamma = gamma_coeff/np.max(np.abs(hessian)) #gamma is the step -> same as in algo   and gamma coef is the xi in the algo 
    k = 0
    conv = False
    while conv==False and k<maxit:#
        ps_old = ps.copy() #keeping the old values to compare with the new ones and check for convergence
        #gradient descent
        ps_bar = ps+gamma*(b-hessian@ps) # x bar in the algo
        #soft thresholding 
        for i in range(source_number):
            ps[i,0] = ps_bar[i,0]*np.max([1-(gamma*T)/np.abs(ps_bar[i,0]),0]) #computing x_{k+1} in the algo,  
        res = ((ps-ps_old).T@(ps-ps_old))[0][0] #checking convergence
        if  res < epsilon: 
            conv = True
        k += 1 #incrementing iteration
    return ps,k


def inversion_steepest(G,Us,lamb,Cm1,rs,maxit,source_number,data_number,constraint,constraint_idx,epsilon,sigma):
    """
    Perform inversion using steepest descent optimization.

    Parameters
    ----------
    G : ndarray
        Model matrix with shape [data_number, source_number].
    Us : ndarray
        Observed data with shape either [data_number,] or [station_number,3].
    lamb : float
        Regularization parameter.
    Cm1 : ndarray or str
        Inverse covariance matrix; if array, use the array as the inverse covariance matrix.
        If string, determine the type of regularization: 'laplacian' or 'gaussian' and uses it build it.
    rs : ndarray
        Mesh of sources, refer to function create_source_mesh from utils module to know how it is built.
    maxit : int
        Maximum number of iterations for optimization.
    source_number : int
        Number of sources (parameters to invert).
    data_number : int
        Number of data points.
    constraint : ndarray
        1D array containing constraint values for specific source indices.
    constraint_idx : ndarray
        Indices of sources with constraints (bounds).
    epsilon : float
        Convergence criterion.
    sigma : float
        Standard deviation for Gaussian regularization.

    Returns
    -------
    ndarray
        Inverted load distribution.
    int
        Number of iterations performed.
    """
    Us_formatted = Us.reshape((data_number,)) #reshaping data to be match the following implementations that us shape n, or m,
    if isinstance(Cm1, np.ndarray)==False: #on apeut déjà avoir créé au préalable la matrice de covariance 
        if Cm1=='laplacian' or Cm1=='Laplacian':
            Cm1 =  build_laplacian(rs) #creating the covariance matrix in case it was not provided
        elif Cm1=='Gaussian' or Cm1=='gaussian':
            Cm1 = build_gaussian_inv(rs,sigma)
    lamb_cov_inv = lamb*Cm1.T@Cm1 #precomputing
    alpha = 1
    conv = False
    i = 0
    ps = np.ones((source_number,))
    if isinstance(constraint,np.ndarray)==True:
        ps[constraint_idx] = constraint[constraint_idx] #adding the constraint  -> bound
    while conv==False and i<maxit:
        i +=1
        delta_m = -compute_grad_misfit(ps=ps,Us_obs=Us_formatted,G=G,lamb_cov_inv=lamb_cov_inv) # - nabla f_k in algo
        if isinstance(constraint,np.ndarray)==True: # we set to 0 the derivative of parameters that need to be constrained because we use lower bound=  upper bound
            delta_m[constraint_idx] = 0
        alpha_old = alpha #use the last alpha as the  first guess 
        alpha = linesearch(alpha=alpha_old,Us_obs=Us_formatted,G=G,ps=ps,delta_m=delta_m,lamb_cov_inv=lamb_cov_inv,constraint_idx=constraint_idx) #we added a condition to max 20 iterations to find alpha
        if alpha==None: #in case linesearch with wolfe conditions failed to find a good lambda, we try again with only the armijo condition 
            alpha = linesearch_armijo(alpha=alpha_old,Us_obs=Us_formatted,G=G,ps=ps,delta_m=delta_m,lamb_cov_inv=lamb_cov_inv,constraint_idx=constraint_idx) #mettre juste condition d'armijo si le strong wolfe n'arrive pas à converger
        if alpha<1e-10:
            alpha=1e-10
        ps += alpha*delta_m #x_{k+1}
        if np.dot(alpha*delta_m,alpha*delta_m) < epsilon: #checking for convergence
            conv = True
    return ps, i

def inversion_nlcg(G,Us,lamb,Cm1,rs,maxit,source_number,data_number,constraint,constraint_idx,epsilon,sigma):
    """
    Perform inversion using NLCG optimization.

    Parameters
    ----------
    G : ndarray
        Model matrix with shape [data_number, source_number].
    Us : ndarray
        Observed data with shape either [data_number,] or [station_number,3].
    lamb : float
        Regularization parameter.
    Cm1 : ndarray or str
        Inverse covariance matrix; if array, use the array as the inverse covariance matrix.
        If string, determine the type of regularization: 'laplacian' or 'gaussian' and uses it build it.
    rs : ndarray
        Mesh of sources, refer to function create_source_mesh from utils module to know how it is built.
    maxit : int
        Maximum number of iterations for optimization.
    source_number : int
        Number of sources (parameters to invert).
    data_number : int
        Number of data points.
    constraint : ndarray
        1D array containing constraint values for specific source indices.
    constraint_idx : ndarray
        Indices of sources with constraints (bounds).
    epsilon : float
        Convergence criterion.
    sigma : float
        Standard deviation for Gaussian regularization.

    Returns
    -------
    ndarray
        Inverted load distribution.
    int
        Number of iterations performed.
    """
    Us_formatted = Us.reshape((data_number,))
    ps = np.ones((source_number,))
    if isinstance(constraint,np.ndarray)==True:
        ps[constraint_idx] = constraint[constraint_idx] #adding the constraint 
    if isinstance(Cm1, np.ndarray)==False: #on apeut déjà avoir créé au préalable la matrice de covariance 
        if Cm1=='laplacian' or Cm1=='Laplacian':
            Cm1 =  build_laplacian(rs)
        elif Cm1=='Gaussian' or Cm1=='gaussian':
            Cm1 = build_gaussian_inv(rs,sigma)
    lamb_cov_inv = lamb*Cm1.T@Cm1
    rk = compute_grad_misfit(ps=ps,Us_obs=Us_formatted,G=G,lamb_cov_inv=lamb_cov_inv)
    pk = -rk #first direction is identical to steepest descent 
    if isinstance(constraint,np.ndarray)==True: # we set to 0 the derivative of parameters that need to be constrained because we use lower bound=  upper bound
            pk[constraint_idx] = 0
    conv = False
    alpha = 1
    i = 0 #k in the algo
    while conv==False and i<maxit:
        rkTrk = np.dot(rk,rk) #keeping the grad(f_k)^Tgrad(f_k) 
        alpha_old = alpha #
        alpha = linesearch(alpha=alpha_old,Us_obs=Us_formatted,G=G,ps=ps,delta_m=pk,lamb_cov_inv=lamb_cov_inv,constraint_idx=constraint_idx)
        if alpha==None:
            alpha = linesearch_armijo(alpha=alpha_old,Us_obs=Us_formatted,G=G,ps=ps,delta_m=pk,lamb_cov_inv=lamb_cov_inv,constraint_idx=constraint_idx) #only using armijo if curvature condition fails
        ps += alpha*pk #computing the x_{k+1}  
        rk = compute_grad_misfit(ps=ps,Us_obs=Us_formatted,G=G,lamb_cov_inv=lamb_cov_inv) # grad(f_{k+1}) in the algo
        beta = np.dot(rk,rk)/rkTrk # same as in the algo, it's beta 
        pk = -rk + beta*pk #computing the conjugated direction that will be used in the next iteration to compute x_{k+1} as in the algo
        if isinstance(constraint,np.ndarray)==True: # again setting the derivative to 0 for parameters that are bounded
            pk[constraint_idx] = 0
        i += 1 #increment
        if np.dot(alpha*pk,alpha*pk) < epsilon: #checking for convergence
            conv = True
    return ps,i


def inversion_CG_scipy(G,Us,lamb,Cm1,rs,maxit,source_number,data_number,constraint,constraint_idx,epsilon,sigma):
    """
    Wrapper for scipy CG inversion function. Scipy does not support bounds, so the provided bounds are only used to adapt the initial guess.
    
    Parameters
    ----------
    G : ndarray
        Model matrix with shape [data_number, source_number].
    Us : ndarray
        Observed data with shape either [data_number,] or [station_number,3].
    lamb : float
        Regularization parameter.
    Cm1 : ndarray or str
        Inverse covariance matrix; if array, use the array as the inverse covariance matrix.
        If string, determine the type of regularization: 'laplacian' or 'gaussian' and uses it build it.
    rs : ndarray
        Mesh of sources, refer to function create_source_mesh from utils module to know how it is built.
    maxit : int
        Maximum number of iterations for optimization.
    source_number : int
        Number of sources (parameters to invert).
    data_number : int
        Number of data points.
    constraint : ndarray
        1D array containing constraint values for specific source indices.
    constraint_idx : ndarray
        Indices of sources with constraints (bounds).
    epsilon : float
        Convergence criterion.
    sigma : float
        Standard deviation for Gaussian regularization.

    Returns
    -------
    ndarray
        Inverted load distribution.
    int
        Number of iterations performed.
    """
    Us_formatted = Us.reshape((data_number,)) #formatting the arrays because scipy only accepts 1D arrays
    ps = np.ones((source_number,)) #first guess -> x0 in the algorithm
    if isinstance(constraint,np.ndarray)==True:
        ps[constraint_idx] = constraint[constraint_idx] #adding the constraint 
    if isinstance(Cm1, np.ndarray)==False: #checks if a covariance matrix was already provided
        if Cm1=='laplacian' or Cm1=='Laplacian':
            Cm1 =  build_laplacian(rs)
        elif Cm1=='Gaussian' or Cm1=='gaussian':
            Cm1 = build_gaussian_inv(rs,sigma)
    lamb_cov_inv = lamb*Cm1.T@Cm1
    result = minimize(fun=compute_cost, x0=ps, args=(Us_formatted,G,lamb_cov_inv), method='CG', tol=epsilon, jac=compute_grad_misfit,
                      options={'maxiter': maxit})
    ps = result.x
    i = result.nit
    return ps,i

####################### LBFGS ####################

def lbfgs_direction(grad, s_hist, y_hist, rho_hist, m):
    """
    Handles the memory for the LBFGS algorithm.

    Parameters
    ----------
    grad : ndarray
        Gradient of the misfit.
    s_hist : list
        List of previous s vectors.
    y_hist : list
        List of previous y vectors.
    rho_hist : list
        List of previous rho values.
    m : int
        Memory parameter for LBFGS algorithm.

    Returns
    -------
    ndarray
        LBFGS direction vector.
    """
    q = grad.copy()
    alpha = [0] * m #adapting the size of alpha to also work when k is smaller than m
    for i in range(len(s_hist) - 1, -1, -1):
        alpha[i] = rho_hist[i] * np.dot(s_hist[i], q)
        q -= alpha[i] * y_hist[i]
    gk = np.dot(s_hist[-1], y_hist[-1]) / np.dot(y_hist[-1], y_hist[-1]) #gamma k used to compute H0k so that when doing linesearch alpha = 1 is often accepted as explained in Nocedal
    r = gk * q #we chose H0 = gamma k*Identity  so r is just gk * q  
    for i in range(len(s_hist)):
        beta = rho_hist[i] * np.dot(y_hist[i], r)
        r += (alpha[i] - beta) * s_hist[i]
    return r
        
def inversion_lbfgs(G,Us,lamb,Cm1,rs,maxit,source_number,data_number,constraint,constraint_idx,epsilon,sigma):
    """
    Perform inversion using L-BFGS optimization.

    Parameters
    ----------
    G : ndarray
        Model matrix with shape [data_number, source_number].
    Us : ndarray
        Observed data with shape either [data_number,] or [station_number,3].
    lamb : float
        Regularization parameter.
    Cm1 : ndarray or str
        Inverse covariance matrix; if array, use the array as the inverse covariance matrix.
        If string, determine the type of regularization: 'laplacian' or 'gaussian' and uses it build it.
    rs : ndarray
        Mesh of sources, refer to function create_source_mesh from utils module to know how it is built.
    maxit : int
        Maximum number of iterations for optimization.
    source_number : int
        Number of sources (parameters to invert).
    data_number : int
        Number of data points.
    constraint : ndarray
        1D array containing constraint values for specific source indices.
    constraint_idx : ndarray
        Indices of sources with constraints (bounds).
    epsilon : float
        Convergence criterion.
    sigma : float
        Standard deviation for Gaussian regularization.

    Returns
    -------
    ndarray
        Inverted load distribution.
    int
        Number of iterations performed.
    """
    Us_formatted = Us.reshape((data_number,))
    ps = np.ones((source_number,))
    if isinstance(constraint,np.ndarray)==True:
        ps[constraint_idx] = constraint[constraint_idx] #adding the constraint 
    if isinstance(Cm1, np.ndarray)==False: #on apeut déjà avoir créé au préalable la matrice de covariance 
        if Cm1=='laplacian' or Cm1=='Laplacian':
            Cm1 =  build_laplacian(rs)
        elif Cm1=='Gaussian' or Cm1=='gaussian':
            Cm1 = build_gaussian_inv(rs,sigma)
    lamb_cov_inv = lamb*Cm1.T@Cm1
    i = 0 #using i as our index instead of k from the algorithm, be careful it's not the same as the i in the loops to estimate the approximation of the inverse of the Hessian 
    m = 5 #number of s and y pairs to keep to act as the hessian estimate, 5 is supposed to be enough based on Nocedal's book
    conv = False
    s_hist = []
    y_hist = []
    rho_hist = []
    q = compute_grad_misfit(ps=ps,Us_obs=Us_formatted,G=G,lamb_cov_inv=lamb_cov_inv) #same name as in the algo
    while conv==False and i<maxit:
        if i == 0:
            pk = -q #first iteration we don't have previous descent direction, so we simply do a steepest descent iteration, like in algo
        else:
            pk = -lbfgs_direction(q, s_hist, y_hist, rho_hist, m) #performs the double loop in the algo to get H_k grad(f_k) = r in the algo
        
        if isinstance(constraint,np.ndarray)==True:
            pk[constraint_idx] = 0 #we make sure the update on the parameters is null for bounded ones since we use identical lower and upper bounds
            
        alpha = linesearch(alpha=1,Us_obs=Us_formatted,G=G,ps=ps,delta_m=pk,lamb_cov_inv=lamb_cov_inv,constraint_idx=constraint_idx) #start with alpha=1 because we use the scaler where H0k = gk I
        if alpha==None:
            alpha = linesearch_armijo(alpha=1,Us_obs=Us_formatted,G=G,ps=ps,delta_m=pk,lamb_cov_inv=lamb_cov_inv,constraint_idx=constraint_idx) #mettre juste condition d'armijo si le strong wolfe n'arrive pas à converger
        if alpha<1e-10:
            alpha=1e-10 #if still managed to fail to estimate alpha then we use this ad hoc value 
        ps_old = ps.copy()
        ps += alpha*pk
        sk = ps-ps_old #vector with shape (source_number,) same as sk in algo
        q_old = q.copy()
        q = compute_grad_misfit(ps=ps,Us_obs=Us_formatted,G=G,lamb_cov_inv=lamb_cov_inv)
        yk = q-q_old #vecotr shape (source_number,), same as yk in the algo
        if np.linalg.norm(yk) < epsilon: #we use this norm for convergence criterion otherwise we have trouble as we may divide by 0.
            break
        rho_k = 1 / np.dot(yk, sk) #rho_k in algo
        if len(s_hist) == m: #discarding the oldest pair of s and y  with the computed rho
            s_hist.pop(0)
            y_hist.pop(0)
            rho_hist.pop(0)
        s_hist.append(sk) #and appending the new ones
        y_hist.append(yk)
        rho_hist.append(rho_k)
        i += 1 # k = k+1 in algo
    return ps, i


#######################  LBFGS WITH SCIPY  #####################

def inversion_lfbgs_scipy(G,Us,lamb,Cm1,rs,maxit,source_number,data_number,constraint,constraint_idx,epsilon,sigma):
    """
    Wrapper for inversion using the L-BFGS optimization with scipy.

    Parameters
    ----------
    G : ndarray
        Model matrix with shape [data_number, source_number].
    Us : ndarray
        Observed data with shape either [data_number,] or [station_number,3].
    lamb : float
        Regularization parameter.
    Cm1 : ndarray or str
        Inverse covariance matrix; if array, use the array as the inverse covariance matrix.
        If string, determine the type of regularization: 'laplacian' or 'gaussian' and uses it build it.
    rs : ndarray
        Mesh of sources, refer to function create_source_mesh from utils module to know how it is built.
    maxit : int
        Maximum number of iterations for optimization.
    source_number : int
        Number of sources (parameters to invert).
    data_number : int
        Number of data points.
    constraint : ndarray
        1D array containing constraint values for specific source indices.
    constraint_idx : ndarray
        Indices of sources with constraints (bounds).
    epsilon : float
        Convergence criterion.
    sigma : float
        Standard deviation for Gaussian regularization.

    Returns
    -------
    ndarray
        Inverted load distribution.
    int
        Number of iterations performed.
    """
    Us_formatted = Us.reshape((data_number,))
    ps = np.ones((source_number,))
    if isinstance(Cm1, np.ndarray)==False: #checking if an inverse covariance matrix was provided 
        if Cm1=='laplacian' or Cm1=='Laplacian': #if not then we need to create the inverse covariance matrix
            Cm1 =  build_laplacian(rs)
        elif Cm1=='Gaussian' or Cm1=='gaussian':
            Cm1 = build_gaussian_inv(rs,sigma)
    lamb_cov_inv = lamb*Cm1.T@Cm1 # lambda Cm-1 in the algo 
    if isinstance(constraint,np.ndarray)==True: #checking if a constraint was added
        ps[constraint_idx] = constraint[constraint_idx] #we make the first guess start with the correct value
        bounds = create_bounds(constraint)#we create bounds equal to our constraints for the specific ps values that need it, that function is a wrapper to convert our                   constraints to constraints accepted by scipy   
        result = minimize(fun=compute_cost, x0=ps, args=(Us_formatted,G,lamb_cov_inv), method='L-BFGS-B', tol=epsilon, jac=compute_grad_misfit,
                     bounds=bounds,options={'maxiter': maxit})
    else:
        result = minimize(fun=compute_cost, x0=ps, args=(Us_formatted,G,lamb_cov_inv), method='L-BFGS-B', tol=epsilon, jac=compute_grad_misfit,
                         options={'maxiter': maxit})
    ps = result.x
    i = result.nit
    return ps, i
        

def create_bounds(constraint):    
    """
    Wrapper to build the bounds from the constraint vector to be used with scipy minimization approaches.

    Parameters
    ----------
    constraint : ndarray
        1D array containing either a number (float/int) to serve as the upper and lower bound for the associated source load,
        or a np.nan if no bound is provided.

    Returns
    -------
    list
        List of lower and upper bounds to feed to scipy minimize function.
    """
    bounds = []
    for i in range(len(constraint)):
        if np.isnan(constraint[i])==False:
            bounds.append((constraint[i],constraint[i]))
        else:
            bounds.append((None,None))
    return bounds


        
####################################################### WRAPPER FOR THE INVERSION ###################

def disp2load(E,v,rs,xs, ys, Us, mode=None, lamb=1, epsilon=1e-2, gamma_coeff=0.1,sigma=1,G=None,Cm1=None,
             constraint=None,maxit=1e6,verbose=0,logpath=None):
    '''
    Function that serves as a wrapper for functions that perform inversion with linear and non-linear schemes using different regularization methods.
    
    Parameters
    ----------
    E : float
        Young's Modulus.
    v : float
        Poisson's ratio.
    rs : ndarray
        Mesh of sources created with the function create_source_mesh from utils module.
    xs, ys : ndarray
        x and y coordinates of the stations.
    Us : ndarray
        Displacements recorded by the stations with shape [n, 3], where n is the number of stations.
    mode : str, optional
        Type of inversion; options are 'linear', 'lbfgs', 'lbfgs_fast', 'tv', 'nlcg', 'nlcg_fast', or None (no regularization and linear inversion).
    lamb : float, optional
        Coefficient for the regularizer term.
    epsilon : float, optional
        Convergence criterion.
    gamma_coeff : float, optional
        Coefficient for the step size gamma in the soft-thresholding implementation.
    sigma : float, optional
        Standard deviation, required if Cm1 = 'gaussian'.
    G : ndarray, optional
        If array, use the array for computation; if None, then G is built with the function build_G.
    Cm1 : ndarray or str, optional
        Inverse covariance matrix; if array, use the array as the inverse covariance matrix. If string, determine the type of regularization: 'laplacian' or 'gaussian'.
    constraint : ndarray, optional
        2D array that has the same shape as the load distribution inverted and contains floats and nan values. Floats constrain the inverted value of the sources while NaNs don't.
    maxit : int, optional
        Maximum number of iterations allowed for the non-linear inversion schemes.
    verbose : int, optional
        Verbosity level.
    logpath : str, optional
        Either None for no logging or a path where to log the runs.

    Returns
    -------
    ndarray
        Inverted load distribution.
    '''
    # Computing the Lame constants
    l = E * v / ((1 + v) * (1 - 2 * v))
    m = E / (2 * (1 + v))    
    
    data_number = len(Us[:,0])*len(Us[0,:])
    source_number = len(rs[:,0,0,0])*len(rs[0,:,0,0])
    
    Us_formatted = Us.reshape((data_number,1)) # trois premier les éléments de la première ligne, puis 3 suivant ce sont les 3 de la seconde ligne..
    if isinstance(G, np.ndarray)==False: #checking if G already precomputed and uses it if so : time gain for L-curve...
        G = build_G(rs,xs,ys,l,m)
    
    if isinstance(constraint,np.ndarray)==True: #checking if there are constraints provided 
        constraint = constraint.reshape(source_number) #reshaping the constraint to match that of ps = inverted parameters vector (x in the algo)
        constraint_idx = np.where(np.isnan(constraint)==False)[0] #the constraint array contains floats and nan values, we keep in memory the locaiton of the non nan idx 
    else:
        constraint_idx = None
        
    t1 = time.time() # used to log the run
    ############################### INVERSION STARTING HERE ######################
    
    if mode==None: #without regularization and linear inversion   
        ps = np.linalg.solve(G.T@G,G.T@Us_formatted) #Us_formatted is the data vector
    elif mode.lower()=='linear': #linear inversion using a chosen Covariance matrix for the regularisation
        ps = inversion_linear(G,Us_formatted,lamb,Cm1,rs,sigma)
    #################################################################################################################################
    elif mode.lower()=='tv': #regularization with Total variation (TV) -> non-linear scheme 
        ps, i = inversion_TV(G,Us_formatted,lamb,gamma_coeff,maxit,source_number,epsilon)
    #################################################################################################################################
    elif mode.lower()=='steepest': #Steepest descent 
        ps, i = inversion_steepest(G,Us,lamb,Cm1,rs,maxit,source_number,data_number,constraint,constraint_idx,epsilon,sigma)
    #################################################################################################################################
    elif mode.lower()=='nlcg': #NLCG
        ps, i = inversion_nlcg(G,Us,lamb,Cm1,rs,maxit,source_number,data_number,constraint,constraint_idx,epsilon,sigma)
    #################################################################################################################################
    elif mode.lower()=='cg_fast': #Conjugate gradient from scipy 
        ps, i = inversion_CG_scipy(G,Us,lamb,Cm1,rs,maxit,source_number,data_number,constraint,constraint_idx,epsilon,sigma)
    #################################################################################################################################
    elif mode.lower()=='l-bfgs' or mode.lower()=='lbfgs': #mode L-BFGS
        ps, i = inversion_lbfgs(G,Us,lamb,Cm1,rs,maxit,source_number,data_number,constraint,constraint_idx,epsilon,sigma)
    #################################################################################################################################
    elif mode.lower()=='l-bfgs_fast' or mode.lower()=='lbfgs_fast': #mode L-BFGS using scipy
        ps, i = inversion_lfbgs_scipy(G,Us,lamb,Cm1,rs,maxit,source_number,data_number,constraint,constraint_idx,epsilon,sigma)
    
    if logpath!=None: #can log various parameters of the run 
        elapsed_time = time.time()-t1
        if mode.lower()=='linear' or mode==None:
            i = None
        utils.log_run(E,v,lamb,epsilon,sigma,constraint,i,maxit,elapsed_time,logpath)
    
    ps = ps.reshape(rs[:,:,0,0].shape) #reshaping to go from a parameter vector to a 2D parameter array as it initially was.
    return ps
