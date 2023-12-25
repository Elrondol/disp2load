import numpy as np
from scipy.integrate import quad
from numpy import arctan, tan, cos, sin, arccos, arcsin, sqrt, log, log10, pi
from scipy.optimize import minimize
import time
import json

def compute_I1(r, z):
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
    S2i = log((a * t1 + np.sqrt(a * (a * t1**2 + e))) / (a * t0 + np.sqrt(a * (a * t0**2 + e)))) / sqrt(a)
    return S2i

def compute_S3i(a, A, B, t0, t1):
    S3i = arctan(t1 * sqrt((B - A) / (A * (B + t1**2)))) - arctan(t0 * sqrt((B - A) / (A * (B + t0**2))))
    S3i = S3i * sqrt((B - A) / A) + log((t1 + sqrt(B + t1**2)) / (t0 + sqrt(B + t0**2))) / sqrt(a)
    return S3i


def alpha(r):
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


def build_G(rs,xs,ys,l,m):
    """Fonction pour build G, la matrice qui relie les pressions au displacements
    :input: rs = source mesh obtained with create mesh
            xs, ys, = x and y locaiton of the station, y is assumed to be 0 (displacement recorded at the surface)
            l,m = Lame parameters"""
    source_number = len(rs[:,0])*len(rs[0,:]) 
    data_number = len(xs)*3
    G = np.zeros((data_number,source_number))
    rs_formatted = rs.reshape(source_number,4,2) 
    for i in range(source_number): # on remplit à présent la matrice G et pour ça on a besoin d'itérer sur les sources        
        r = rs_formatted[i] 
        for j in range(len(xs)): #on doit aussi boucler sur les positions de la station; 1 station fournit 3 data
            xyz = [xs[j],ys[j],0] #z = 0 on présume car c'est des enregistrements à la surface et qu'il accepte pas des altitutdes positives 
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
            tmp = -I1/(4*np.pi*(l+m)) #coefficient pour la source i et pour la station j pour les déplacements horizontaux
            alpha = tmp[0]
            beta  = tmp[1]
            gamma = I3/(4*np.pi*m) + I4/(4*np.pi*(l + m)) #coefficent pour la source i  et la station j pour les déplacements verticaux
            
            starting_idx = j*3 #car 3 données par station
            G[starting_idx,i] = alpha
            G[starting_idx+1,i] = beta
            G[starting_idx+2,i] = gamma
    
    return G


def kfromij(i,j,nx): # fonction pour trouver les indices globaux des sources car les m sont regroupés dans un vecteur 
    return j*nx+i #numerotation globale pour crééer le laplacien


def build_laplacian(rs):
    """Fonction qui compute le laplacien par différence finie servant pour la régularization
    :input: rs = mesh created with create mesh"""
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
    """Builds a Gaussian matrix for regularization and returns its inverse to remain consistant with the Laplacian implementation"""
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



##############" POUR INVERISON NON LINEAIRE ####### 
def linesearch(alpha,Us_obs,G,ps,delta_m,lamb_cov_inv):
    c1, c2 = 1e-4, 0.9
    cond2 = False
    grad  = compute_grad_misfit(ps,Us_obs,G,lamb_cov_inv)
    it = 0 
    while cond2==False and it<20:
        it +=1
        if compute_cost(ps+alpha*delta_m,Us_obs,G,lamb_cov_inv) <= compute_cost(ps,Us_obs,G,lamb_cov_inv) + c1*alpha*np.dot(grad,delta_m):
            if np.dot(compute_grad_misfit(ps=ps+alpha*delta_m,Us_obs=Us_obs,G=G,lamb_cov_inv=lamb_cov_inv),delta_m) >= c2*np.dot(grad,delta_m):
                return alpha
            else:
                alpha=alpha*10
        else:
            alpha=alpha/2
    return None #pas réussi à trouver de bon alpha en suffisement peut de steps


def linesearch_armijo(alpha,Us_obs,G,ps,delta_m,lamb_cov_inv):
    c1, c2 = 1e-4, 0.9
    cond = False
    grad  = compute_grad_misfit(ps,Us_obs,G,lamb_cov_inv)
    it = 0 
    while cond==False:
        it +=1
        if compute_cost(ps+alpha*delta_m,Us_obs,G,lamb_cov_inv) <= compute_cost(ps,Us_obs,G,lamb_cov_inv) + c1*alpha*np.dot(grad,delta_m):
                return alpha
        else:
            alpha=alpha/2
    return None #pas réussi à trouver de bon alpha en suffisement peut de steps


def compute_cost(ps,Us_obs,G,lamb_cov_inv):
    return np.sum((Us_obs-np.matmul(G,ps))**2)

def compute_grad_misfit(ps,Us_obs,G,lamb_cov_inv):
    return np.matmul(G.T,(np.matmul(G,ps)-Us_obs)) + np.matmul(lamb_cov_inv,ps)


def create_bounds(constraint):
    '''Function to build the bounds from the constraint vector to be used with scipy minimization approaches
    -> only used for L-BFGS because scipy doesn't have NLCG and CG doesn't accept bounds...'''
    bounds = []
    for i in range(len(constraint)):
        if np.isnan(constraint[i])==False:
            bounds.append((constraint[i],constraint[i]))
        else:
            bounds.append((None,None))
    return bounds

def lbfgs_direction(grad, s_hist, y_hist, rho_hist, m):
    '''Handles the memory for the LBFGS algorithm'''
    q = grad.copy()
    alpha = [0] * m #adapting the size of alpha to also work when k is smaller than m
    for i in range(len(s_hist) - 1, -1, -1):
        alpha[i] = rho_hist[i] * np.dot(s_hist[i], q)
        q -= alpha[i] * y_hist[i]
    gk = np.dot(s_hist[-1], y_hist[-1]) / np.dot(y_hist[-1], y_hist[-1])  #gamma k  used to compute H0k so that when doing linesearch alpha = 1 is often accepted as explained in Nocedal
    r = gk * q #we chose H0 = gamma k*Identity  so r is just gk * q  
    for i in range(len(s_hist)):
        beta = rho_hist[i] * np.dot(y_hist[i], r)
        r += (alpha[i] - beta) * s_hist[i]
    return r

###### logging function ##### -> saves files to easily retrieve run information for complementary analysis

def log_run(E,v,lamb,epsilon,sigma,constraint,it,maxit,elapsed_time,logpath):
    """ Creates a log file for the inversion """
    #not gonna put the coordinates in the file assumed to be known because we always use the same G matrix.
    if isinstance(constraint,np.ndarray)==True:
        constraint = True #donc soit None soit True
    dic = {
    'E': E,
    'v': v,
    'epsilon': epsilon,
    'lamb' : lamb, 
    'sigma' : sigma,
    'constraint' : constraint,
    'iterations' : it,
    'maxit' : maxit,
    'elapsed_time' : elapsed_time
}
    filename = f'{logpath}/meta_sig_{sigma}_lamb_{lamb}.json'    
    with open(filename, 'w') as file:
        json.dump(dic, file)

#######################################################"

def disp2load(E,v,rs,xs, ys, Us, mode=None, lamb=1, epsilon=1e-2, gamma_coeff=0.1,sigma=1,G=None,Cm1=None,
             constraint=None,maxit=1e6,verbose=0,logpath=None):
    '''Need to provide the fucntion the shape of ps expected and Us the data in the format [[x1,y1,z1],
                                                                                            [x2,y2,z2],
                                                                                            ...
                                                                                            [xn,yn,zn]]
                                                                                            
            For  mode=1  alpha is the value of lmabda, the coefficient of the regularizer                                                                        '''
    # Computing the Lame constants
    l = E * v / ((1 + v) * (1 - 2 * v))
    m = E / (2 * (1 + v))    
    
    data_number = len(Us[:,0])*len(Us[0,:])
    source_number = len(rs[:,0,0,0])*len(rs[0,:,0,0])
    
    Us_formatted = Us.reshape((data_number,1)) # trois premier les éléments de la première ligne, puis 3 suivant ce sont les 3 de la seconde ligne..
    if isinstance(G, np.ndarray)==False: #checking if G already precomputed and uses it if so : massive time gain for L-curve...
        G = build_G(rs,xs,ys,l,m)
    
    if isinstance(constraint,np.ndarray)==True: #checking if there are constraints 
        constraint = constraint.reshape(source_number) #reshaping the constraint to match that of ps = inverted parameters vector
        constraint_idx = np.where(np.isnan(constraint)==False)[0]
    
    t1 = time.time() #
    ############################### INVERSION STARTING HERE ######################
    
    if mode==None: #without regularization    
        ps = np.linalg.solve(G.T@G,G.T@Us_formatted)
        
    elif mode.lower()=='linear': #linear inversion -> non non linear approach 
        if isinstance(Cm1, np.ndarray)==False: #on apeut déjà avoir créé au préalable la matrice de covariance 
            if Cm1=='laplacian' or Cm1=='Laplacian':
                Cm1 =  build_laplacian(rs)
            elif Cm1=='Gaussian' or Cm1=='gaussian':
                Cm1 = build_gaussian_inv(rs,sigma)
        GTG = G.T@G
        GTD = G.T@Us_formatted
        ps = np.linalg.solve(GTG+lamb*Cm1.T@Cm1,GTD)
    #################################################################################################################################
    elif mode.lower()=='tv': #regularization with Total variation (TV)
        Gstar = G
        G = Gstar.T
        ps = np.zeros((source_number,1))
        b = G@Us_formatted
        T = lamb #threshold pour le  soft thresholding -> plus le threshold est faible moins on lisse!!! 
        hessian = G@Gstar
        gamma = gamma_coeff/np.max(np.abs(hessian))
        k = 0
        conv = False
        while conv==False and k<maxit:#
            ps_old = ps.copy()
            #gradient descent
            ps_bar = ps+gamma*(b-hessian@ps)
            #soft thresholding 
            for i in range(source_number):
                ps[i,0] = ps_bar[i,0]*np.max([1-(gamma*T)/np.abs(ps_bar[i,0]),0])
            res = ((ps-ps_old).T@(ps-ps_old))[0][0]
            if  res < epsilon: 
                conv = True    
            k += 1 
    #################################################################################################################################
    elif mode.lower()=='steepest': #Steepest descent 
        Us_formatted = Us.reshape((data_number,)) #reshaping data to be match the following implementations that us shape n, or m,
        if isinstance(Cm1, np.ndarray)==False: #on apeut déjà avoir créé au préalable la matrice de covariance 
            if Cm1=='laplacian' or Cm1=='Laplacian':
                Cm1 =  build_laplacian(rs)
            elif Cm1=='Gaussian' or Cm1=='gaussian':
                Cm1 = build_gaussian_inv(rs,sigma)
        lamb_cov_inv = lamb*Cm1.T@Cm1 
        alpha = 1
        conv = False
        i = 0
        ps = np.ones((source_number,))
        while conv==False and i<maxit:
            i +=1 
            delta_m = -compute_grad_misfit(ps=ps,Us_obs=Us_formatted,G=G,lamb_cov_inv=lamb_cov_inv)
            alpha_old = alpha
            alpha = linesearch(alpha=alpha_old,Us_obs=Us_formatted,G=G,ps=ps,delta_m=delta_m,lamb_cov_inv=lamb_cov_inv)
            if alpha==None:
                alpha = linesearch_armijo(alpha=alpha_old,Us_obs=Us_formatted,G=G,ps=ps,delta_m=delta_m,lamb_cov_inv=lamb_cov_inv) #mettre juste condition d'armijo si le strong wolfe n'arrive pas à converger
            if alpha<1e-10:
                alpha=1e-10
            ps += alpha*delta_m
            if isinstance(constraint,np.ndarray)==True:
                ps[constraint_idx] = constraint[constraint_idx] #on rajoute la constrainte -> par ex mettre à 0 la région souaitée 
            if np.dot(alpha*delta_m,alpha*delta_m) < epsilon:
                conv = True
    #################################################################################################################################
    elif mode.lower()=='nlcg': #mode NLCG -> faudra refaire au propre sans doute -> en appelant ces approches NLCG fast
        Us_formatted = Us.reshape((data_number,))
        ps = np.ones((source_number,))
        if isinstance(Cm1, np.ndarray)==False: #on apeut déjà avoir créé au préalable la matrice de covariance 
            if Cm1=='laplacian' or Cm1=='Laplacian':
                Cm1 =  build_laplacian(rs)
            elif Cm1=='Gaussian' or Cm1=='gaussian':
                Cm1 = build_gaussian_inv(rs,sigma)
        lamb_cov_inv = lamb*Cm1.T@Cm1
        rk = compute_grad_misfit(ps=ps,Us_obs=Us_formatted,G=G,lamb_cov_inv=lamb_cov_inv)
        pk = -rk
        conv = False
        alpha = 1
        i = 0
        while conv==False and i<maxit:
            Gpk = np.matmul(G,pk)
            #precompute the  r.T@r
            rkTrk = np.dot(rk,rk)
            alpha_old = alpha
            alpha = linesearch(alpha=alpha_old,Us_obs=Us_formatted,G=G,ps=ps,delta_m=pk,lamb_cov_inv=lamb_cov_inv)
            if alpha==None:
                alpha = linesearch_armijo(alpha=alpha_old,Us_obs=Us_formatted,G=G,ps=ps,delta_m=pk,lamb_cov_inv=lamb_cov_inv) #mettre juste condition d'armijo si le strong wolfe n'arrive pas à converger
            ps += alpha*pk
            if isinstance(constraint,np.ndarray)==True: #checking if a constraint was added
                ps[constraint_idx] = constraint[constraint_idx] #on rajoute la constrainte -> par ex mettre à 0 la région souaitée 
            rk = compute_grad_misfit(ps=ps,Us_obs=Us_formatted,G=G,lamb_cov_inv=lamb_cov_inv)
            beta = np.dot(rk,rk)/rkTrk
            pk = -rk + beta*pk
            i += 1
            # if np.linalg.norm(alpha*pk) < epsilon: #we use the norm to be more consistent with scipy value
            #     conv = True
            if np.dot(alpha*pk,alpha*pk) < epsilon:
                conv = True
    #################################################################################################################################
    elif mode.lower()=='nlcg_fast':
        Us_formatted = Us.reshape((data_number,))
        ps = np.ones((source_number,))
        if isinstance(Cm1, np.ndarray)==False: #on apeut déjà avoir créé au préalable la matrice de covariance 
            if Cm1=='laplacian' or Cm1=='Laplacian':
                Cm1 =  build_laplacian(rs)
            elif Cm1=='Gaussian' or Cm1=='gaussian':
                Cm1 = build_gaussian_inv(rs,sigma)
        lamb_cov_inv = lamb*Cm1.T@Cm1
        result = minimize(fun=compute_cost, x0=ps, args=(Us_formatted,G,lamb_cov_inv), method='CG', tol=epsilon, jac=compute_grad_misfit,
                          options={'maxiter': maxit})
        ps = result.x
    #################################################################################################################################
    elif mode.lower()=='l-bfgs' or mode.lower()=='lbfgs': #mode L-BFGS
        Us_formatted = Us.reshape((data_number,))
        ps = np.ones((source_number,))
        if isinstance(Cm1, np.ndarray)==False: #on apeut déjà avoir créé au préalable la matrice de covariance 
            if Cm1=='laplacian' or Cm1=='Laplacian':
                Cm1 =  build_laplacian(rs)
            elif Cm1=='Gaussian' or Cm1=='gaussian':
                Cm1 = build_gaussian_inv(rs,sigma)
        lamb_cov_inv = lamb*Cm1.T@Cm1
        i = 0
        m = 5 #number of s and y pairs to keep to act as the hessian estimate, 5 is supposed to be enough based on Nocedal
        conv = False
        s_hist = []
        y_hist = []
        rho_hist = []
        q = compute_grad_misfit(ps=ps,Us_obs=Us_formatted,G=G,lamb_cov_inv=lamb_cov_inv)
        while conv==False and i<maxit:
            if i == 0:
                pk = -q
            else:
                pk = -lbfgs_direction(q, s_hist, y_hist, rho_hist, m)
            alpha = linesearch(alpha=1,Us_obs=Us_formatted,G=G,ps=ps,delta_m=pk,lamb_cov_inv=lamb_cov_inv) #start with alpha=1 because we use the scaler for pk
            if alpha==None:
                alpha = linesearch_armijo(alpha=1,Us_obs=Us_formatted,G=G,ps=ps,delta_m=pk,lamb_cov_inv=lamb_cov_inv) #mettre juste condition d'armijo si le strong wolfe n'arrive pas à converger
            if alpha<1e-10:
                alpha=1e-10 #if still managed to fail to estimate alpha then we use this ad hoc value 
            ps_old = ps.copy()
            ps += alpha*pk
            if isinstance(constraint,np.ndarray)==True:
                ps[constraint_idx] = constraint[constraint_idx] #on rajoute la constrainte -> par ex mettre à 0 la région souaitée 
            sk = ps-ps_old #vecteur de shape source_number,
            q_old = q.copy()
            q = compute_grad_misfit(ps=ps,Us_obs=Us_formatted,G=G,lamb_cov_inv=lamb_cov_inv)
            yk = q-q_old #vecteur de shape source_number,
            if np.linalg.norm(yk) < epsilon: #we use this norm for convergence criterion otherwise we have trouble as we may divide by 0.
                break
            rho_k = 1 / np.dot(yk, sk)
            if len(s_hist) == m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(sk)
            y_hist.append(yk)
            rho_hist.append(rho_k)
            i += 1
            # if np.dot(alpha*pk,alpha*pk) < epsilon:
            #     conv = True
    #################################################################################################################################
    elif mode.lower()=='l-bfgs_fast' or mode.lower()=='lbfgs_fast': #mode L-BFGS fast avec scipy alsos uses Wolfe conditions to asses the value of alpha
        Us_formatted = Us.reshape((data_number,))
        ps = np.ones((source_number,))
        if isinstance(Cm1, np.ndarray)==False: #on apeut déjà avoir créé au préalable la matrice de covariance 
            if Cm1=='laplacian' or Cm1=='Laplacian':
                Cm1 =  build_laplacian(rs)
            elif Cm1=='Gaussian' or Cm1=='gaussian':
                Cm1 = build_gaussian_inv(rs,sigma)
        lamb_cov_inv = lamb*Cm1.T@Cm1
        if isinstance(constraint,np.ndarray)==True: #checking if a constraint was added
            ps[constraint_idx] = constraint[constraint_idx] #we make the first guess start with the correct value
            bounds = create_bounds(constraint)#we create bounds equal to our constraints for the specific ps values that need it 
            result = minimize(fun=compute_cost, x0=ps, args=(Us_formatted,G,lamb_cov_inv), method='L-BFGS-B', tol=epsilon, jac=compute_grad_misfit,
                         bounds=bounds,options={'maxiter': maxit})
        else:
            result = minimize(fun=compute_cost, x0=ps, args=(Us_formatted,G,lamb_cov_inv), method='L-BFGS-B', tol=epsilon, jac=compute_grad_misfit)
        ps = result.x
    
    if logpath!=None:
        elapsed_time = time.time()-t1
        if mode.lower()=='linear' or mode==None:
            i = None
        elif mode.lower()=='lbfgs_fast' or mode.lower()=='nlcg_fast': 
            i = result.nit
        log_run(E,v,lamb,epsilon,sigma,constraint,i,maxit,elapsed_time,logpath)
    
    ps = ps.reshape(rs[:,:,0,0].shape)
    return ps