import numpy as np
from scipy.integrate import quad
from numpy import arctan, tan, cos, sin, arccos, arcsin, sqrt, log, log10, pi

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


def build_G(rs,xs,ys,Us_formatted,l,m):
    """Fonction pour build G, la matrice qui relie les pressions au displacements"""
    source_number = len(rs[:,0])*len(rs[0,:]) 
    data_number = len(Us_formatted)
    
    G = np.zeros((data_number,source_number))
    rs_formatted = rs.reshape(source_number,4,2) 
        
    for i in range(source_number): # on remplit à présent la matrice G et pour ça on a besoin d'itérer sur les sources        
        r = rs_formatted[i] 
        #on doit aussi boucler sur les positions de la station car le r_translated en dépend en fait ... ->  par contre 1 station fournit 3 data
        for j in range(len(xs)): 
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
            tmp = -I1/(4*np.pi*(l+m)) #coefficient pour la source i et pour la station j  pour les déplacements horizontaux
            alpha = tmp[0]
            beta  = tmp[1]
            gamma = I3/(4*np.pi*m) + I4/(4*np.pi*(l + m)) #coefficent pour la source i  et la station j pour les déplacements verticaux
            
            starting_idx = j*3 #car 3 données par station
            G[starting_idx,i] = alpha
            G[starting_idx+1,i] = beta
            G[starting_idx+2,i] = gamma
    
    return G


def kfromij(i,j,nx): # fonction pour trouver les indices globaux des sources car les m sont regroupés dans un vecteur 
    return j*nx+i #numerotation globale 


def build_laplacian(rs):
    """Fonction qui compute le laplacien servant pour la régularization afin d'avoir une solution smooth"""
    
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




def disp2load(E,v,rs,xs, ys, Us, mode=0, alpha=1, epsilon=1e-2, gamma_coeff=0.1):
    '''Need to provide the fucntion the shape of ps expected and Us the data in the format [[x1,y1,z1],
                                                                                            [x2,y2,z2],
                                                                                            ...
                                                                                            [xn,yn,zn]]
                                                                                            
            For  mode=1  alpha is the value of lmabda, the coefficient of the regularizer                                                                        '''
    # Lame constants
    l = E * v / ((1 + v) * (1 - 2 * v))
    m = E / (2 * (1 + v))    
    
    data_number = len(Us[:,0])*len(Us[0,:])
    source_number = len(rs[:,0,0,0])*len(rs[0,:,0,0])
    
    Us_formatted = Us.reshape((data_number,1)) #reshape en mettant en trois premier les éléments de la première ligne, puis 3 suivant ce sont les 3 de la seconde ligne..
    G = build_G(rs,xs,ys,Us_formatted,l,m)
    
    if mode==0: #without regularization    
        ps = np.linalg.solve(G.T@G,G.T@Us_formatted)
        
    elif mode==1: #smooth regularization
        laplacian =  build_laplacian(rs)
        GTG = G.T@G
        GTD = G.T@Us_formatted
        ps = np.linalg.inv(GTG + alpha*laplacian.T@laplacian) @ GTD 
    
    elif mode==2: #regularization with something 
        Gstar = G
        G = Gstar.T
        ps = np.zeros((source_number,1))
        b = G@Us_formatted
        T = alpha #threshold pour le  soft thresholding -> plus le threshold est faible moins on lisse!!! 
        hessian = G@Gstar
        gamma = gamma_coeff/np.max(np.abs(hessian))
        k = 0
        conv = False
        while conv==False :#
            ps_old = ps.copy()
            #gradient descent
            ps_bar = ps+gamma*(b-G@Gstar@ps)
            #soft thresholding 
            for i in range(source_number):
                ps[i,0] = ps_bar[i,0]*np.max([1-(gamma*T)/np.abs(ps_bar[i,0]),0])
            res = ((ps-ps_old).T@(ps-ps_old))[0][0]
            if  res < epsilon: #on vérifie la convergence, attention convergence basée sur des valeurs en radians donc faut espsilon assez faible pour avoir une bonne précision 
                conv = True    
            k += 1 
        
    ps = ps.reshape(rs[:,:,0,0].shape)
    return ps
    