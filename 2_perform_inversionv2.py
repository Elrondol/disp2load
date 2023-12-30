import pandas as pd
import fastparquet
import os 
import numpy as np
import utils
import disp2load
import random
import glob
import re
""""""
noise = 'noisy3' #can choose the noise we use in order to make sure we always use the same displacements for all runs
data = np.load(f'data_Hilary_synthetic_{noise}.npy') 
zone = 11 #zone utm pour la conversion

E = 75e9
v = 0.25

G = np.load('/home/parisnic/project/disp2load/G.npy') #load G qui a déjà été précomputed

mode = 'lbfgs' #None, linear, lbfgs, nlcg, nlcg_fast,lbfgs_fast, TV
epsilon=1e-35
maxit = 1e3
Cm1 = np.load('/home/parisnic/project/disp2load/gaussian_1.0.npy') #'gaussian' or 'laplacian' to compute, but we can precompute it to save time
sigma =  1.0 #penser à use 
#lambs = np.logspace(-24,6,100)
lambs = [1.3219411484660289e-12,2.1544346900318868e-11]

constraint = None #np.load('constraint.npy')


if isinstance(constraint,np.ndarray)==True:
    is_constrained = True
else:
    is_constrained = False


if sigma==None:
    savepath = f'/home/parisnic/project/disp2load/inversion_laplacian_{mode}_{noise}_constraint_{is_constrained}'
else:
    savepath = f'/home/parisnic/project/disp2load/inversion_gaussian_{mode}_{noise}_constraint_{is_constrained}'

try:
    os.mkdir(savepath)
except:
    pass
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################

lat, lon = data[:,0], data[:,1]
Us = data[:,2:5]
xs, ys = utils.latlon2xy(latitudes=lat, longitudes=lon, elevations=None, zone=zone)

#création du mesh
nxinv = 100
nyinv = 100
xrinv = np.linspace(-31241.831049184548, 749768.9670936721, nxinv) #ces dooronnées correcpondent au milieu des pixels -> donc rs doit être un peu au dela de ces bounds
yrinv = np.linspace(3638619.8502579452, 4431792.876044568, nyinv)
dxinv = xrinv[1]-xrinv[0]
dyinv = yrinv[1]-yrinv[0]
rs4inversion =  utils.create_source_mesh(xrinv[0]-dxinv/2,xrinv[-1]+dxinv/2,yrinv[0]-dyinv/2,yrinv[-1]+dyinv/2,np.zeros((nyinv,nxinv))) 

#INVERSION
for lamb in lambs:
    ps = disp2load.disp2load(E,v,rs4inversion,xs,ys,Us,mode=mode, lamb=lamb, epsilon=epsilon, gamma_coeff=1e-2, sigma=sigma,G=G,
                            constraint=constraint, maxit=maxit, Cm1=Cm1, logpath=savepath)
    np.save(f'{savepath}/ps_sig_{sigma}_lamb_{lamb}.npy', ps)

    source_number = len(ps[0,:])*len(ps[:,0])
    ps = ps.reshape(source_number,1)
    Us_cal = G@ps
    Us_cal = Us_cal.reshape((len(xs),3))

    np.save(f'{savepath}/Us_sig_{sigma}_lamb_{lamb}.npy', Us_cal)
