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
data_directory = '/media/parisnic/STOCKAGE/M2/GNSS_US_daily/'   
day = '23AUG21' #renseigner dans le format YYMMMDD
zone = 11 #zone utm pour la conversion

filename = 'ps_LBFGS'

nsta_selec = None #indique le nombre de stations à sélectionner histoire de savoir  -> None pour pas faire de sélection
E = 75e9
v = 0.25

mode = 5 #0 for no regu, 1 for smooth, 2 for TV, 3  Gaussian, 4 NLCG, 5 L-BFGS 
epsilon=1e-25

lambs = np.logspace(-5,9,200)

OK = False
################ si le alpha a déja et calculé alors 
while OK==False:
    lamb = lambs[np.random.randint(0,len(lambs))] #essaye de dchoisir une valeur de apha 
    
    if os.path.exists(f'{filename}_{lamb}.lock')==False and os.path.exists(f'{filename}_{lamb}.npy')==False: #si pas en train de calculer et pas déjà calculée
        #### doit maintenant créer le fichier temporaire pour lock la ligne

        with open(f'{filename}_{lamb}.lock', 'w') as lock_file:
            lock_file.write("")  # You can write some data or leave it empty
    
        OK = True #c'est bon c'est une nouvelle valeur de alpha donc on la calcule   


nxinv = 100
nyinv = 100
xrinv = np.linspace(-31241.831049184548, 749768.9670936721, nxinv) #ces dooronnées correcpondent au milieu des pixels -> donc rs doit être un peu au dela de ces bounds
yrinv = np.linspace(3638619.8502579452, 4431792.876044568, nyinv)


####### 
# data = np.load(f'{data_directory}data_{day}.npy') 
data = np.load(f'data_Hilary_synthetic_noisy.npy') 
lat, lon = data[:,0], data[:,1]
Us = data[:,2:6]

#### conversion des coordoonées en x y 
xs, ys = utils.latlon2xy(latitudes=lat, longitudes=lon, elevations=None, zone=11)

### sélection de données aléatoirement dans le dataset au cas où y'a trop de données
if nsta_selec!=None:
    sta_idx = random.sample(range(0, len(xs)), nsta_selec)
    xs = xs[sta_idx]
    ys = ys[sta_idx]
    Us = Us[sta_idx,:]

#création du mesh
dxinv = xrinv[1]-xrinv[0]
dyinv = yrinv[1]-yrinv[0]
rs4inversion =  utils.create_source_mesh(xrinv[0]-dxinv/2,xrinv[-1]+dxinv/2,yrinv[0]-dyinv/2,yrinv[-1]+dyinv/2,np.zeros((nyinv,nxinv))) 

#INVERSION
ps = disp2load.disp2load(E,v,rs4inversion,xs,ys,Us,mode=mode, lamb=lamb, epsilon=epsilon, gamma_coeff=1e-2)
np.save(f'{filename}_{lamb}.npy', ps)
os.remove(f'{filename}_{lamb}.lock')







# l = E * v / ((1 + v) * (1 - 2 * v))
# m = E / (2 * (1 + v))    
# data_number = len(Us[0,:])*len(Us[:,0])
# source_number = len(ps[0,:])*len(ps[:,0])
# G = disp2load.build_G(rs4inversion,xs,ys,data_number,l,m)
# ps = ps.reshape(source_number,1)
# Us_cal = G@ps
# Us_cal = np.reshape((len(Us[:,0]),len(Us[0,:])))

# np.save(f'Us_{alpha}.npy', Us_cal)


#np.save(f'Us_{alpha}.npy', Us)
