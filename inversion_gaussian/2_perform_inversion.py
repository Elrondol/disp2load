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
zone = 11 #zone utm pour la conversion

sigma = 1e-1

G = np.load('G.npy') #load G qui a déjà été précomputed

nsta_selec = None #indique le nombre de stations à sélectionner histoire de savoir  -> None pour pas faire de sélection
E = 75e9
v = 0.25

mode = 3 #0 for no regu, 1 for smooth, 2 for TV, 3  Gaussian, 4 NLCG, 5 L-BFGS 
epsilon=1e-25


lambs = np.logspace(-13,-5,200)

OK = False
################ si le alpha a déja et calculé alors 
while OK==False:
    lamb = lambs[np.random.randint(0,len(lambs))] #essaye de dchoisir une valeur de apha 
    
    if os.path.exists(f'ps_sig_{sigma}_lamb_{lamb}.lock')==False and os.path.exists(f'ps_sig_{sigma}_lamb_{lamb}.npy')==False: #si pas en train de calculer et pas déjà calculée
        #### doit maintenant créer le fichier temporaire pour lock la ligne

        with open(f'ps_sig_{sigma}_lamb_{lamb}.lock', 'w') as lock_file:
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
ps = disp2load.disp2load(E,v,rs4inversion,xs,ys,Us,mode=mode, lamb=lamb, epsilon=epsilon, gamma_coeff=1e-2, sigma=sigma,G=G)
np.save(f'ps_sig_{sigma}_lamb_{lamb}.npy', ps)
os.remove(f'ps_sig_{sigma}_lamb_{lamb}.lock')

##### calcul des déplacements associés
#l = E * v / ((1 + v) * (1 - 2 * v))
#m = E / (2 * (1 + v))    
#data_number = len(Us[0,:])*len(Us[:,0])
source_number = len(ps[0,:])*len(ps[:,0])
#G = disp2load.build_G(rs4inversion,xs,ys,data_number,l,m)
ps = ps.reshape(source_number,1)
Us_cal = G@ps
Us_cal = Us_cal.reshape((len(xs),3))

#nsta = len(xs)
#z = 0 #va mettre toutes les stations à 0 m
#Us_cal = np.zeros((nsta, 3)) 
#for vy_idx in range(nyinv): 
   # for vx_idx in range(nxinv): 
    #    p = ps[vy_idx,vx_idx]
     #   if p!=0: #si p =0 alors pas besoin de calculer l'influence de cette source car nulle 
      #      r = rs4inversion[vy_idx,vx_idx]
       #     for i in range(nsta):
        #        xyz = [xs[i], ys[i], z]
         #       U = load2disp.load2disp( xyz, r, p, E, v)
          #      Us_cal[i, :] += U.reshape(3)

np.save(f'Us_sig_{sigma}_lamb_{lamb}.npy', Us_cal)
