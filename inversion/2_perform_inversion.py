import pandas as pd
import fastparquet
import os 
import numpy as np
import utils
import disp2load
import random
""""""
data_directory = '/media/parisnic/STOCKAGE/M2/GNSS_US_daily/'   
day = '23AUG21' #renseigner dans le format YYMMMDD
zone = 11 #zone utm pour la conversion

nsta_selec = None #indique le nombre de stations à sélectionner histoire de savoir  -> None pour pas faire de sélection

E = 75e9
v = 0.25

mode = 1 #0 for no regu, 1 for smooth, 2 for TV
alpha = 1  #lambda in regularizer term, peut direct le mettre si on le connait déjà, sinon juste on peut le 
alpha = random.sample(np.log10(-5,5,100),1)
nxinv = 80
nyinv = 80
xrinv = np.linspace(-31241.831049184548, 749768.9670936721, nxinv) #ces dooronnées correcpondent au milieu des pixels -> donc rs doit être un peu au dela de ces bounds
yrinv = np.linspace(3638619.8502579452, 4431792.876044568, nyinv)


####### 
# data = np.load(f'{data_directory}data_{day}.npy') 
data = np.load(f'data_Hilary_synthetic.npy') 
lat, lon = data[:,0], data[:,1]
Us = data[:,2:6]

#### conversion des coordoonées en x y 
xs, ys = utils.latlon2xy(latitudes=lat, longitudes=lon, elevations=None, zone=11)

### sélection de données aléatoirement dans le dataset au cas où y'a trop de données
if nsta_selec!=None:
    start_range = 0
    end_range = len(xs)
    sta_idx = random.sample(range(start_range, end_range), nsta_selec)
    xs = xs[sta_idx]
    ys = ys[sta_idx]
    Us = Us[sta_idx,:]

#création du mesh
dxinv = xrinv[1]-xrinv[0]
dyinv = yrinv[1]-yrinv[0]
rs4inversion =  utils.create_source_mesh(xrinv[0]-dxinv/2,xrinv[-1]+dxinv/2,yrinv[0]-dyinv/2,yrinv[-1]+dyinv/2,np.zeros((nyinv,nxinv))) 



#INVERSION
ps = disp2load.disp2load(E,v,rs4inversion,xs,ys,Us,mode=mode, alpha=alpha, epsilon=1e-6, gamma_coeff=1e-2)

np.save(f'ps_mode={mode}_alpha={alpha}_nx={nx}_ny={ny}.npy')