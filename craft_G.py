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
zone = 11 #zone utm pour la conversion
E = 75e9
v = 0.25

# nxinv = 100
# nyinv = 100
# xrinv = np.linspace(-31241.831049184548, 749768.9670936721, nxinv) #ces dooronnées correcpondent au milieu des pixels -> donc rs doit être un peu au dela de ces bounds
# yrinv = np.linspace(3638619.8502579452, 4431792.876044568, nyinv)

# ####### 
# data = np.load(f'data_Hilary_synthetic_noisy.npy') 
# lat, lon = data[:,0], data[:,1]
# Us = data[:,2:6]

# #### conversion des coordoonées en x y 
# xs, ys = utils.latlon2xy(latitudes=lat, longitudes=lon, elevations=None, zone=11)

# #création du mesh
# dxinv = xrinv[1]-xrinv[0]
# dyinv = yrinv[1]-yrinv[0]
# rs4inversion =  utils.create_source_mesh(xrinv[0]-dxinv/2,xrinv[-1]+dxinv/2,yrinv[0]-dyinv/2,yrinv[-1]+dyinv/2,np.zeros((nyinv,nxinv))) 


xstart, ystart = utils.latlon2xy(40.5,-113.5,zone=11)
xend, yend = utils.latlon2xy(42,-111.5,zone=11)
ps = np.zeros((100,100))
rs4inversion = utils.create_source_mesh(xstart,xend,ystart,yend,ps) #sources should have part out of the domain in which we compute 

xs = np.linspace(xstart, xend, 80) #defining the coordinates of the receivers
ys = np.linspace(ystart, yend, 80)

YY, XX = np.meshgrid(ys,xs)

xs = XX.reshape(6400)
ys = YY.reshape(6400)

##### calcul des déplacements associés
l = E * v / ((1 + v) * (1 - 2 * v))
m = E / (2 * (1 + v))    
# data_number = len(Us[0,:])*len(Us[:,0])
G = disp2load.build_G(rs4inversion,xs,ys,l,m)
np.save(f'G_salt.npy', G)
