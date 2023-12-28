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

sigmas = np.power(10, np.arange(-4., 4.)) 

nxinv = 100
nyinv = 100
xrinv = np.linspace(-31241.831049184548, 749768.9670936721, nxinv) #ces dooronnées correcpondent au milieu des pixels -> donc rs doit être un peu au dela de ces 
yrinv = np.linspace(3638619.8502579452, 4431792.876044568, nyinv)
#création du mesh
dxinv = xrinv[1]-xrinv[0]
dyinv = yrinv[1]-yrinv[0]
rs4inversion =  utils.create_source_mesh(xrinv[0]-dxinv/2,xrinv[-1]+dxinv/2,yrinv[0]-dyinv/2,yrinv[-1]+dyinv/2,np.zeros((nyinv,nxinv))) 

# xstart, ystart = utils.latlon2xy(40.5,-113.5,zone=11)
# xend, yend = utils.latlon2xy(42,-111.5,zone=11)
# rs4inversion =  utils.create_source_mesh(xstart,xend,ystart,yend,np.zeros((70,70))) 

for sigma in sigmas:
    gaussian = disp2load.build_gaussian_inv(rs4inversion,sigma,normalized=True)
    np.save(f'gaussian_{sigma}.npy', gaussian)
