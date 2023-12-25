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
nxinv = 100
nyinv = 100
xrinv = np.linspace(-31241.831049184548, 749768.9670936721, nxinv) #ces dooronnées correcpondent au milieu des pixels -> donc rs doit être un peu au dela de ces 
yrinv = np.linspace(3638619.8502579452, 4431792.876044568, nyinv)
dxinv = xrinv[1]-xrinv[0]
dyinv = yrinv[1]-yrinv[0]
rs4inversion =  utils.create_source_mesh(xrinv[0]-dxinv/2,xrinv[-1]+dxinv/2,yrinv[0]-dyinv/2,yrinv[-1]+dyinv/2,np.zeros((nyinv,nxinv))) 

laplacian = disp2load.build_laplacian(rs4inversion)
np.save(f'laplacian.npy', laplacian)
