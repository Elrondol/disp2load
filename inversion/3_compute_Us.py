import pandas as pd
import fastparquet
import os 
import numpy as np
import utils
import disp2load
import random
import glob
import re
import load2disp
""""""
zone = 11 #zone utm pour la conversion



E = 75e9
v = 0.25

nxinv = 100
nyinv = 100
xrinv = np.linspace(-31241.831049184548, 749768.9670936721, nxinv) #ces dooronnées correcpondent au milieu des pixels -> donc rs doit être un peu au dela de ces bounds
yrinv = np.linspace(3638619.8502579452, 4431792.876044568, nyinv)


####### 

# data = np.load(f'{data_directory}data_{day}.npy') 
data = np.load(f'data_Hilary_synthetic_noisy.npy') 
lat, lon = data[:,0], data[:,1]

#### conversion des coordoonées en x y 
xs, ys = utils.latlon2xy(latitudes=lat, longitudes=lon, elevations=None, zone=11)

#création du mesh
dxinv = xrinv[1]-xrinv[0]
dyinv = yrinv[1]-yrinv[0]
rs4inversion =  utils.create_source_mesh(xrinv[0]-dxinv/2,xrinv[-1]+dxinv/2,yrinv[0]-dyinv/2,yrinv[-1]+dyinv/2,np.zeros((nyinv,nxinv))) 

##### FORWARD MODELLING REQUIRED TO MAKE THE L CURVE
# l = E * v / ((1 + v) * (1 - 2 * v))
# m = E / (2 * (1 + v))    
# data_number = len(xs)*3
# source_number = nyinv*nxinv
# G = disp2load.build_G(rs4inversion,xs,ys,data_number,l,m)

#on doit à présent trouver les fichiers qui vont bien et extraire la valeur de alpha


# Find all .npy files in the directory that end with the specified string
file_list = glob.glob(f'ps_*.npy')

for i in range(len(file_list)):
    if file_list[i] == 'ps_mode=1_alpha=0.0033516026509388406_nx=100_ny=100.npy':
        idx = i

print(file_list)

file_list.pop(idx)
#c'est bon on a la liste des fichiers on peut les load et extraire la valeur de alpha  osef d'extraire la valeur de alpha pour le moment 


def extract_number_from_filename(filename):
    pattern = r'ps_([0-9]+\.[0-9]+)\.npy'
    match = re.search(pattern, filename)
    if match:
        extracted_number = match.group(1)
        return float(extracted_number)
    else:
        return None

nsta = len(xs)
z = 0 #va mettre toutes les stations à 0 m
    
for j, file in enumerate(file_list): #boucle sur les distribution de sources calculées 
    alpha = extract_number_from_filename(file)
    #vérifie que le Us pour le alpha donné a pas encoré été calculé  
    if  os.path.exists(f'Us_{alpha}.lock')==False and os.path.exists(f'Us_{alpha}.npy')==False: #si pas en train de calculer et pas déjà calculée
        #### doit maintenant créer le fichier temporaire pour lock la ligne

        with open(f'Us_{alpha}.lock', 'w') as lock_file:
            lock_file.write("")  # You can write some data or leave it empty

        ps = np.load(file)    
        #y'a que ps qui change 
        Us = np.zeros((nsta, 3)) 
        for vy_idx in range(nyinv): 
            for vx_idx in range(nxinv): 
                p = ps[vy_idx,vx_idx]
                if p!=0: #si p =0 alors pas besoin de calculer l'influence de cette source car nulle 
                    r = rs4inversion[vy_idx,vx_idx]
                    for i in range(nsta):
                        xyz = [xs[i], ys[i], z]
                        U = load2disp.load2disp( xyz, r, p, E, v)
                        Us[i, :] += U.reshape(3)

        # ps = ps.reshape(source_number,1)
        # Us_cal = G@ps
        # Us_cal = np.reshape((len(xs),3))
        np.save(f'Us_{alpha}.npy', Us)
        os.remove(f'Us_{alpha}.lock')
    
