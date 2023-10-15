import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange
import load2disp
import utils
from multiprocessing import Pool
import time
import concurrent.futures

#################### WHERE THE DISPLACEMENT IS COMPUTED #################"
#providing boundaries and ticks for the general domain  ->
xs = np.linspace(-500, 500, 40)
ys = np.linspace(-500, 500, 40)
zs = np.linspace(0,100,40) #fait de la 3D! 
Us = np.zeros((len(ys), len(xs), len(zs),3)) #créé la map des déplacement initialisée à 0 

##################################### DEFINING VERTICES WITH THEIR PARAMETERS #############
ps = np.random.randint(low=0, high=9.81*10*1000, size=(2,2))  
# ps = np.ones((2,2))*9.81*10*1000
E = 20e9
v = 0.25
#creating a mesh wtih all vertices (100 vertices in our case where we use ps of size (10*10)) 
rs = utils.create_source_mesh(-450,450,-450,450,ps) #vertices should have part out of the domain in which we compute 

############################ LOOPING OVER VERTICES (VERTICES STORED IN 2D ARRAY TO BE SIMILAR TO GRIDDING OF PRESSURE)######
tim = time.time()
        
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = []
    #computing the combinations of coordinate points where the displacement will have to be evaluated
    for vx_idx in range(len(ps[0,:])): #boucle sur les vertices et on va sommet les réusltats sur les vectices pour faire le délacement gloabel 
        for vy_idx in range(len(ps[:,0])): 
            p = ps[vy_idx,vx_idx]
            r = rs[vy_idx,vx_idx]
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    for k, z in enumerate(zs):
                        xyz = [x, y, z]
                        futures.append(executor.submit(load2disp.load2disp, xyz, r, p, E, v))

    ######### now finally computing the displacement and putting it in Us 
    for vx_idx in range(len(ps[0,:])): #boucle sur les vertices et on va sommet les réusltats sur les vectices pour faire le délacement gloabel 
        for vy_idx in range(len(ps[:,0])): 
            p = ps[vy_idx,vx_idx]
            r = rs[vy_idx,vx_idx]
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    for k, z in enumerate(zs):
                        U = futures.pop(0).result()
                        Us[j, i,k,:] += U.reshape(3)  #we add up the contribution of vertices

print('It took',time.time()-tim,f's to compute the displacement at {Us.shape[0]*Us.shape[1]*Us.shape[2]} points for {ps.shape[0]*ps.shape[1]} vertices ') #