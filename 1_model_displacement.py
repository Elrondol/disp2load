import numpy as np
import load2disp
import os

"""#ici on va devoir dire quelles lignes de source il va calculer avec ce run, il dva calculer ça pour tous les points renseigner 
#et il faudrait qu'à chaque fois qu'il fait une nouvelle ligne  il exporte un fichier 
### il faut déjà avoir créé le fichier avec le """

#################### OU VA T ON EXECUTER LE RUN #### -> les fichiers avec les lignes de sources seront rassemblés dans ce dossier 
directory = 'run_test2' 

######################################## SOURCES ET CONSTANTES ################
lines_to_compute = range(140,150) # (premiere ligne de sources calculée, dernière ligne calculée )


ps_file_path = 'ps.npy' #le chemin d'accès au fichier qui contient la grille de presisons  
rs_file_path = 'rs.npy' #le chemin d'accès au fichier contenant la grille avec les coordonnées des sommets des sources
E = 20e9
v = 0.25

###################################### POSITIONS À CALCULER ###########

xs = np.linspace(-500, 500, 200)  #nested loops -> il calcule la grille complète
ys = np.linspace(-500, 500, 200)
zs = np.linspace(0,100,3)

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

## creates the run directory if it doesn't exist 
try:
    os.mkdir(directory)
except:
    pass

ps = np.load(ps_file_path)  
rs = np.load(rs_file_path)
### selecting the rows to compute 
ps = ps[lines_to_compute,:]
rs = rs[lines_to_compute,:]

for vy_idx in range(len(ps[:,0])):  #we loop over rows so that we can clear the displacement for the following line of 
    Us = np.zeros((len(ys), len(xs), len(zs),3)) 
    for vx_idx in range(len(ps[0,:])):
        p = ps[vy_idx,vx_idx]
        r = rs[vy_idx,vx_idx]
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    xyz = [x, y, z]
                    U = load2disp.load2disp(xyz, r, p, E, v)
                    Us[j, i,k,:] += U.reshape(3)
    np.save(f'{directory}/source_line_{lines_to_compute[vy_idx]:03}.npy', Us) #saving the computed line 
    
