import numpy as np
import load2disp
import os
import tempfile

"""#ici on va devoir dire quelles lignes de source il va calculer avec ce run, il dva calculer ça pour tous les points renseigner 

ce script a une parallélisaiton naturelle : il créé un fiichier pour indiquer qu'il travail sur une ligne donnée  et il créé le fichier de la ligne et supprime le fichier lock -> on peut lancer le job autant de fois que l'on veut et chacun se mettra à travailler sur une ligne diféfrente  """

#################### OU VA T ON EXECUTER LE RUN #### -> les fichiers avec les lignes de sources seront rassemblés dans ce dossier 
directory = 'run_verif_convert' 

######################################## SOURCES ET CONSTANTES ################

ps_file_path = 'ps_verif_convert.npy' #le chemin d'accès au fichier qui contient la grille de presisons  
rs_file_path = 'rs_verif_convert.npy' #le chemin d'accès au fichier contenant la grille avec les coordonnées des sommets des sources
E = 75e9
v = 0.25

###################################### POSITIONS À CALCULER ###########

xs = np.linspace(-1932000,-1735000, 200)  #nested loops -> il calcule la grille complète
ys = np.linspace(-4406000,-4442000, 200)
zs = np.linspace(0,2,3)

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

for vy_idx in range(len(ps[:,0])):  #we loop over rows so that we can clear the displacement for the following line of 

    #### on vérifie si y'a pas déjà un job en train de bosser sur la ligne 
    if  os.path.exists(f'{directory}/source_line_{vy_idx:03}.lock')==False and os.path.exists(f'{directory}/source_line_{vy_idx:03}.npy')==False: #si pas en train de calculer et pas déjà calculée
        #### doit maintenant créer le fichier temporaire pour lock la ligne

        with open(f'{directory}/source_line_{vy_idx:03}.lock', 'w') as lock_file:
            lock_file.write("")  # You can write some data or leave it empty

        Us = np.zeros((len(ys), len(xs), len(zs),3)) 
        for vx_idx in range(len(ps[0,:])):
            p = ps[vy_idx,vx_idx]
            r = rs[vy_idx,vx_idx]
            if p ==0: #If load is 0, then we don't need to perform the computation
                Us[:,:,:,:] = 0
            else:
                for i, x in enumerate(xs):
                    for j, y in enumerate(ys):
                        for k, z in enumerate(zs):
                            xyz = [x, y, z]
                            U = load2disp.load2disp(xyz, r, p, E, v)
                            Us[j, i,k,:] += U.reshape(3)
        np.save(f'{directory}/source_line_{vy_idx:03}.npy', Us) #saving the computed line 
        ## il  a créé le ficihier de la ligne; on peut par conséquent supprimer le lock file 
        os.remove(f'{directory}/source_line_{vy_idx:03}.lock')
