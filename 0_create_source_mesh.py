import numpy as np
import matplotlib.pyplot as plt
import load2disp
import utils
import time

"""pas besoin d'utiliser ce script si on fait ça après l'inversion, car l'inversion va justement return une grille de ps et pourra 
#aussi retourner le mesh de coordonnées à associer aux sources  

ce code servirait à faire le mode forward où on a des données de précipitation non griddées et où faudrait les interpoler et tout 
-> globalement pas nécessaire dans la plupart des utilisations""""


#on lui donne à manger des données par exemple de water 




#### interpolation des données sur grille régulière 


ps = 




#on créé le mesh avec les positions de source 
rs = utils.create_source_mesh(-450,450,-450,450,ps) #vertices should have part out of the domain in which we compute 

