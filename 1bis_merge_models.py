import numpy as np
import os

"""Ce script sert à merge les fichiers de déplacement associés à chacune des lignes de source """

#################### OU VA T ON EXECUTER LE RUN #### -> les fichiers avec les lignes de sources seront rassemblés dans ce dossier 
directory = 'run_test' 


######################################## on load les fichiers de source et on les stack 

init =  np.load(f'{directory}/source_line_000.npy')#loading the first file to know the shape 
Us = np.zeros(init.shape) #initiliazing Us

for filename in os.listdir(directory):
    if filename.startswith('source_line'):
        Us_tmp = np.load(f'{directory}/{filename}')
        Us += Us_tmp #summing up all the matrices from all the lines
        
np.save(f'{directory}/source_line_full.npy', Us)
