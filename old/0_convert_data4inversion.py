import pandas as pd
import fastparquet
import os 

"""Ce script a pour vocation de convertir les données des stations qui sont sauvegardées dans des fichiers CSV en fichiers parquet pour gagner en vitesse de lecture et en espace"""

data_directory = '/media/parisnic/STOCKAGE/M2/GNSS_US_daily/' 

######################################## on load les fichiers de source et on les stack 

for filename in os.listdir(data_directory):
    if filename.endswith('.csv'):
        df = pd.read_csv(f'{data_directory}{filename}', sep=' ')
        df.to_parquet(f'{data_directory}{filename[:-4]}.parquet')
        os.remove(f'{data_directory}{filename}') #remove le csv
