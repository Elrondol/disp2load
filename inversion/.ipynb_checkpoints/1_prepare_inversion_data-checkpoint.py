import pandas as pd
import fastparquet
import os 

"""Ce script a pour vocation de load les fichiers de donné, de vérifier si la station est dans une région spécifiée qui sera la région dans laquelle on fera l'inversion
et si c'est le cas alors   on récupère le déplacement au jour indiqué et on doit se servir de la valeur de déplacement avant hillary pour pouvoir détemriner la qte de
déplacement qui est directement liée  à la Hilary (peut aussi utiliser régression sur la période d'vana thillary pour calculer la valeur de déplacement attendue ce jour
là pour alors estimer la valeur de déplacement liée à Hialry )


l'ouput devra alors être un fichier npy  avec en ligne les diverses stations et selon les colonnes : latitude_sta, longitude_sta, déplacement e, deplacement n, deplacement up"""



data_directory = '/media/parisnic/STOCKAGE/M2/GNSS_US_daily/'   
lat_min, lat_max, lon_min, lon_max = 

day = '23AUG21' #denseigner dans le format YYMMMDD

######################################## on load les fichiers de source et on les stack 

stations = []

for filename in os.listdir(data_directory):
    if filename.endswith('.parquet'):
        df = pd.read_parquet(f'{data_directory}{filename}', sep=' ')
        idx_day = np.where(df['YYMMMDD'].values==day)[0]
        try : #in case there is no data for the given day at this station 
            if lat_min<=df['_latitude(deg)'].values[idx_day]<= lat_max and lon_min<=df['_longitude(deg)'].values[idx_day]<= lon_max: #if  inside the window in which we put sources
                
                
        except:
            pass
        
        
stations = np.array(stations)
np.save(f'{data_directory}data_{day}.npy')