import pandas as pd
import fastparquet
import os 
import numpy as np

""""""


data_directory = '/media/parisnic/STOCKAGE/M2/GNSS_US_daily/'   
lat_min, lat_max, lon_min, lon_max = 

day = '23AUG21' #denseigner dans le format YYMMMDD

####### 

data = np.load(f'{data_directory}data_{day}.npy')

...