import numpy as np

#considère que E et  v sont des constantes, il faudra juste donner à manger au code la matrice avec les valeur de p et la les boundaries 
#du premier et dernier vectice et il s'occupe alors de créér un maillage régulier avec des vectices aayant le 

def create_source_mesh(x0,x1,y0,y1,ps):
    # """x0 et x1 = le plus petit et plus grand x du domaine ,  
    # la focntion va return un maillage  2D/3D qui contiendra  les différents vertices (les vertices sont chacun un array de taille 4 pour faire 
    # on polynome carré)""""
    vertice_mesh = np.zeros((ps.shape[0],ps.shape[1],4,2))
    x_num = ps.shape[1]
    y_num = ps.shape[0]
        
    xv = np.linspace(x0,x1,x_num+1)
    yv = np.linspace(y0,y1,y_num+1)
    
    for i in range(x_num):
        for j in range(y_num):
            vertice_mesh[j,i,:,:] = np.array([[xv[i],yv[j]],
                                             [xv[i+1],yv[j]],
                                             [xv[i+1],yv[j+1]],
                                             [xv[i],yv[j+1]]])
    return vertice_mesh


def compute_strain_tensor_at_point(Up,dx,dy,dz):
    """Calcule le tensueur de déformation a un point donné du maillage  p , on doit lui donner à manger un array qui contient les déplacement en x y et z au point, au point d'vaant
    et au point d'après pour faire dérivée d'ordre 2 = euler scheme d'ordre 2  
    Up = [[u_x-1,u_x+1],[y_x-1,y_x+1],[z_x-1,z_x+1]]"""
    strain_tensor = np.zeros((3,3))
    
    #computing derivatives of displacement  -> composantes normales:
    strain_tensor[0,0] = (Up[0,1]-Up[0,0])/(2*dx)
    strain_tensor[1,1] = (Up[1,1]-Up[1,0])/(2*dy)
    strain_tensor[2,2] = (Up[2,1]-Up[2,0])/(2*dz)
    #composantes cisaillantes 
    duxy = (Up[0,1]-Up[0,0])/(2*dy)
    duyx = (Up[1,1]-Up[1,0])/(2*dx)
    duzy = (Up[2,1]-Up[2,0])/(2*dy)
    duyz = (Up[1,1]-Up[1,0])/(2*dz)
    duxz = (Up[0,1]-Up[0,0])/(2*dz)
    duzx = (Up[2,1]-Up[2,0])/(2*dx)
    
    strain_tensor[0,1] = strain_tensor[1,0] = 0.5*(duxy+duyx)
    strain_tensor[0,2] = strain_tensor[2,0] = 0.5*(duxz+duzx)
    strain_tensor[2,1] = strain_tensor[1,2] = 0.5*(duzy+duyz)
    
    return strain_tensor


def lat_lng_to_cartesian(lat, lng):
    """Converts lat lon coordinates to cartesian"""
    EARTH_RADIUS = 6371000
    # Convert latitude and longitude from degrees to radians
    lat_rad = np.deg2rad(lat)
    lng_rad = np.deg2rad(lng)
    # Calculate Cartesian coordinates
    x = EARTH_RADIUS * np.cos(lat_rad) * np.cos(lng_rad)
    y = EARTH_RADIUS * np.cos(lat_rad) * np.sin(lng_rad)
    z = EARTH_RADIUS * np.sin(lat_rad)
    return x, y, z

def cartesian_to_lat_lng(x, y, z):
    """Converts cartesian coordinates to lat lon"""
    EARTH_RADIUS = 6371000
    # Calculate latitude in radians
    lat_rad = np.arctan2(z, np.sqrt(x**2 + y**2))
    # Calculate longitude in radians
    lng_rad = np.arctan2(y, x)
    # Convert latitude and longitude from radians to degrees
    lat = np.rad2deg(lat_rad)
    lng = np.rad2deg(lng_rad)
    return lat, lng