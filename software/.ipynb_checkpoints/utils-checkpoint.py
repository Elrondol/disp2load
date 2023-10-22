import numpy as np
import pyproj

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




def lat_lon_to_xy(latitude, longitude, zone=None):
    # Define the projection. You can use UTM, Lambert Conformal Conic, or another projection.
    # Here's an example using a UTM projection for the Great Salt Lake area.
    if zone is None:
        # Determine the UTM zone based on the longitude of the location.
        # In this example, we're using the WGS 84 coordinate system.
        utm_zone = int((longitude + 180) / 6) + 1
        zone = f'EPSG:326{utm_zone:02d}'  # UTM zone for northern Utah
        
    # Create a pyproj transformer.
    transformer = pyproj.Transformer.from_crs("EPSG:4326", zone, always_xy=True)

    # Perform the coordinate conversion.
    x, y = transformer.transform(longitude, latitude)

    return x, y


def xy_to_lat_lon(x, y, zone=None):
    if zone is None:
        # Determine the UTM zone based on the X coordinate.
        utm_zone = int((x + 180) / 6) + 1
        zone = f'EPSG:326{utm_zone:02d}'  # UTM zone for northern Utah

    # Create a pyproj transformer.
    transformer = pyproj.Transformer.from_crs(zone, "EPSG:4326", always_xy=True)

    # Perform the coordinate conversion.
    longitude, latitude = transformer.transform(x, y)

    return latitude, longitude