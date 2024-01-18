import numpy as np
import pyproj
from pyproj import Proj, Transformer
import glob
import re
import json

def create_source_mesh(x0,x1,y0,y1,ps):
    """
    This function creates a 2D mesh for which each cell is associated with a rectangular source and contains its 4 (x, y) coordinates.

    Parameters
    ----------
    x0 : float
        Smallest x value.
    x1 : float
        Largest x value.
    y0 : float
        Smallest y value.
    y1 : float
        Largest y value.
    ps : ndarray
        Load distribution.

    Returns
    -------
    ndarray
        Numpy array of shape (ny, nx, 4, 2).
    """
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
    """
    Calculates the strain tensor at a given point in the mesh.

    Parameters
    ----------
    Up : ndarray
        Array containing displacements in x, y, and z at the point, before the point, and after the point to compute second-order derivatives.
        Up = [[u_x-1,u_x+1],[y_x-1,y_x+1],[z_x-1,z_x+1]]
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    dz : float
        Grid spacing in the z direction.

    Returns
    -------
    ndarray
        Strain tensor at the given point.
    """
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


def latlon2xy(latitudes, longitudes, elevations=None, projection_type='utm', zone=None, ellps='WGS84'):
    """
    Convert latitude, longitude, and elevation coordinates to x and y coordinates.

    Parameters:
    - latitudes: List of latitude coordinates.
    - longitudes: List of longitude coordinates.
    - elevations: List of elevation coordinates (optional).
    - projection_type: Projection type, e.g., 'utm'.
    - zone: UTM zone (required if using UTM projection).
    - ellps: Ellipsoid for the projection, e.g., 'WGS84'.

    Returns:
    - x: List of x coordinates.
    - y: List of y coordinates.
    """

    # Create a projection
    projection = Proj(proj=projection_type, zone=zone, ellps=ellps)

    # Convert latitude, longitude, elevation to x, y
    x, y = projection(longitudes, latitudes)

    # if elevations:
    #     return x, y, elevations
    # else:
    return x, y


def xy2latlon(x, y, projection_type='utm', zone=None, ellps='WGS84'):
    """
    Convert x and y coordinates back to latitude and longitude.

    Parameters:
    - x: List of x coordinates.
    - y: List of y coordinates.
    - projection_type: Projection type, e.g., 'utm'.
    - zone: UTM zone (required if using UTM projection).
    - ellps: Ellipsoid for the projection, e.g., 'WGS84'.

    Returns:
    - latitudes: List of latitude coordinates.
    - longitudes: List of longitude coordinates.
    """
    # Create a projection
    projection = Proj(proj=projection_type, zone=zone, ellps=ellps)

    # Inverse transformation from x, y to latitude, longitude
    transformer = Transformer.from_proj(Proj(proj=projection_type, zone=zone, ellps=ellps), Proj(proj='latlong', ellps='WGS84'))
    longitudes, latitudes = transformer.transform(x, y)

    return latitudes, longitudes



def is_point_inside_polygon(polygon, x, y):    
    """
    Function that determines if a point is inside a polygon. Useful for mapping pressure distribution front contours.

    Parameters
    ----------
    polygon : list of tuples
        List of (x, y) coordinates representing the polygon.
    x : float
        x-coordinate of the point.
    y : float
        y-coordinate of the point.

    Returns
    -------
    bool
        True if the point is inside the polygon, False otherwise.
    """
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def second_deriv_Lcurve(norm_regu,misfit,lambda_list,lambda_range):
    """
    Computes the second derivative of the L curve, allowing detection of the elbow of the L-curve.

    Parameters
    ----------
    norm_regu : ndarray
        Array containing the norm of the regularizer (x axis of the L-curve).
    misfit : ndarray
        Array containing the norm of the misfit (y axis of the L-curve).
    lambda_list : ndarray
        Array containing the regularization parameter values.
    lambda_range : tuple
        Tuple containing the range of lambda values to consider.

    Returns
    -------
    ndarray
        x values sorted and cropped to the provided range of lambda.
    ndarray
        Second derivative of the L curve.
    ndarray
        Lambda values sorted and cropped to the provided range.
    """
    sorted_indices = np.argsort(norm_regu)
    x_sorted = norm_regu[sorted_indices]
    y_sorted = misfit[sorted_indices]
    lambda_list_sorted = lambda_list[sorted_indices]
    idx2keep = np.where((lambda_list_sorted>=lambda_range[0])&(lambda_list_sorted<=lambda_range[1]))[0]
    x_sorted_cropped = x_sorted[idx2keep]
    y_sorted_cropped = y_sorted[idx2keep]
    lambda_list_cropped = lambda_list_sorted[idx2keep]
    dy_dx = np.gradient(y_sorted_cropped, x_sorted_cropped)
    d2y_dx2 = np.gradient(dy_dx, x_sorted_cropped)
    return x_sorted_cropped, d2y_dx2, lambda_list_cropped


def log_run(E,v,lamb,epsilon,sigma,constraint,it,maxit,elapsed_time,logpath):
    """
    Creates a log file for the inversion.

    Parameters
    ----------
    E : float
        Young's Modulus.
    v : float
        Poisson's ratio.
    lamb : float
        Coefficient for the regularizer term.
    epsilon : float
        Convergence criterion.
    sigma : float
        Standard deviation.
    constraint : bool or None
        Indicates whether a constraint is used.
    it : int
        Number of iterations.
    maxit : int
        Maximum number of iterations.
    elapsed_time : float
        Elapsed time for the inversion.
    logpath : str
        Path to store the log file.
    """
    if isinstance(constraint,np.ndarray)==True:
        constraint = True
    dic = {
    'E': E,
    'v': v,
    'epsilon': epsilon,
    'lamb' : lamb, 
    'sigma' : sigma,
    'constraint' : constraint,
    'iterations' : it,
    'maxit' : maxit,
    'elapsed_time' : elapsed_time
    }
    filename = f'{logpath}/meta_sig_{sigma}_lamb_{lamb}.json'    
    with open(filename, 'w') as file:
        json.dump(dic, file)


def extract_part(file_name):
    """
    Function used to extract all the names of the metadata files in a directory for importing results in the notebook.

    Parameters
    ----------
    file_name : str
        Full file name including path.

    Returns
    -------
    str
        Extracted part of the file name.
    """
    start_index = file_name.rfind('/meta') + 5
    end_index = file_name.rfind('.json')
    return file_name[start_index:end_index]
