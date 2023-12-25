import numpy as np
from scipy.integrate import quad
from numpy import arctan, tan, cos, sin, arccos, arcsin, sqrt, log, log10, pi

def compute_I1(r, z):
    nv = r.shape[0]
    rot = np.array([[0, 1], [-1, 0]])
    I1 = np.array([0, 0])

    for i in range(nv):
        ri = r[i, :]

        if i == nv - 1:
            rj = r[0, :]
        else:
            rj = r[i + 1, :]

        rjio = np.dot(rot, (rj - ri))
        a = np.dot(rj-ri,rj-ri)
        b = np.dot(ri,rj-ri)
        c = np.dot(ri,ri)
        d = c + z**2
        e = d - b**2 / a
        t0 = b / a
        t1 = 1 + b / a

        if np.linalg.norm(rjio) != 0:
            S1i = compute_S1i(a, e, z, t0, t1)
            I1 = I1 + np.dot(S1i, rjio)

    return I1

def compute_I2(r, z):
    nv = r.shape[0]
    rot = np.array([[0, 1], [-1, 0]])
    I2 = 0

    for i in range(nv):
        ri = r[i, :]

        if i == nv - 1:
            rj = r[0, :]
        else:
            rj = r[i + 1, :]

        rjio = np.dot(rot, (rj - ri))
        a = np.dot(rj - ri, rj - ri)
        b = np.dot(ri, rj - ri)
        c = np.dot(ri, ri)
        d = c + z**2
        e = d - b**2 / a
        t0 = b / a
        t1 = 1 + b / a

        if np.linalg.norm(rjio) != 0:
            S2i = compute_S2i(a, e, t0, t1)
            I2 = I2 + np.dot(rjio, S2i)

    return I2

def compute_I3(r, z):
    nv = r.shape[0]
    rot = np.array([[0, 1], [-1, 0]])
    I3 = 0

    for i in range(nv):
        ri = r[i, :]

        if i == nv - 1:
            rj = r[0, :]
        else:
            rj = r[i + 1, :]

        rjo = np.dot(ri, np.dot(rot, rj))
        a = np.dot(rj - ri, rj - ri)
        b = np.dot(ri, rj - ri)
        c = np.dot(ri, ri)
        d = c + z**2
        e = d - b**2 / a
        t0 = b / a
        t1 = 1 + b / a

        if rjo != 0:
            S2i = compute_S2i(a, e, t0, t1)
            I3 = I3 + rjo * S2i

    return I3

def compute_I4(r, z):
    nv = r.shape[0]
    rot = np.array([[0, 1], [-1, 0]])
    I4 = 0

    for i in range(nv):
        ri = r[i, :]

        if i == nv - 1:
            rj = r[0, :]
        else:
            rj = r[i + 1, :]

        rjo = np.dot(ri, np.dot(rot, rj))
        a = np.dot(rj - ri, rj - ri)
        b = np.dot(ri, rj - ri)
        c = np.dot(ri, ri)
        d = c + z**2
        A = c / a - b**2 / a**2
        B = d / a - b**2 / a**2
        t0 = b / a
        t1 = 1 + b / a

        if rjo != 0:
            S3i = compute_S3i(a, A, B, t0, t1)
            I4 = I4 + rjo * S3i

    return I4

def compute_S1i(a, e, z, t0, t1):
    S1i = sqrt(a) * t0 - sqrt(a) * t1

    if e - z**2 != 0:
        S1i = S1i - sqrt(e - z**2) * arctan(sqrt(a) * t0 / sqrt(e - z**2)) + \
              sqrt(e - z**2) * arctan(sqrt(a) * t1 / sqrt(e - z**2)) + \
              sqrt(e - z**2) * arctan(sqrt(a) * t0 * z / (sqrt(e + a * t0**2) * sqrt(e - z**2))) - \
              sqrt(e - z**2) * arctan(sqrt(a) * t1 * z / (sqrt(e + a * t1**2) * sqrt(e - z**2)))

    if z != 0:
        S1i = S1i - z * np.log(a * t0 + np.sqrt(a) * np.sqrt(e + a * t0**2)) + \
              z * log(a * t1 + sqrt(a) * sqrt(e + a * t1**2))

    if t0 != 0:
        S1i = S1i - sqrt(a) * t0 * log(sqrt(e + a * t0**2) + z)

    if t1 != 0:
        S1i = S1i + sqrt(a) * t1 * log(sqrt(e + a * t1**2) + z)

    return S1i / sqrt(a)

def compute_S2i(a, e, t0, t1):
    S2i = log((a * t1 + np.sqrt(a * (a * t1**2 + e))) / (a * t0 + np.sqrt(a * (a * t0**2 + e)))) / sqrt(a)
    return S2i

def compute_S3i(a, A, B, t0, t1):
    S3i = arctan(t1 * sqrt((B - A) / (A * (B + t1**2)))) - arctan(t0 * sqrt((B - A) / (A * (B + t0**2))))
    S3i = S3i * sqrt((B - A) / A) + log((t1 + sqrt(B + t1**2)) / (t0 + sqrt(B + t0**2))) / sqrt(a)
    return S3i


def alpha(r):
    a = 0
    for i in range(len(r)):
        ri = r[i, :]
        if i < len(r) - 1:
            rj = r[i + 1, :]
        else:
            rj = r[0, :]
        if np.linalg.norm(rj) > 0 and np.linalg.norm(ri) > 0:
            aj = np.arctan2(rj[1], rj[0])
            ai = np.arctan2(ri[1], ri[0])
            da = aj - ai
            if da > np.pi:
                da = da - 2 * np.pi
            if da < -np.pi:
                da = da + 2 * np.pi
            if np.abs(da) == np.pi:
                da = 0
            a = a + da
    return a

def load2disp(xyz, r, p, E, v):
    """This function computes the displacement of a given point at coordinates xyz for a given source of vertices r which is a polygon of any shape 
      
    The required shape for xyz is (3,) and the required size for r is (n,2). 
    For a square source has a shape (4,2): [[x_bot_left,y_bot_left],
                                           [x_bot_right],[y_bot_right],
                                           [x_top_right],[y_top_right]
                                           [x_top_left] ,[y_top_left]] -> anti clockwise order"""
    
    # Translating the coordinates of the vertices so that the considered point is located at the middle
    r_translated = np.zeros(r.shape)
    r_translated[:,0] = -r[:,0] + xyz[0] 
    r_translated[:,1] = -r[:,1] + xyz[1]   
    z = xyz[2]
    
    # print(r_translated)
    
    # Lame constants
    l = E * v / ((1 + v) * (1 - 2 * v))
    m = E / (2 * (1 + v))
    
    # Integrals
    I1 = compute_I1(r_translated, z)
    I3 = compute_I3(r_translated, z)
    I4 = compute_I4(r_translated, z)
    
    # Displacements
    U = np.zeros((3, 1))
    U[0:2, 0] = -I1*p/(4*pi*(l+m))
    U[2, 0] = (I3*p/(4*pi*m)) + (I4 * p / (4 * pi * (l + m)))

    if z != 0:
        I2 = compute_I2(r_translated, z)
        al = alpha(r_translated)
        U[0:2, 0] -= I2 * z * p / (4 * np.pi * m)
        U[2, 0] -= z * al * p / (4 * np.pi * (l + m))

    return U

