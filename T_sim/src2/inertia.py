import numpy as np
from quat import *

"""
These functions 
"""

def cylinder_inertia(mass, length, radius):
    '''
    Get bodyframe inertia tensor for cylinder
    '''
    # I_x and I_y:
    I_x = I_y = (1 / 12) * mass * length ** 2 + (1 / 4) * mass * radius ** 2
    I_z = (1 / 2) * mass * radius**2

    I_body = np.diag([I_x, I_y, I_z])

    return I_body

def hollow_cylinder_inertia(mass, length, radius_out, radius_in):
    '''
    Get bodyframe inertia tensor for hollow cylinder
    '''
    # Use this as radius param in regular cylinder computation
    sum_sq_r = radius_out**2 + radius_in**2
    
    return cylinder_inertia(mass, length, sum_sq_r)


def solid_sphere_inertia(mass, radius):
    '''
    Get bodyframe inertia tensor for solid sphere
    '''
    x = (2. / 5.) * mass * radius
    return np.eye(3) * x


