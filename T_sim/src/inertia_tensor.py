from data import *
import taichi.math as tm


'''
This file defines functions that compute the inertia tensor based on formulas for certain common shapes

These are python scope functions but return taichi data structures so that it's compatible with rigid.py, etc.
'''

def cylinder(mass, length, radius, dtype = default_dtype) -> ti.Matrix:
    '''
    Get bodyframe inertia tensor for cylinder
    '''
    # I_x and I_y:
    I_x = I_y = (1 / 12) * mass * length ** 2 + (1 / 4) * mass * radius ** 2
    I_z = (1 / 2) * mass * radius**2

    I_body = ti.Matrix([[0] * 3 for _ in range(3)], dtype)
    I_body[0,0] = I_x
    I_body[1, 1] = I_y
    I_body[2,2] = I_z

    return I_body

def hollow_cylinder(mass, length, radius_out, radius_in, dtype = default_dtype) -> ti.Matrix:
    '''
    Get bodyframe inertia tensor for hollow cylinder

    :param radius_out: outer radius of hollow cylinder
    :param radius_in: inner radius of hollow cylinder
    '''
    # Use this as radius param in regular cylinder computation
    sum_sq_r = radius_out**2 + radius_in**2
    
    return cylinder(mass, length, sum_sq_r, dtype=dtype)


def solid_sphere(mass, radius) -> ti.Matrix:
    '''
    Get bodyframe inertia tensor for solid sphere

    dtype is float64, same as default_dtype
    '''
    x = (2./5.) * mass * radius**2

    return ti.Matrix.diag(3, x)


