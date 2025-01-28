from data import *
from quat import *
import taichi.math as tm


'''
This file defines functions that compute the inertia tensor based on formulas for certain common shapes

These are python scope functions but return taichi data structures so that it's compatible with rigid.py, etc.
'''

@ti.pyfunc
def cylinder_inertia(mass, length, radius, dtype = default_dtype) -> mat33:
    '''
    Get bodyframe inertia tensor for cylinder
    '''
    # I_x and I_y:
    I_x = I_y = (1 / 12) * mass * length ** 2 + (1 / 4) * mass * radius ** 2
    I_z = (1 / 2) * mass * radius**2

    I_body = ti.Matrix([[0.] * 3 for _ in range(3)], dtype)
    I_body[0,0] = I_x
    I_body[1, 1] = I_y
    I_body[2,2] = I_z

    return I_body

@ti.pyfunc
def hollow_cylinder_inertia(mass, length, radius_out, radius_in, dtype = default_dtype) -> mat33:
    '''
    Get bodyframe inertia tensor for hollow cylinder

    :param radius_out: outer radius of hollow cylinder
    :param radius_in: inner radius of hollow cylinder
    '''
    # Use this as radius param in regular cylinder computation
    sum_sq_r = radius_out**2 + radius_in**2
    
    return cylinder_inertia(mass, length, sum_sq_r, dtype=dtype)


def solid_sphere_inertia(mass, radius) -> mat33:
    '''
    Get bodyframe inertia tensor for solid sphere

    dtype is float64, same as default_dtype
    '''
    x = (2./5.) * mass * radius**2

    return ti.Matrix.diag(3, x)


@ti.pyfunc
def composite_inertia(CoM, shapes, quat, dtype = default_dtype) -> mat33:
    R = quat_to_matrix(quat)
    R_inv = R.inverse()
    I_body_total = ti.Matrix([[0] * 3 for _ in range(3)], dtype)

    for s in shapes:
        # find the offset b/w this rb CoM and overall CoM in world coordinates
        offset_world = s.body.state.pos - CoM
        # Convert this offset to body frame by multiplying it with inverse of rotation matrix of Composite
        offset_body = R_inv @ offset_world
        # Compute inertia tensor I body now w.r.t. composite CoM using parallel axis theorem
        I_b = parallel_axis_offset(s.body.I_body, s.body.mass, offset_body)
        I_body_total += I_b

    return I_body_total


@ti.pyfunc
def parallel_axis_offset(I_body: mat33, mass: default_dtype, offset: vec3) -> mat33:
    """
    Computes the inertia tensor adjusted for an offset using the parallel axis theorem.
    
    Args:
        I_body: The inertia tensor of the rigid body in its local frame.
        mass: The mass of the rigid body.
        offset: The offset vector from the center of mass to the new reference point.

    Returns:
        I_offset: The adjusted inertia tensor considering the parallel axis offset.
    """
    # Compute outer product of the offset vector to make matrix consisting of product all possible combinations
    r_outer = offset.outer_product(offset)

    # Compute the squared magnitude of offset vector
    r_squared = offset @ offset

    # Create identity matrix
    id_mat = eye3

    # Apply parallel axis theorem formula
    I_offset = I_body + mass * r_squared * (id_mat - r_outer)
    
    return I_offset


@ti.pyfunc
def inertia_body_to_world(body_inertia: mat33, R: mat33) -> mat33:
    """
    Convert inertia tensor from body frame to world frame via equation:
    I_world = R @ I_body @ R^T
    """