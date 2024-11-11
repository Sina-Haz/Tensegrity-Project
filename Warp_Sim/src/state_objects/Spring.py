import warp as wp
from actuation.DC_Motor import *
import numpy as np
from typing import Any, Optional
from utils.define import default_dtype, vec3, mat33

    

@wp.struct
class Spring:
    ke: default_dtype # Spring stiffness
    kd: default_dtype # Spring damping
    L0: default_dtype # Rest length
    x1: vec3 # Endpoint 1
    x2: vec3 # Endpoint 2

@wp.func
def spr_init(spr:Spring, ke: default_dtype, kd: default_dtype, L0: default_dtype, x1: vec3, x2: vec3):
    '''
    Initialize the Spring instance spr with given arguments
    Returns None (does it in place)
    '''
    spr.ke=ke
    spr.kd=kd
    spr.L0=L0
    spr.x1=x1
    spr.x2=x2


@wp.func
def compute_len(spr: Spring) -> default_dtype:
    '''
    Takes in Spring spr and returns length b/w its endpoints
    '''
    return wp.length(spr.x2 - spr.x1)


@wp.func
def compute_force(spr: Spring, v1: vec3, v2: vec3) -> vec3:
    '''
    Takes in velocities of the endpoints of Spring spr.
    Computes the spring force using equation F = ke * (currLen - restLen) - kd * relative velocity
    Force is relative to (endpt2 - endpt1), unit vector
    returns a vec3 force vector 
    Based on warp's own implementation
    '''
    # Get unit direction of spr and length of spring (this will be vec3 and scalar)
    # Get relative velocity
    unit = vec3(wp.normalize(spr.x2 - spr.x1))
    len = compute_len(spr)
    v_rel = v2 - v1

    # Total force is the sum of spring force and kd force
    # Hooke's Law: F_spr = -k*(L - L0) where L0 is rest length
    fs = - spr.ke * (len - spr.L0)

    # Spring damping law:
    fd = - spr.kd * wp.dot(v_rel, unit)

    # Total force is sum of magnitudes along unit direction of the spring
    f_tot = default_dtype((fs + fd)) * unit
    return f_tot


if __name__ == '__main__':
    # Testing Spring
    spr = Spring()

    spr_init(spr, 3.87, 0.3, 4.0, vec3(0.0, 0.0, 0.0), vec3(0.0, 5.0, 0.0))
    
    print("Testing Spring:")
    print("Initial Spring Length:", compute_len(spr))

    # # Random velocities
    v1 = vec3(0.0, 1.0, 0.0)  # Endpoint 1 velocity
    v2 = vec3(0.0, 0.5, 0.0)  # Endpoint 2 velocity

    # print(f'here are the types of each object: spr:{type(spr), isinstance(spr, type(Spring))}')

    # # Compute and print force
    print("Spring Force:", compute_force(spr, v1, v2))