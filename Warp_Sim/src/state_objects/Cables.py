# Implement all the state objects here using warp structs

# So warp doesn't allow for using strings in User structs so I will define them without that and then use some external way
# To name the structs whenever I call them?

# Base State Object -> Includes name, dtype, device. 
 #  Important functions seem to be set_attribute which is doing some complex nested stuff
 #  Other than that defines some functions to move data from one device to another. Can achieve the same effect w/ overloading

# Spring and Cable:
# Spring -> Extends Base + stiffness, kd, L0, end_pts
 #  Important here is initialize, compute_len, and compute force

import warp as wp
from actuation.DC_Motor import *
import numpy as np
from typing import Any, Optional
import torch

# # Compute generic dot product that computes dot product of 2 vectors along some axis
# # Final output stored in out. For now we just assume axis == 1 
# @wp.kernel
# def row_dot(a: wp.array(dtype=Any),
#             b: wp.array(dtype=Any),
#             out: wp.array(dtype=Any)):
    
#     # This represents the row of the matrix we are doing row-wise dot product on
#     tid = wp.tid()

#     out[tid] = wp.dot(a[tid], b[tid])


    
    

@wp.struct
class Spring:
    ke: wp.float32 # Spring stiffness
    kd: wp.float32 # Spring damping
    L0: wp.float32 # Rest length
    x1: wp.vec3 # Endpoint 1
    x2: wp.vec3 # Endpoint 2

@wp.func
def spr_init(spr:Spring, ke: wp.float32, kd: wp.float32, L0: wp.float32, x1: wp.vec3, x2: wp.vec3):
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
def compute_length(spr: Spring):
    '''
    Takes in Spring spr and returns length b/w its endpoints
    '''
    return wp.length(spr.x2 - spr.x1)


@wp.func
def compute_force(spr: Spring, v1: Any, v2: Any):
    '''
    Takes in velocities of the endpoints of Spring spr.
    Computes the spring force using equation F = ke * (currLen - restLen) - kd * relative velocity
    Force is relative to (endpt2 - endpt1), unit vector
    returns a wp.vec3 force vector 
    Based on warp's own implementation
    '''
    # Get unit direction of spr and length of spring (this will be vec3 and scalar)
    # Get relative velocity
    unit = wp.normalize(spr.x2 - spr.x1)
    len = compute_length(spr)
    v_rel = v2 - v1

    # Total force is the sum of spring force and kd force
    # Hooke's Law: F_spr = -k*(L - L0) where L0 is rest length
    fs = - spr.ke * (len - spr.L0)

    # Spring damping law:
    fd = - spr.kd * wp.dot(v_rel, unit)

    # Total force is sum of magnitudes along unit direction of the spring
    f_tot = (fs + fd) * unit
    return f_tot


@wp.struct
class Cable: # Same variables as spring
    ke: wp.float32 # Spring stiffness
    kd: wp.float32 # Spring damping
    L0: wp.float32 # Rest length
    x1: wp.vec3 # Endpoint 1
    x2: wp.vec3 # Endpoint 2

@wp.func
def cable_init(cable: Cable, ke: wp.float32, kd: wp.float32, L0: wp.float32, x1: wp.vec3, x2: wp.vec3):
    '''
    Initialize the Cable instance cable with given arguments
    Returns None (does it in place)
    '''
    cable.ke=ke
    cable.kd=kd
    cable.L0=L0
    cable.x1=x1
    cable.x2=x2

@wp.func
def compute_len(cable: Cable):
    '''
    Computes length of a cable based on its endpoint position
    '''
    return wp.length(cable.x2 - cable.x1)

@wp.func
def compute_force(cable: Cable, v1: wp.vec3, v2: wp.vec3):
    '''
    Computes the spring force using equation F = ke * (currLen - restLen) - kd * relative velocity
    Only applies force if the cable is stretched, no pushing force for when it's compressed
    Force direction is relative to (endpt2 - endpt1) direction
    '''
    # Get unit direction of cable and length of spring (this will be vec3 and scalar)
    # Get relative velocity
    unit = wp.normalize(cable.x2 - cable.x1)
    len = compute_length(cable)
    v_rel = v2 - v1

    # Only apply tension if the cable is stretched, no pushing
    if len > cable.L0:
        # Hooke's Law: F_cable = -k*(L - L0) when L > L0
        ft = - cable.ke * (len - cable.L0)

        # Damping force: fd = - kd * dot(v_rel, unit)
        fd = - cable.kd * wp.dot(v_rel, unit)

        # Total force is sum of magnitudes along unit direction of the cable
        f_tot = (ft + fd) * unit
    else:
        f_tot = wp.vec3(0.0, 0.0, 0.0)
    
    return f_tot


@wp.struct
class ActuatedCable:
    ke: wp.float32 # Spring stiffness
    kd: wp.float32 # Spring damping
    L0: wp.float32 # Rest length (Base, not taking into acct any actuation length)
    x1: wp.vec3 # Endpoint 1
    x2: wp.vec3 # Endpoint 2
    winch_r: wp.float32 # Winch radius to convert b/w angular and linear velocity
    _winch_r: wp.float32 # Winch radius in space of logits
    min_winch_r: wp.float32 # min winch_r (for when all of cable is unspooled)
    max_winch_r: wp.float32 # max winch_r (for when all of cable wrapped around the winch)
    motor: DCMotor # The DC motor with angular velocity and speed
    act_L0: wp.float32 # Actuation length
    init_act_L0: wp.float32 # Initial actuation length (for easy reset)


def act_cable_init(act_cable: ActuatedCable,
                ke: wp.float32,
                kd: wp.float32,
                L0: wp.float32,
                x1: wp.vec3,
                x2: wp.vec3,
                winch_r: wp.float32, 
                min_winch_r: wp.float32 = 0.01,
                max_winch_r: wp.float32 = 0.07,
                motor: Optional[DCMotor] =None,
                motor_speed: wp.float32 = 0.6,
                init_act_len: wp.float32 = 0.0):
    act_cable.ke=ke
    act_cable.kd=kd
    act_cable.L0=L0
    act_cable.x1=x1
    act_cable.x2=x2
    act_cable.min_winch_r=wp.float32(min_winch_r)
    act_cable.max_winch_r=wp.float32(max_winch_r)
    act_cable.motor=DCMotor(wp.float32(motor_speed)) if motor is None else motor
    act_cable.act_L0=wp.float32(init_act_len)
    act_cable.init_act_L0=wp.float32(init_act_len)
    act_cable._winch_r=_set_winch_r(act_cable, winch_r)




@wp.func
def _set_winch_r(act_cable: ActuatedCable, winch_r: float):
    '''
    Takes cable and winch_r, asserts it's valid (w/in defn range)
    It then normalizes it and puts it into logit space
    Returns float w_r in (-∞, +∞)
    '''
    assert act_cable.min_winch_r <= winch_r <= act_cable.max_winch_r

    act_cable.winch_r = winch_r # If assertion is true then we can keep winch_r in [min, max] winch_r's

    range = act_cable.max_winch_r - act_cable.min_winch_r

    norm_r = (winch_r - act_cable.min_winch_r)/range # norm_r in [0, 1]

    w_r = wp.log(norm_r/(1-norm_r)) # apply logit function to map this normalized value to space of all reals
    return w_r

@wp.func
def reset_cable(act_cable:ActuatedCable):
    '''
    Set actuation length to initial value and motor angular velocity to 0
    '''
    act_cable.act_L0 = act_cable.init_act_L0
    reset(act_cable.motor.state)

@wp.func
def compute_rest_len(act_cable: ActuatedCable):
    '''
    A helper function to dynamically give current remaining rest length based on base rest length and current actuation length
    returns curr_rest_len: float32
    '''
    curr_rest_len = act_cable.L0 - act_cable.act_L0
    return curr_rest_len

@wp.func
def update_cable(act_cable:ActuatedCable, control: wp.float32, cable_len: wp.float32, dt: wp.float32):
    '''
    Updates the actuation rest length and endpoints of the actuated cable based on the control input, winch radius, 
    and time step. Ensures that the actuation length doesn't exceed the original rest length.
     - NOTE: The endpoint update strategy may not be correct as right now we just distribute the change equally
     among each endpoint (maybe in reality one endpoint is fixed and the other isn't and we need to change this function)

    Args:
        act_cable (ActuatedCable): The actuated cable object to update.
        control (wp.float32): Control signal used to determine motor actuation.
        cable_len (wp.float32): Current length of the cable.
        dt (wp.float32): Time step for the simulation.

    Ensures:
        act_cable.act_L0 <= act_cable.L0, so the actuation length does not exceed the original rest length.
    '''
    assert act_cable.act_L0 != None

    # compute change in cable length based on our motor, control and winch radius
    dl = compute_cable_length_delta(act_cable.motor, control, act_cable.winch_r, dt) 

    # We scale this and add to our actuation length
    act_cable.act_L0 += dl * compute_rest_len(act_cable) / cable_len

    # Ensure actuation length is <= base rest length
    act_cable.act_L0 = min(act_cable.act_L0, act_cable.L0) # cable cannot be longer than original rest length

    # Update the positions of the endpoints MAY NEED TO ADJUST LATER
    delta_pos = (act_cable.act_L0 - act_cable.L0) * wp.normalize(act_cable.x2 - act_cable.x1)
    act_cable.x1 += delta_pos * 0.5  # Move x1 towards x2
    act_cable.x2 -= delta_pos * 0.5  # Move x2 towards x1

@wp.func
def compute_force(act_cable: ActuatedCable, v1: wp.vec3, v2: wp.vec3):
    '''
    Use same exact methodology to compute force of Actuated Cable as we did with Cable
     - NOTE that this assumes that the end points update strategy we use in update_cable is correct
     - NOTE this replaces base rest length of cable with compute_rest_len (accounts for actuation length)
    Returns a wp.vec3, force vector aligned along unit direction of the cable
    '''
    dir = act_cable.x2 - act_cable.x1
    unit = wp.normalize(dir)
    len = wp.length(dir)
    v_rel = v2 - v1
    eff_L0 = compute_rest_len(act_cable)

    # Only apply force if cable is under tension
    if len > eff_L0:
        # Hooke's Law: F_act_cable = -k*(L - L0) when L > L0
        ft = - act_cable.ke * (len - eff_L0)

        # Damping force: fd = - kd * dot(v_rel, unit)
        fd = - act_cable.kd * wp.dot(v_rel, unit)

        # Total force is sum of magnitudes along unit direction of the cable
        f_tot = (ft + fd) * unit
    else:
        f_tot = wp.vec3(0.0, 0.0, 0.0)
    
    return f_tot










if __name__ == '__main__':
    # Testing Spring
    spr = Spring()
    spr_init(spr, 3.87, 0.3, 4.0, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 5.0, 0.0))
    
    print("Testing Spring:")
    print("Initial Spring Length:", compute_length(spr))

    # Random velocities
    v1 = wp.vec3(0.0, 1.0, 0.0)  # Endpoint 1 velocity
    v2 = wp.vec3(0.0, 0.5, 0.0)  # Endpoint 2 velocity

    # Compute and print force
    print("Spring Force:", compute_force(spr, v1, v2))

    # Testing Cable
    cable = Cable()
    cable_init(cable, 2.5, 0.2, 3.0, wp.vec3(1.0, 0.0, 0.0), wp.vec3(1.0, 4.0, 0.0))
    
    print("\nTesting Cable:")
    print("Initial Cable Length:", compute_len(cable))

    # Random velocities for Cable
    v1_cable = wp.vec3(0.1, 0.0, 0.0)  # Endpoint 1 velocity
    v2_cable = wp.vec3(-0.1, 0.0, 0.0)  # Endpoint 2 velocity

    # Compute and print force
    print("Cable Force:", compute_force(cable, v1_cable, v2_cable))

    # Test for ActuatedCable can be added similarly once functionality is verified.
    # Testing Actuated Cable
    act_cable = ActuatedCable()
    motor = DCMotor()  # Initialize motor with a speed of 0.6
    motor.speed = wp.float32(0.6)
    act_cable_init(
        act_cable, 
        3.0, 0.2, 5.0, 
        wp.vec3(0.0, 0.0, 0.0), 
        wp.vec3(0.0, 6.0, 0.0), 
        0.05, 
        motor=motor
    )

    print("\nTesting Actuated Cable:")
    print("Initial Actuated Cable Length:", compute_length(act_cable))
    
    # Compute and print force before update
    v1_act = wp.vec3(0.2, 0.1, 0.0)  # Velocity for endpoint 1
    v2_act = wp.vec3(-0.1, 0.1, 0.0)  # Velocity for endpoint 2
    print("Actuated Cable Force (before update):", compute_force(act_cable, v1_act, v2_act))

    # Apply some control input and update the cable
    control_signal = 0.5  # Example control signal for motor actuation
    cable_len = compute_length(act_cable)  # Get the current length of the cable
    dt = 0.01  # Time step for update
    
    update_cable(act_cable, control_signal, cable_len, dt)

    # Print updated length and forces
    print("Actuated Cable Length (after update):", compute_length(act_cable))
    print("Actuated Cable Force (after update):", compute_force(act_cable, v1_act, v2_act))

    # Reset cable and check if it resets correctly
    reset_cable(act_cable)
    print("Actuated Cable Length (after reset):", compute_length(act_cable))


