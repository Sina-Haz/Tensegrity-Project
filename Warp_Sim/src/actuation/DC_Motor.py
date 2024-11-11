import warp as wp
from typing import Any
import numpy as np
from utils.define import default_dtype, vec3, mat33


@wp.struct
class MotorState:
    omega_t: default_dtype # Angular velocity of a motor



@wp.func
def reset(m_state: MotorState):
    '''
    Reset the motor's angular velocity to 0
    '''
    m_state.omega_t = default_dtype(0)




@wp.struct
class DCMotor:
    max_omega: default_dtype
    speed: default_dtype
    state: MotorState

@wp.func
def motor_init(motor: DCMotor, speed: default_dtype):
    motor.max_omega = 220*2*wp.pi / 60
    motor.speed = default_dtype(speed)
    motor.state = MotorState()
    reset(motor.state)

@wp.func
def compute_cable_length_delta(motor: Any, control: Any, winch_r: Any, delta_t: Any,  dim_scale: Any = 1.0) -> default_dtype:
    """
    Computes the change in cable length (delta_l) based on motor control, winch radius, time step, 
    and an optional dimensional scaling factor.

    Parameters:
    -----------
    motor : DCMotor
        The motor object that contains the current motor state, maximum angular velocity, and speed.
    control : default_dtype
        A control input for the motor, typically a value between -1 and 1, where 1 represents full 
        forward control, -1 represents full reverse, and 0 represents no control.
    winch_r : default_dtype
        The radius of the winch to which the cable is attached. This is used to convert angular 
        velocity to linear velocity of the cable.
    delta_t : default_dtype
        The time step over which the change in cable length is calculated.
    dim_scale : default_dtype, optional (default=1.0)
        A scaling factor applied to the computed cable length delta. This can be used for unit 
        conversions or other dimensional adjustments.

    Returns:
    --------
    delta_l : default_dtype
        The change in cable length based on the motor's angular velocity and control input.

    Notes:
    ------
    - The motor's angular velocity (`omega_t`) is updated based on the control input, and the average 
      of the previous and updated angular velocities is used to compute the linear displacement of 
      the cable over the given time step.
    - The function assumes the control is applied for the entire duration of the time step `delta_t`.
    """
    start_omega = motor.state.omega_t # get starting motor angular velocity

    # Convert all variables to float, do all computation with float
    delta_omega = start_omega
    speed = motor.speed
    max_omega = motor.max_omega
    control = control

    # compute new angular velocity based on controls and speed
    delta_omega = speed * max_omega * control
    
    # Finally based on angular velocity compute the change in cable length
    delta_l = (start_omega + delta_omega) / 2.0 * winch_r * dim_scale * delta_t

    motor.state.omega_t = default_dtype(delta_omega)

    return default_dtype(delta_l)



# Prev fn signature: 
# def compute_cable_length_delta(motor: DCMotor, control: default_dtype, winch_r: default_dtype, delta_t: default_dtype,  dim_scale: default_dtype = 1.0):

if __name__ == '__main__':
    motor = DCMotor()  # Initialize motor with a speed of 0.6
    motor_init(motor, default_dtype(0.6))
     # Apply some control input and update the cable
    control_signal = default_dtype(0.5)  # Example control signal for motor actuation
    dt = default_dtype(0.01)  # Time step for update
    winch_r = default_dtype(0.05)

    compute_cable_length_delta(motor, control_signal, winch_r, dt)
