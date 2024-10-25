import warp as wp

@wp.struct
class MotorState:
    omega_t: wp.float32 # Angular velocity of a motor



@wp.func
def reset(m_state: MotorState):
    '''
    Reset the motor's angular velocity to 0
    '''
    m_state.omega_t = 0




@wp.struct
class DCMotor:
    max_omega: wp.float32
    speed: wp.float32
    state: MotorState

    def __init__(self, speed: wp.float32):
        self.max_omega = 220*2*wp.pi / 60
        self.speed = speed
        self.state = MotorState()

@wp.func
def compute_cable_length_delta(motor: DCMotor, control: wp.float32, winch_r: wp.float32, delta_t: wp.float32,  dim_scale: wp.float32 = 1.0):
    """
    Computes the change in cable length (delta_l) based on motor control, winch radius, time step, 
    and an optional dimensional scaling factor.

    Parameters:
    -----------
    motor : DCMotor
        The motor object that contains the current motor state, maximum angular velocity, and speed.
    control : wp.float32
        A control input for the motor, typically a value between -1 and 1, where 1 represents full 
        forward control, -1 represents full reverse, and 0 represents no control.
    winch_r : wp.float32
        The radius of the winch to which the cable is attached. This is used to convert angular 
        velocity to linear velocity of the cable.
    delta_t : wp.float32
        The time step over which the change in cable length is calculated.
    dim_scale : wp.float32, optional (default=1.0)
        A scaling factor applied to the computed cable length delta. This can be used for unit 
        conversions or other dimensional adjustments.

    Returns:
    --------
    delta_l : wp.float32
        The change in cable length based on the motor's angular velocity and control input.

    Notes:
    ------
    - The motor's angular velocity (`omega_t`) is updated based on the control input, and the average 
      of the previous and updated angular velocities is used to compute the linear displacement of 
      the cable over the given time step.
    - The function assumes the control is applied for the entire duration of the time step `delta_t`.
    """
    start_omega = motor.state.omega_t # get starting motor angular velocity

    # compute new angular velocity based on controls and speed
    motor.state.omega_t = motor.speed * motor.max_omega * control
    
    # Finally based on angular velocity compute the change in cable length
    delta_l = (start_omega + motor.state.omega_t) / 2.0 * winch_r * dim_scale * delta_t

    return delta_l
