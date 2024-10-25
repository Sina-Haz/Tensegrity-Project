from actuation.DC_Motor import *

import warp as wp

def main():
    # Initialize warp
    wp.init()

    # Parameters for testing
    motor_speed = 0.5  # Speed factor of the motor (0 to 1 scale)
    control_input = 0.75  # Control input between -1 and 1
    winch_radius = 0.1  # Radius of the winch in meters
    delta_time = 0.1  # Time step in seconds
    dim_scale = 1.0  # No scaling for this test

    # Create an instance of DCMotor
    motor = DCMotor(motor_speed)

    # Print the initial motor state
    print(f"Initial Motor State: omega_t = {motor.state.omega_t}")

    # Compute the change in cable length
    delta_l = compute_cable_length_delta(motor, control_input, winch_radius, delta_time, dim_scale)

    # Print the results
    print(f"Control Input: {control_input}")
    print(f"Winch Radius: {winch_radius}")
    print(f"Time Step: {delta_time}")
    print(f"Change in Cable Length (delta_l): {delta_l}")
    print(f"Updated Motor State: omega_t = {motor.state.omega_t}")

    # Reset motor state
    reset(motor.state)
    print(f"Motor State after Reset: omega_t = {motor.state.omega_t}")

if __name__ == "__main__":
    main()
