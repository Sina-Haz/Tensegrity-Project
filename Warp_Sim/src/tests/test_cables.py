from state_objects.Cables import *

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
        ke=3.0, kd=0.2, L0=5.0, 
        x1=wp.vec3(0.0, 0.0, 0.0), 
        x2=wp.vec3(0.0, 6.0, 0.0), 
        winch_r=0.05, 
        motor=motor
    )

    print("\nTesting Actuated Cable:")
    print("Initial Actuated Cable Length:", compute_length(act_cable))
    
    # Compute and print force before update
    v1_act = wp.vec3(0.2, 0.1, 0.0)  # Velocity for endpoint 1
    v2_act = wp.vec3(-0.1, 0.1, 0.0)  # Velocity for endpoint 2
    print("Actuated Cable Force (before update):", compute_force(act_cable, v1_act, v2_act))

    # Apply some control input and update the cable
    control_signal = wp.float32(0.5)  # Example control signal for motor actuation
    cable_len = compute_length(act_cable)  # Get the current length of the cable
    dt = wp.float32(0.01)  # Time step for update
    
    update_cable(act_cable, control_signal, cable_len, dt)

    # Print updated length and forces
    print("Actuated Cable Length (after update):", compute_length(act_cable))
    print("Actuated Cable Force (after update):", compute_force(act_cable, v1_act, v2_act))

    # Reset cable and check if it resets correctly
    reset_cable(act_cable)
    print("Actuated Cable Length (after reset):", compute_length(act_cable))
