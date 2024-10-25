from state_objects.Rigid import *

# Test functions for rigid body and related operations
def test_rigid_body():
    # Initialize parameters for rigid body
    mass = 5.0
    I_body = wp.mat33(1.0, 0.0, 0.0, 
                      0.0, 1.0, 0.0, 
                      0.0, 0.0, 1.0)  # Example inertia tensor (identity for simplicity)
    
    pos = wp.vec3(1.0, 2.0, 3.0)  # Initial position
    quat = wp.quat(1.0, 0.0, 0.0, 0.0)  # Initial orientation (identity quaternion)
    v = wp.vec3(0.1, 0.2, 0.3)  # Initial velocity
    w = wp.vec3(0.01, 0.02, 0.03)  # Initial angular velocity

    # Create RigidBody object
    rbody = RigidBody()

    # Initialize the rigid body
    rigid_body_init(rbody, mass, I_body, pos, quat, v, w)

    # Print initial values
    print("Initial Rigid Body State:")
    print("Mass:", rbody.mass)
    print("Inertia Tensor (Body Frame):", rbody.I_body)
    print("Inverse Inertia Tensor (Body Frame):", rbody.I_body_inv)
    print("Position:", rbody.state.pos)
    print("Quaternion (Orientation):", rbody.state.quat)
    print("Velocity:", rbody.state.v)
    print("Angular Velocity:", rbody.state.w)

    # Test world_to_body_coords function
    world_point = wp.vec3(4.0, 5.0, 6.0)  # Example world coordinates
    body_point = world_to_body_coords(rbody, world_point)
    print("\nTransformed Point (World to Body Coordinates):", body_point)

    # Test body_to_world_coords function
    world_point_transformed_back = body_to_world_coords(rbody, body_point)
    print("Transformed Point (Body to World Coordinates):", world_point_transformed_back)

    # Test updating rigid state
    new_pos = wp.vec3(2.0, 4.0, 6.0)
    new_quat = wp.quat(0.707, 0.0, 0.707, 0.0)  # Example rotation quaternion
    new_v = wp.vec3(0.5, 0.6, 0.7)
    new_w = wp.vec3(0.02, 0.04, 0.06)

    update_rigid_state(rbody.state, new_pos, new_quat, new_v, new_w)
    
    print("\nUpdated Rigid Body State:")
    print("Position:", rbody.state.pos)
    print("Quaternion (Orientation):", rbody.state.quat)
    print("Velocity:", rbody.state.v)
    print("Angular Velocity:", rbody.state.w)

if __name__ == '__main__':
    test_rigid_body()
