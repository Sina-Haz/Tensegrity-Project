import warp as wp
from utils.define import default_dtype, vec3, mat33
import numpy as np


# Holds all time varying data for rigid body
@wp.struct
class rigid_state:
    pos: vec3 # (x,y,z) coordinates of CoM
    quat: wp.quat # quaternion representing orientation
    v: vec3 # linear velocity
    w: vec3 # angular velocity, direction represents axis of rotation and magnitude is speed of rotation


@wp.struct
class RigidBody:
    mass: default_dtype # Static mass
    I_body:mat33 # Inertia tensor
    I_body_inv:mat33
    state: rigid_state
    # sites: rigid_sites # Not sure what to do about this, pretty sure this won't compile here

@wp.func
def rigid_body_init(rbody: RigidBody, mass: float, I_body: mat33, pos: vec3, quat: wp.quat, v: vec3, w: vec3) -> None:
    rbody.mass = mass
    rbody.I_body = I_body
    rbody.I_body_inv = wp.inverse(I_body)
    rbody.state = rigid_state()
    rbody.state.pos = pos
    rbody.state.quat = quat
    rbody.state.v = v
    rbody.state.w = w

@wp.func
def world_to_body_coords(rbody: RigidBody, world_coords: vec3) -> vec3:
    """
    Transforms a point from world coordinates to the rigid body's local (body) coordinates.

    Args:
        rbody (RigidBody): The rigid body containing the position and orientation (quaternion).
        world_coords (vec3): The point in world coordinates to be transformed.

    Returns:
        vec3: The transformed point in the body's local coordinate frame.
    
    This function first translates the world coordinates to be relative to the rigid body's
    center of mass and then applies the inverse of the rigid body's rotation (using the quaternion)
    to get the coordinates in the local frame.
    """
    rot_mat = wp.quat_to_matrix(rbody.state.quat)
    rot_mat_inv = wp.inverse(rot_mat)
    rel_coords = world_coords - rbody.state.pos
    body_coords = rot_mat_inv @ rel_coords
    return body_coords

@wp.func
def body_to_world_coords(rbody: RigidBody, body_coords: vec3) -> vec3:
    """
    Transforms a point from the rigid body's local (body) coordinates to world coordinates.

    Args:
        rbody (RigidBody): The rigid body containing the position and orientation (quaternion).
        body_coords (vec3): The point in the body's local coordinate frame to be transformed.

    Returns:
        vec3: The transformed point in world coordinates.
    
    This function applies the rigid body's rotation (using the quaternion) to the local coordinates
    and then translates the result by the body's position to get the coordinates in the world frame.
    """
    rot_mat = wp.quat_to_matrix(rbody.state.quat)
    new_coords = (rot_mat @ body_coords) + rbody.state.pos
    return new_coords

@wp.func
def update_rigid_state(r_state: rigid_state, pos: vec3, quat: wp.quat, v: vec3, w:vec3) -> None:
    r_state.pos = pos
    r_state.quat = quat
    r_state.v = v
    r_state.w = w


if __name__ == '__main__':
    # Set up example data
    mass = 1.0
    I_body = mat33(np.eye(3, dtype=default_dtype))
    initial_pos = wp.vec3(0.0, 0.0, 0.0)
    initial_quat = wp.quat(1.0, 0.0, 0.0, 0.0)  # No rotation
    initial_v = wp.vec3(1.0, 0.0, 0.0)  # Linear velocity along x-axis
    initial_w = wp.vec3(0.0, 1.0, 0.0)  # Angular velocity along y-axis

    # Initialize a RigidBody instance
    rbody = RigidBody()
    rigid_body_init(rbody, mass, I_body, initial_pos, initial_quat, initial_v, initial_w)

    # Print the initialized state
    print("Initial RigidBody State:")
    print(f"Position: {rbody.state.pos}")
    print(f"Quaternion: {rbody.state.quat}")
    print(f"Linear Velocity: {rbody.state.v}")
    print(f"Angular Velocity: {rbody.state.w}")

    # Test coordinate transformations
    world_point = wp.vec3(1.0, 1.0, 1.0)
    body_point = world_to_body_coords(rbody, world_point)
    print(f"\nWorld to Body Coordinates:\nWorld Point: {world_point} -> Body Point: {body_point}")

    # Convert back to world coordinates to check correctness
    new_world_point = body_to_world_coords(rbody, body_point)
    print(f"\nBody to World Coordinates:\nBody Point: {body_point} -> World Point: {new_world_point}")

    # Update the rigid body state and print the results
    new_pos = wp.vec3(2.0, 2.0, 2.0)
    new_quat = wp.quat(0.707, 0.707, 0.0, 0.0)  # 90-degree rotation around x-axis
    new_v = wp.vec3(0.0, 1.0, 0.0)  # New linear velocity
    new_w = wp.vec3(0.0, 0.0, 1.0)  # New angular velocity around z-axis

    update_rigid_state(rbody.state, new_pos, new_quat, new_v, new_w)

    print("\nUpdated RigidBody State:")
    print(f"Position: {rbody.state.pos}")
    print(f"Quaternion: {rbody.state.quat}")
    print(f"Linear Velocity: {rbody.state.v}")
    print(f"Angular Velocity: {rbody.state.w}")



