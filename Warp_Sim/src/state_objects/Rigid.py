import warp as wp

# Holds all time varying data for rigid body
@wp.struct
class rigid_state:
    pos: wp.vec3 # (x,y,z) coordinates of CoM
    quat: wp.quat # quaternion representing orientation
    v: wp.vec3 # linear velocity
    w: wp.vec3 # angular velocity, direction represents axis of rotation and magnitude is speed of rotation


@wp.struct
class RigidBody:
    mass: wp.float32 # Static mass
    I_body:wp.mat33 # Inertia tensor
    I_body_inv:wp.mat33
    state: rigid_state
    # sites: rigid_sites # Not sure what to do about this, pretty sure this won't compile here

@wp.func
def rigid_body_init(rbody: RigidBody, mass: float, I_body: wp.mat33, pos: wp.vec3, quat: wp.quat, v: wp.vec3, w: wp.vec3) -> None:
    rbody.mass = mass
    rbody.I_body = I_body
    rbody.I_body_inv = wp.inverse(I_body)
    rbody.state = rigid_state()
    rbody.state.pos = pos
    rbody.state.quat = quat
    rbody.state.v = v
    rbody.state.w = w

@wp.func
def world_to_body_coords(rbody: RigidBody, world_coords: wp.vec3) -> wp.vec3:
    """
    Transforms a point from world coordinates to the rigid body's local (body) coordinates.

    Args:
        rbody (RigidBody): The rigid body containing the position and orientation (quaternion).
        world_coords (wp.vec3): The point in world coordinates to be transformed.

    Returns:
        wp.vec3: The transformed point in the body's local coordinate frame.
    
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
def body_to_world_coords(rbody: RigidBody, body_coords: wp.vec3) -> wp.vec3:
    """
    Transforms a point from the rigid body's local (body) coordinates to world coordinates.

    Args:
        rbody (RigidBody): The rigid body containing the position and orientation (quaternion).
        body_coords (wp.vec3): The point in the body's local coordinate frame to be transformed.

    Returns:
        wp.vec3: The transformed point in world coordinates.
    
    This function applies the rigid body's rotation (using the quaternion) to the local coordinates
    and then translates the result by the body's position to get the coordinates in the world frame.
    """
    rot_mat = wp.quat_to_matrix(rbody.state.quat)
    new_coords = (rot_mat @ body_coords) + rbody.state.pos
    return new_coords

@wp.func
def update_rigid_state(r_state: rigid_state, pos: wp.vec3, quat: wp.quat, v: wp.vec3, w:wp.vec3) -> None:
    r_state.pos = pos
    r_state.quat = quat
    r_state.v = v
    r_state.w = w


