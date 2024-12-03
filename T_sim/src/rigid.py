import taichi as ti
import taichi.math as tm
from data import default_dtype, vec4, vec3, mat33
from quat import *

@ti.dataclass
class rigid_state:
    pos: vec3 # (x,y,z) coordinates of CoM
    quat: vec4 # quaternion representing orientation
    v: vec3 # linear velocity
    w: vec3 # angular velocity, direction represents axis of rotation and magnitude is speed of rotation

    @ti.func
    def update(self, x, q, v, w):
        self.pos = x
        self.quat = q
        self.v = v
        self.w = w


@ti.dataclass
class RigidBody:
    mass: default_dtype # Static mass
    I_body:mat33 # Inertia tensor
    I_body_inv:mat33
    state: rigid_state

    @ti.func
    def world_to_body(self, world_coords: vec3) -> vec3:
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
        R = quat_to_matrix(self.state.quat)
        R_inv = tm.inverse(R)
        rel_coords = world_coords - self.state.pos
        body_coords = R_inv @ rel_coords
        return body_coords
    
    @ti.func
    def body_to_world(self, body_coords: vec3) -> vec3:
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
        R = quat_to_matrix(self.state.quat)
        world_coords = (R @ body_coords) + self.state.pos
        return world_coords
    
    @ti.func
    def I_t(self) -> mat33:
        '''
        Computes the new "Inertia Tensor" of the rigid body based on it's current orientation and 
        I_body the inertia tensor at initial position and orientation
        Eqn: I_t = R(t)I_bodyR(t)^T
        '''
        R = quat_to_matrix(self.state.quat)
        return R @ self.I_body @ R.transpose()
    
    @ti.func
    def I_t_inv(self) -> mat33:
        '''
        Computes inverse of I_t, just faster to do it this way than compute and then invert
        '''
        R = quat_to_matrix(self.state.quat)
        return R @ self.I_body_inv @ R.transpose()
    
    

# Define the kernel to test the rigid body transformations
@ti.kernel
def test_rigid_body_transform():
    # Initialize rigid body with mass, inertia tensor, and state
    body_pos = ti.Vector([1.0, 2.0, 3.0])
    body_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (no rotation)
    body_velocity = ti.Vector([0.0, 0.0, 0.0])
    body_angular_velocity = ti.Vector([0.0, 0.0, 0.0])
    rigid_body = RigidBody(mass=1.0, 
                           I_body=ti.Matrix.identity(ti.f32, 3), 
                           I_body_inv=ti.Matrix.identity(ti.f32, 3), 
                           state=rigid_state(pos=body_pos, quat=body_quat, v=body_velocity, w=body_angular_velocity))

    # Test transformation between world and body coordinates
    world_coords = ti.Vector([4.0, 5.0, 6.0])
    body_coords = rigid_body.world_to_body(world_coords)
    print(body_coords)

    print(rigid_body.body_to_world(body_coords))

if __name__ == '__main__':

    ti.init(arch=ti.cpu)
    test_rigid_body_transform()