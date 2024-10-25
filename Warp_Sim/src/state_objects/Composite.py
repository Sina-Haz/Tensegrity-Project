from state_objects.Rigid import *

class CompositeBody:
    def __init__(self, v: wp.vec3, w: wp.vec3, quat: wp.quat, rigid_names: list[str], rigid_bodies: list[RigidBody], sites: list[str]) -> None:
        self.state = rigid_state() # The state of our overall body

        # This boolean helps us determine whether we need to update inner body states after we update overall composite state
        self.inner_bodies_updated = True 

        # We can keep track of all the rigid bodies using this dictionary which tells us which rigid body is at what index
        self.name_to_idx = {name:i for i, name in enumerate(rigid_names)}

        # A warp array of all the rigid body structs
        self._rigid_bodies = wp.array(rigid_bodies, dtype=RigidBody)

        # Overall mass is sum of all the component masses
        mass_array, com_array = wp.array([0.], dtype=float), wp.array(wp.vec3())
        wp.launch(compute_total_mass, dim=len(self._rigid_bodies), inputs=[self._rigid_bodies, mass_array])
        self.mass = mass_array[0]

        # Can find the center of mass by doing a weighted sum of all body positions * fraction of mass they take up
        wp.launch(compute_com, dim = len(self._rigid_bodies), inputs=[self._rigid_bodies, com_array, self.mass])
        com = com_array[0]

        self.state.pos = com
        self.state.quat = quat
        self.state.v = v
        self.state.w = w

        self.I_body = self._compute_inertia_tensor(com, quat)

        self.body_offsets = self._compute_body_vecs(com, quat)
    

    def _compute_inertia_tensor(self, com, quat) -> wp.mat33:
        '''
        Computes the inertia tensor of the composite body by applying parallel axis theorem and summing the 
        inertia tensors of its component rigid bodies
        '''
        rot_mat = wp.quat_to_matrix(quat)
        rot_mat_inv = wp.inverse(rot_mat)
        I_body_total = wp.array(wp.mat33()) # 0 initialization

        # Use the helper kernel to compute the inertia tensor
        wp.launch(comp_inertia_helper, dim=len(self._rigid_bodies), inputs = [self._rigid_bodies, com, rot_mat_inv, I_body_total])

        return I_body_total[0]
    
    def _compute_body_vecs(self, com, quat) -> wp.array(dtype=wp.vec3):
        '''
        Compute the positions of each rigid body relative to composite CoM and orientation and store
        in a dictionary that matches rigid body name to offset
        '''
        vecs = wp.array(dtype=wp.vec3, shape = (self.rigid_bodies[0], ))
        rot_mat = wp.quat_to_matrix(quat)
        rot_mat_inv = wp.inverse(rot_mat)

        wp.launch(kernel=compute_offsets, dim= vecs.shape[0], inputs=[self._rigid_bodies, com, rot_mat_inv, vecs])
        
        return vecs
    
    def update_state(self, pos, quat, v, w) -> None:
        '''
        Updates the state variables and notes that the component bodies haven't been updated yet
        '''
        update_rigid_state(self.state, pos, quat, v, w)
        self.inner_bodies_updated = False

    @property
    def rigid_bodies(self) -> wp.array(dtype=RigidBody):
        if not self.inner_bodies_updated:
            wp.launch(kernel=update_all_rigid_bodies, dim=self._rigid_bodies.shape[0], 
                      inputs=[self._rigid_bodies, self.body_offsets, self.state])
            self.inner_bodies_updated = True

        return self._rigid_bodies
            



@wp.func
def parallel_axis_offset(I_body: wp.mat33, mass: wp.float32, offset: wp.vec3) -> wp.mat33:
    """
    Computes the inertia tensor adjusted for an offset using the parallel axis theorem.
    
    Args:
        I_body: The inertia tensor of the rigid body in its local frame.
        mass: The mass of the rigid body.
        offset: The offset vector from the center of mass to the new reference point.

    Returns:
        I_offset: The adjusted inertia tensor considering the parallel axis offset.
    """
    # Compute the outer product of the offset vector with itself (offset ⊗ offset)
    r_outer = wp.outer(offset, offset)

    # Compute the squared magnitude of the offset vector (r dot r)
    r_squared = wp.dot(offset, offset)

    # Create an identity matrix
    identity = wp.identity(3, dtype=wp.float32)

    # Apply the parallel axis theorem formula
    I_offset = I_body + mass * (r_squared * identity - r_outer)

    return I_offset

@wp.kernel
def compute_total_mass(bodies: wp.array(dtype=RigidBody), mass_arr: wp.array(dtype=float)):
    '''
    This kernel sums all the masses of bodies and atomically adds them to the first index of the warp array total_mass
    Atomic add operations need to write to a mutable memory location (i.e. an array)
    '''
    tid = wp.tid()
    wp.atomic_add(mass_arr, 0, bodies[tid].mass)

@wp.kernel
def compute_com(bodies: wp.array(dtype=RigidBody), com_arr: wp.array(dtype=wp.vec3), total_mass: wp.float32):
    '''
    Similar to previous kernel, computes the CoM by taking each component CoM and doing a weighted sum of them 
    based on what fraction of the composite's total mass they represent
    '''
    tid = wp.tid()
    weighted = bodies[tid].pos * (bodies[tid].mass / total_mass)
    wp.atomic_add(com_arr, 0, weighted)


@wp.kernel
def comp_inertia_helper(rbodies: wp.array(dtype=RigidBody), com: wp.vec3, rot_mat_inv: wp.mat33, I_body_total: wp.array(dtype = wp.mat33)):
    '''
    This kernel is used to compute the total inertia tensor of composite body using its center of mass, rotation matrix
    as well as the variables inside all of its component rigid bodies. How it works is it takes the regular inertia tensor
    of each body, computes it's offset and then atomically adds it to the overall inertia tensor
    '''
    tid = wp.tid()

    rb = rbodies[tid]
    # find the offset b/w this rb CoM and overall CoM in world coordinates
    offset_world = rb.pos - com
    # Convert this offset to body frame by multiplying it with inverse of rotation matrix of Composite
    offset_body = rot_mat_inv @ offset_world
    # Compute inertia tensor I body now w.r.t. composite CoM using parallel axis theorem
    I_b = parallel_axis_offset(rb.I_body, rb.mass, offset_body)
    wp.atomic_add(I_body_total,0, I_b)

@wp.kernel
def compute_offsets(bodies: wp.array(dtype=RigidBody), com: wp.vec3, rot_mat_inv: wp.mat33, vecs: wp.array(wp.vec3)):
    tid = wp.tid()
    offset_world = bodies[tid].state.pos - com
    offset_body = rot_mat_inv @ offset_world
    vecs[tid] = offset_body

@wp.kernel
def update_all_rigid_bodies(bodies: wp.array(dtype=RigidBody), offsets: wp.array(dtype=wp.vec3), composite_state: rigid_state):
    '''
    This kernel updates all of the states of the rigid bodies of the composite based on their respective offsets and the
    new composite state. It assumes children rigid bodies are only being influenced by parent
    '''
    tid = wp.tid()
    rot_mat = wp.quat_to_matrix(composite_state.quat)

    # First we compute the position of the body in the world frame based on its offset from composite CoM
    bvec = offsets[tid]
    wvec = (rot_mat @ bvec) + composite_state.pos

    # Total linear velocity is sum of parent linear velocity + angular velocity that aligns with it
    v = composite_state.v + wp.cross(composite_state.w, bvec)
    update_rigid_state(bodies[tid].state, wvec, composite_state.quat, v, composite_state.w)











