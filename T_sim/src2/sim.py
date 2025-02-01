import numpy as np
from data import *

"""
In this file we will define all of the functions that operate over our data and embody our simulation pipeline

Simulation pipeline:
    step:
         - updateSites() -> in: body position, velocity, omega, orientation || out: global_pos, site_velocity
         - getTendonForce() -> in: global_pos, site_velocity, Stiffness, damping, type || out: tendon forces
         - BodyForceTorque() -> in: tendon forces, attach1 id, attach2 id, sites.body_id || out: body forces, body torques
         - FwdDynamics() -> in: body forces, body torques, mass, inertia tensor || out: updated body position, velocity, orientation
         - contacts() ... future worries

    BE CAREFUL WHEN USING Numpy indexing and +=,

    Leads to unexpected behavior b/c indexing creates a temporary copy, and thus when using repeated indices, only
    the last update is kept, they don't cancel out. Instead use np.add.at
"""

def updateSites(body_position, body_velocities, body_omegas, body_orientations, body_id, local_pos):
    """
    This is the first step of simulation pipeline, updates the global position and velocity of all the sites, 
    based on updated body states that were obtained from the previous step. This is needed to compute the tendon forces
    since they attach to the bodies at these sites.

    Args:
      length: N_BODIES
        body_position: An array of positions of CoM of all rigid bodies
        body_velocities: An array of linear velocities of all rigid bodies
        body_omegas: An array of angular velocities of all rigid bodies
        body_orientations: An array of all orientations of rigid bodies (as quaternions)
     
     length: K_SITES
        body_id: An array of body indices which correspond to the body of each site we are updating
        local_pos: An array of the site locations w.r.t to their bodies' local reference frame

    Returns:
     length: K_SITES
        global_site_pos: An array of all the updated global site positions 
        site_velocities: An arry of all the updated site velocities
    """
    # Firstly, we index the quantities and store them in contiguous memory
    com_pos = body_position[body_id].copy()
    com_V = body_velocities[body_id].copy()
    com_W = body_omegas[body_id].copy()
    com_Q = body_orientations[body_id].copy()

    # Using this we can map the locals to global coordinates
    global_site_pos = LocalToGlobal(local_pos, com_pos, com_Q)

    site_velocities = np.zeros_like(global_site_pos)

    # Site velocity = linear V + angular W x (global_pos - com_pos)
    site_velocities += com_V
    site_velocities += np.cross(com_W, (global_site_pos - com_pos))

    return global_site_pos, site_velocities


    
def getTendonForces(global_pos, site_V, tendons: Tendons):
    """
    This function takes the site data generated at the previous step and uses it to compute tendon forces for each
    of the M_TENDONS. 

    Args:
     length: K_SITES
        global_pos: Array of global positions of the sites
        site_V: Array of site velocities

     length: M_TENDONS
        tendons: A Tendons data structure holding arrays with information about stiffness, damping, rest lengths,
        and attachment point site ids

    Returns:
     length: M_TENDONS
        tendon_forces: an array of 3D vectors representing the force of each tendon
    """
    # index global positions and site velocity to get an array of x1 and x2 attachment points, as well as their respective velocities
    x1, x2 = global_pos[tendons.attach1_id], global_pos[tendons.attach2_id]
    v1, v2 = site_V[tendons.attach1_id], site_V[tendons.attach2_id]

    # Get relative positions and velocities
    x_rel, v_rel = x2 - x1, v2 - v1
    current_lengths = np.linalg.norm(x_rel, axis=1, keepdims=True)
    unit = x_rel / current_lengths

    # Total force is sum of spring force and damping force
    fs = - tendons.ke * (current_lengths.squeeze(-1) - tendons.rest_len)
    fd = - tendons.kd * np.sum(v_rel * unit, axis=1)

    # add an extra dimension to force magnitudes so that it multiplies each unit vector of same index
    tendon_forces = (fs[..., np.newaxis] + fd[..., np.newaxis]) * unit

    # This creates a boolean array of tendons which are both cables, and which are not being stretched but rather compressed
    # In this scenario they should have no force
    cable_mask = (tendons.type == 1) & (current_lengths.squeeze(-1) <= tendons.rest_len)
    tendon_forces[cable_mask] *= 0.0

    return tendon_forces

# spring force (fs): -0.000196200000
# spring force (fs): -0.000098100000


def BodyForceTorque(tendon_force, attach1_id, attach2_id, global_site_pos, body_id, body_position):
    """
    Compute the force and torque on each body, based on tendon forces and site locations. 
    NOTE: This function DOES NOT add gravity to the forces, this will be done at the next step in the pipeline

    Args:
     length: M_TENDONS
        tendon_force: An array of tendon forces 
        attach1_id: Site id of first attachment point
        attach2_id: Site id of 2nd attachment point
    
     length: K_SITES
        global_site_pos: Absolute coordinates of where all the sites are
        body_id: The id of the body each site belongs to

     length: N_BODIES
        body_positions: An array of absolute coordinates for the center of mass of each body

    Returns:
     length: N_BODIES
        body_force: Force on each body as a 3D vector
        body_torque: Torque on each body as a 3D vector
    """
    # Zero initialize body forces and torques
    n_bodies, n_sites = len(body_position), len(global_site_pos)
    body_force, body_torque = np.zeros((n_bodies, 3)), np.zeros((n_bodies, 3))

    # First we need to obtain force at each site:
    site_force = np.zeros((n_sites, 3))
    # For every tendon force it applies negative direction of force to site with 1st attachment point, positive for 2nd
    np.add.at(site_force, attach1_id, -tendon_force)
    np.add.at(site_force, attach2_id, tendon_force)

    # Add each site force to the body it has an id for
    np.add.at(body_force, body_id, site_force)

    # For torque, first compute the radius or "lever" by seeing how far the site is from the body CoM
    r = global_site_pos - body_position[body_id]
    np.add.at(body_torque, body_id, np.cross(r, site_force))

    return body_force, body_torque


def FwdDynamics(body_force, body_torque, rbs: Bodies, env: Env):
    """
    Given forces and torques acting on the body, compute forward dynamics based on the timestep dt and 
    return updated body states.
    NOTE: Gravity force is added at this step in the pipeline before computing forward dynamics

    Args:
     length: N_BODIES
        body_force: An array of forces acting on the bodies
        body_torque: An array of torques acting on the bodies
        rbs: A 'Bodies' data structure holding important information such as state data, mass, inertia, etc.
    
     length: None
        env: Environment context variable containing quantites like dt, gravitational acceleration, etc.

    Returns:
        None, updates rbs directly.
    """
    # Add gravitational force
    body_force += rbs.mass[..., np.newaxis] * env.g

    # Compute the inverse of the inertia tensor in world frame to get angular acceleration
    R = quat_to_matrix(rbs.Q)
    R_T = np.transpose(R, axes=(0, 2, 1))
    I_t_inv = np.matmul(R, np.matmul(rbs.I_b_inv, R_T))

    # Get linear and angular acceleration
    accel = body_force / rbs.mass[..., np.newaxis]
    alpha = np.matmul(I_t_inv, body_torque[..., np.newaxis]).squeeze(-1)

    # If any of the bodies are fixed then we apply this to zero out their acceleration and keep their state unchanged
    if env.fixed is not None:
        accel[env.fixed] *= 0.0
        alpha[env.fixed] *= 0.0

    rbs = EulerIntegrate(accel, alpha, rbs, env.dt)


def EulerIntegrate(acc, alpha, bodies: Bodies, dt: float):
    """
    Update state based on acceleration (linear and angular), with Euler's method

    Args:
     length: N_BODIES
        acc: array of linear accelerations of all the bodies
        alpha: array of angular accelerations of all bodies
        bodies: Data structure holding body info
    
     length: None
        dt: timestep size

    Returns:
     length: N_BODIES
        bodies: Data structure with body info, with updated state data
    """
    bodies.V += acc * dt
    bodies.W += alpha * dt
    bodies.P += bodies.V * dt

    omega_as_q = np.zeros_like(bodies.Q)
    omega_as_q[:, 1:] = bodies.W
    bodies.Q = quat_mul_batch(bodies.Q, quat_exp(omega_as_q * 0.5 * dt))

    return bodies


def euler_step(rbs: Bodies, sites: Sites, tendons: Tendons, env: Env):
    """
    Completes a timestep for updating body states, the sites, etc.
    """
    # First get/update the global positions and velocities of the sites
    sites.global_pos, sites.site_V = updateSites(rbs.P, rbs.V, rbs.W, rbs.Q, sites.body_id, sites.local_pos)

    # Next using these updated global positions and velocities compute the force of each tendon
    tendon_force = getTendonForces(sites.global_pos, sites.site_V, tendons)

    # Translate these tendon forces into body forces and torques
    bf, bt = BodyForceTorque(tendon_force, tendons.attach1_id, tendons.attach2_id, sites.global_pos, sites.body_id, rbs.P)

    # Do forward dynamics and update states, (updates bodies directly)
    FwdDynamics(bf, bt, rbs, env)

