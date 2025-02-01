import numpy as np
from quat import *

"""
In this python file we designate all of the data that is necessary for our simulation

ECS architecture:
loosely, each class can be thought of as an "Entity", each is a SoA where each array is a "Component"
the functions that will operate over such data (aka our "Systems) for simulation will be written elsewhere. 

here we will only define a few necessary functions for transforming and organizing the data
"""

def GlobalToLocal(global_coords, body_positions, body_orientations):
    """
    Takes 3 arrays of equal first dimension

    global_coords: shape = (batch_sz, 3), the absolute position of coordinates
    body_positions: shape = (batch_sz, 3), position of the CoM of body each point is in reference to
    body_orientations: shape = (batch_sz, 4), orientation as a quaternion of body each point is in reference to

    returns: local_coords: shape = (batch_sz, 3)
    """
    rot_matrs = quat_to_matrix(body_orientations)
    rot_matr_invs = np.transpose(rot_matrs, axes = (0, 2, 1))

    offset = global_coords - body_positions

    # Add extra dimension to offsets to allow for matrix multiplication element-wise, then squeeze them back down
    local_coords = np.matmul(rot_matr_invs, offset[..., np.newaxis]).squeeze(-1)
    return local_coords


def LocalToGlobal(local_coords, body_positions, body_orientations):
    """
    Takes 3 arrays of equal first dimension

    local_coords: shape = (batch_sz, 3), the position of coordinates in body space
    body_positions: shape = (batch_sz, 3), position of the CoM of body each point is in reference
    body_orientations: shape = (batch_sz, 4), orientation as a quaternion of body we want to compute local reference w.r.t.

    returns: global_coords: shape = (batch_sz, 3)
    """
    rot_matrs = quat_to_matrix(body_orientations)

    # Add extra dimension to coordinates so that we can do an element wise matrix multiply to rotate each point, then squeeze back down
    rot_coords = np.matmul(rot_matrs, local_coords[..., np.newaxis]).squeeze(-1)

    global_coords = rot_coords + body_positions
    return global_coords


class Bodies:
    def __init__(self, n_bodies, positions: np.ndarray, quats: np.ndarray, V: np.ndarray, W: np.ndarray, mass: np.ndarray, I_b: np.ndarray):
        assert positions.shape == (n_bodies, 3)
        assert quats.shape == (n_bodies, 4)
        assert V.shape == (n_bodies, 3)
        assert W.shape == (n_bodies, 3)
        assert mass.shape == (n_bodies, )
        assert I_b.shape == (n_bodies, 3, 3)

        # State data (get's overwritten every timestep)
        self.P = positions
        self.Q = quats
        self.V = V
        self.W = W

        # static data
        self.n_bodies = n_bodies
        self.mass = mass
        self.I_b = I_b
        self.I_b_inv = np.linalg.inv(I_b)


    def getBody(self, idx: int):
        return {
            "pos": self.P[idx], 
            "velocity": self.V[idx],
            "quat": self.Q[idx],
            "omega": self.W[idx],
            "mass": self.masses[idx],
            "Inertia": self.I_b[idx]
            }
    

class Sites:
    def __init__(self, n_sites, body_id: np.ndarray, global_pos: np.ndarray, body_positions, body_orientations):
        assert body_id.shape == (n_sites, )
        assert global_pos.shape == (n_sites, 3)

        
        self.n_sites = n_sites
        self.body_id = body_id # what body each site is on (just an index)
        self.global_pos = global_pos # This will be updated every timestep
        self.local_pos = GlobalToLocal(global_pos, body_positions[body_id], body_orientations[body_id]) # immutable once written
        self.site_V = np.zeros_like(global_pos) # Initialize site velocities to 0's

    def getSite(self, idx):
        return {
            "body_id": self.body_id[idx],
            "global_pos": self.global_pos[idx],
            "local_pos": self.local_pos[idx],
            "site_velocity": self.site_V[idx]
        }


class Tendons:
    def __init__(self, n_tendons, stiffness: np.ndarray, damping: np.ndarray, rest_len: np.ndarray, attach1_id: np.ndarray, attach2_id: np.ndarray, type: np.ndarray):
        assert stiffness.shape == (n_tendons, )
        assert damping.shape == (n_tendons, )
        assert rest_len.shape == (n_tendons, )
        assert attach1_id.shape == (n_tendons, )
        assert attach2_id.shape == (n_tendons, )
        assert type.shape == (n_tendons, )

        self.n_tendons = n_tendons
        self.ke = stiffness # stiffness for each spring, immutable
        self.kd = damping # damping coeff for each spring, immutable
        self.rest_len = rest_len # rest length for each spring, immutable
        self.attach1_id = attach1_id # site id for the first attachment point (negative force)
        self.attach2_id = attach2_id # site id for the 2nd attachment point (positive force)
        self.type = type # Assign a boolean array of 0 or 1 for whether this tendon is a spring or not. (0=spring, 1=cable)

    def getSpring(self, idx: int):
        return {
            "ke": self.ke[idx],
            "kd": self.kd[idx],
            "L0": self.rest_len[idx],
            "x1_id": self.attach1_id[idx],
            "x2_id": self.attach2_id[idx],
            "type": self.type[idx]
        }


class Env:
    """
    A context class that stores metadata about the environment and simulation
    """
    def __init__(self, n_bodies, n_sites, n_tendons, dt, g, duration, fixed=None):
        self.nb = n_bodies
        self.ns = n_sites
        self.nt = n_tendons
        self.dt = dt
        self.g = g
        self.duration = duration
        self.fixed = fixed # can set this to be a list of body indices which should be fixed (i.e. their state doesn't change)
