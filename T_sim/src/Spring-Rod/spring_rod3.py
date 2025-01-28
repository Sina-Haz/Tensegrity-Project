import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

import taichi as ti
import taichi.math as tm
from data import *
from quat import *
from cables import Spring
from rigid import RigidBody, rigid_state
from inertia_tensor import *
import json

ti.init(arch=ti.cpu)

# Helper methods
@ti.func
def update_globals():
    for i, j in ti.ndrange(locals.shape[0], locals.shape[1]):
        globals[i, j] = rbs[i].body_to_world(locals[i, j])

@ti.func
def update_locals():
    for i, j in ti.ndrange(locals.shape[0], locals.shape[1]):
        locals[i, j] = rbs[i].world_to_body(globals[i, j])

@ti.kernel
def init_field(field: ti.template(), val: ti.template()):
    for i in ti.grouped(field):
        field[i] = val

@ti.func
def compute_endpoint_velocities(spr_idx):
    # Compute v1:
    v1 = vec3(0,0,0)
    if spring_site_inds[spr_idx, 0][0] != -1:
        body_idx, _ = spring_site_inds[spr_idx, 0]
        rbstate = rbs[body_idx].state
        v1 = rbstate.v + tm.cross(rbstate.w, springs[spr_idx].x1 - rbstate.pos)
    
    # Compute v2:
    v2 = vec3(0,0,0)
    if spring_site_inds[spr_idx, 1][0] != -1:
        body_idx, _ = spring_site_inds[spr_idx, 1]
        rbstate = rbs[body_idx].state
        v2 = rbstate.v + tm.cross(rbstate.w, springs[spr_idx].x2 - rbstate.pos)

    return v1, v2

@ti.func
def update_spring_attachments():
    for spr_idx in ti.static(range(n_springs)):
        # Update x1
        x1_idx = spring_site_inds[spr_idx, 0]
        if x1_idx[0] != -1: # i.e. if it's not fixed
            springs[spr_idx].x1 = globals[*x1_idx]

        x2_idx = spring_site_inds[spr_idx, 1]
        if x2_idx[0] != -1: # i.e. if it's not fixed
            springs[spr_idx].x2 = globals[*x2_idx]





# 1) PARSE Config data into taichi fields
file = 'spring_rod_cfg.json'

with open(file, 'r') as file:
    data = json.load(file)

# Parameters needed to initialize our world data structures (as they are not dynamic storage like python lists)
n_bodies = len(data['bodies'])
n_springs = len(data['springs'])
max_sites_per_body = 2 # For this eventually make a function or something that determines the max sites needed

# Initialize world data (for now world is only made up of rigid bodies and springs)
rbs = RigidBody.field(shape=n_bodies)
springs = Spring.field(shape=n_springs)
spring_site_inds = ti.Vector.field(2, dtype=ti.i32, shape=(n_springs, 2)) # For each spring store 2 vectors representing indices of sites for spr.x1, spr.x2
spring_force_map = ti.Matrix.field(max_sites_per_body, n_springs, dtype=ti.i32, shape = (n_bodies, )) # Maps spring forces to bodies, described more in previous iteration
locals = ti.Vector.field(3, dtype=default_dtype, shape=(n_bodies, max_sites_per_body))
globals = ti.Vector.field(3, dtype=default_dtype, shape=(n_bodies, max_sites_per_body))
g = vec3(0, 0, -9.81)

def parse_rbs():
    for i, body_data in enumerate(data['bodies']):

        # Initialize state data into taichi classes
        if 'state' in body_data:
            velocity = vec3(body_data['state']['velocity'])
            omega = vec3(body_data['state']['omega'])
        # Default
        else:
            velocity = vec3([0.0, 0.0, 0.0])
            omega = vec3([0.0, 0.0, 0.0])

        endpts = body_data['endpoints']
        position = (vec3(endpts[0]) + vec3(endpts[1])) / 2
        rot = quat_from_endpts(*endpts)
  

        # Get rigid body state
        rstate = rigid_state(pos = position, quat = rot, v = velocity, w = omega)
        mass = body_data['mass']

        # Get I_body and I_body_inv
        if body_data['type'].lower() == 'cylinder':
            I_body = cylinder_inertia(mass,body_data['length'], body_data['radius'])
            I_body_inv = I_body.inverse()

        rb = RigidBody(state = rstate, mass = mass, I_body = I_body, I_body_inv = I_body_inv)

        rbs[i] = rb

        for j in range(len(body_data['sites'])):
            site = vec3(body_data['sites'][j])
            globals[i, j] = site

def parse_springs():
    # First initialize map to all zeros:
    init_field(spring_force_map, ti.Matrix([[0] * max_sites_per_body for _ in range(n_springs)]))

    # Now we load in the springs
    for i, spr_data in enumerate(data['springs']):
        # Load in x1:
        x1 = spr_data['x1']
        if 'fixed' in x1:
            x1 = vec3(x1['fixed']) # Don't need to apply a transformation to this point
            spring_site_inds[i, 0] = ti.Vector([-1, -1], dt=ti.int16) # Add to spring site inds
        else:
            idx = x1['site']
            x1 = globals[*idx] # Use globals as site location
            spring_site_inds[i, 0] = ti.Vector(idx, dt=ti.int16) # Update spring site inds
            # If there is an attachment point on a rigid body we need to update the spring force map
            body_idx, site_idx, spr_idx = idx, i
            map[body_idx][site_idx, spr_idx] = -1
        
        x2 = spr_data['x2']
        if 'fixed' in x2:
            x2 = vec3(x2['fixed']) # Don't need to apply a transformation to this point
            spring_site_inds[i, 1] = ti.Vector([-1, -1], dt=ti.int16)

        else:
            idx = x2['site']
            x2 = globals[*idx]
            spring_site_inds[i, 1] = ti.Vector(idx, dt=ti.int16)
            # If there is an attachment point on a rigid body we need to update the spring force map
            body_idx, site_idx = idx
            spring_force_map[body_idx][site_idx, i] = 1


        spr = Spring(ke=spr_data['ke'], kd = spr_data['kd'], L0 = spr_data['L0'], x1=x1, x2=x2)
        springs[i] = spr

# Call these functions to load data into taichi fields
parse_rbs()
parse_springs()


# 2) Define simulation hyperparameters and simulate:
duration = 1
dt = 1e-3
# currT has to be a taichi field to use it in a taichi kernel
currT = ti.field(dtype=ti.f32, shape=())  # Current time
currT[None] = 0.0  # Initialize current time


@ti.kernel
def simulate():
    update_locals()
    while currT[None] < duration:

        # 1) Compute spring site velocities and populate spring forces
        fspr = ti.Matrix.zero(dt = default_dtype, n = n_springs, m = 3) # 3D force vector for each spring
        for i in ti.static(range(n_springs)):
            v1, v2 = compute_endpoint_velocities(i)
            fspr[i, :] = springs[i].force(v1, v2)
            
        # 2) Compute net f_ext and tau_ext using spring force mappings
        for i in ti.static(range(n_bodies)): # ti.static may cause a bug here
            map_matrix = spring_force_map[i]
            site_spr_force = map_matrix @ fspr # give m x 3 matrix of spring forces at each site
            f_net = vec3(0,0,0)
            tau_net = vec3(0,0,0)

            for j in ti.static(range(max_sites_per_body)):
                f_net += site_spr_force[j,:]
                r = globals[i,j] - rbs[i].state.pos
                tau_net += tm.cross(r, site_spr_force[j, :])

            # Add gravity
            f_net += rbs[i].mass * g
            
            # Use net force and torque to compute linear and angular acceleration
            a = f_net / rbs[i].mass
            alpha = rbs[i].I_t_inv() @ tau_net

            # Update rigid body state
            rbs[i].state.v += a * dt
            rbs[i].state.w += alpha * dt
            rbs[i].state.pos += rbs[i].state.v * dt
            rbs[i].state.quat = quat_mul(rbs[i].state.quat, quat_exp(.5 * dt * vec4(0, rbs[i].state.w)))

            # Print state:
            print(f'position: {rbs[i].state.pos:.3f}, orientation: {rbs[i].state.quat:.3f}, linear velocity: {rbs[i].state.v:.3f}, angular velocity: {rbs[i].state.w:.3f}')

        # 3) Update global site locations and update spring attachments
        update_globals()
        update_spring_attachments()

        # 4) Update time
        currT[None] += dt

simulate()








