import taichi as ti
import taichi.math as tm
from data import *
from quat import *
from cables import Spring
from rigid import RigidBody, rigid_state
from inertia_tensor import *
import json

ti.init(arch=ti.cpu)

def quat_to_matrix_py(q) -> mat33:
    '''
    quat to matrix function in python scope
    '''
    w, x, y, z = q
    return ti.Matrix([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

@ti.kernel
def init_field(field: ti.template(), val: ti.template()):
    for i in ti.grouped(field):
        field[i] = val


# What I need is to store rigid bodies, store springs, and use attachment points to build
# Some sort of matrix that maps spring forces to the sites on the rigid bodies

with open('spring_rod_cfg.json', 'r') as file:
    data = json.load(file)

# Parameters needed to initialize our world data structures (as they are not dynamic storage like python lists)
n_bodies = len(data['bodies'])
n_springs = len(data['springs'])
max_sites_per_body = 2

# Initialize world data (for now world is only made up of rigid bodies and springs)
rbs = RigidBody.field(shape=n_bodies)
springs = Spring.field(shape=n_springs)
locals = ti.Vector.field(3, dtype=default_dtype, shape=(n_bodies, max_sites_per_body))
globals = ti.Vector.field(3, dtype=default_dtype, shape=(n_bodies, max_sites_per_body))
g = vec3(0, 0, -9.81)

@ti.func
def update_globals():
    for i, j in ti.ndrange(locals.shape[0], locals.shape[1]):
        globals[i, j] = rbs[i].body_to_world(locals[i, j])

'''
Mapping spring forces to bodies:
1. Each body has some vector of sites (m x 1) -> this is row i of body_sites
2. We have n springs with spring force at each timestep (n x 1)
3. For each body then we have a matrix (m x n) s.t  maps spring forces to each attachment point
'''

# Populate Rigid Bodies from JSON data
for i, body_data in enumerate(data['bodies']):

    # Initialize state data into taichi classes
    position = vec3(body_data['state']['pos'])
    endpts = body_data['endpoints']
    rot = quat_from_endpts(*endpts)
    velocity = vec3(body_data['state']['velocity'])
    omega = vec3(body_data['state']['omega'])

    # Get rigid body state
    rstate = rigid_state(pos = position, quat = rot, v = velocity, w = omega)
    mass = body_data['mass']

    # Get I_body and I_body_inv
    if body_data['type'].lower() == 'cylinder':
        I_body = cylinder(mass,body_data['length'], body_data['radius'])
        I_body_inv = I_body.inverse()

    rb = RigidBody(state = rstate, mass = mass, I_body = I_body, I_body_inv = I_body_inv)

    rbs[i] = rb

    # Finally we make sure to add sites to our global body sites
    q_inv = rb.state.quat * -1
    q_inv[0] *= -1
    R = quat_to_matrix_py(q_inv)
    for j in range(len(body_data['sites'])):
        site = vec3(body_data['sites'][j])
        
        # Rotate site by the inverse of quaternion rotation so that site is in local frame of reference (w/ principal axis = z)
        rotated_to_local = R @ site
        locals[i, j] = rotated_to_local


# Call this to populate global site positions
# update_globals()
# HARDCODED FOR NOW:
globals[0,0] = vec3(-1, 0, 4)
globals[0,1] = vec3(1,0,4)

# Now we load in the springs
for i, spr_data in enumerate(data['springs']):
    # Load in x1:
    x1 = spr_data['x1']
    if 'fixed' in x1:
        x1 = vec3(x1['fixed']) # Don't need to apply a transformation to this point
    else:
        site_idx = x1['site']
        x1 = globals[*site_idx]
    
    x2 = spr_data['x2']
    if 'fixed' in x2:
        x2 = vec3(x2['fixed']) # Don't need to apply a transformation to this point
    else:
        site_idx = x2['site']
        x2 = globals[*site_idx]


    spr = Spring(ke=spr_data['ke'], kd = spr_data['kd'], L0 = spr_data['L0'], x1=x1, x2=x2)
    springs[i] = spr


# Now that we have initialized all of the springs and rigid bodies lets build the spring force map as described above
# Recall that in our map for each body we have an m x n matrix where m = number of sites and n = number of springs
map = ti.Matrix.field(max_sites_per_body, n_springs, dtype=ti.i32, shape = (n_bodies, ))

# TODO: Test which is right direction 
def build_map():
    # First initialize map to all zeros:
    init_field(map, ti.Matrix([[0] * max_sites_per_body for _ in range(n_springs)]))

    for spr_idx, spr_data in enumerate(data['springs']):
        # If x1 is a site then we want to apply negative force by convention
        x1 = spr_data['x1']
        if 'site' in x1:
            body_idx, site_idx = x1['site']
            map[body_idx][site_idx, spr_idx] = -1
            
        # If x2 is a site then we want to apply positive force by convention
        x2 = spr_data['x2']
        if 'site' in x2:  # If it's attached to a body
            body_idx, site_idx = x2['site']
            map[body_idx][site_idx, spr_idx] = 1


# Now we want to integrate this system for a Duration of 3 seconds with small timesteps dt = 0.1
duration = 3
dt = 1e-3
# currT has to be a taichi field to use it in a taichi kernel
currT = ti.field(dtype=ti.f32, shape=())  # Current time
currT[None] = 0.0  # Initialize current time
build_map()



@ti.kernel
def simulate2():
    ct = 0
    while currT[None] < duration:

        # populate spring forces 
        # ASK NELSON HOW TO AUTOMATE THE COMPUTING OF ENDPOINT VELOCITIES, for now copying how I did it last time
        fspr = ti.Matrix.zero(dt = default_dtype, n = n_springs, m = 3)
        for i in ti.static(range(n_springs)):
            v1 = vec3(0,0,0)
            # I think to get this we need to somehow invert our mapping to figure out a map from spring to site and body
            v2 = rbs[0].state.v + tm.cross(rbs[0].state.w, springs[i].x2 - rbs[0].state.pos)

            fspr[i, :] = springs[i].force(v1, v2)

        # print(springs[0].x2, springs[1].x2)

        # To get net f_ext and tau_ext for each rigid body we use our mapping to get a matrix and then use that to sum forces
        # And then do something a bit more complicated for tau
        for i in range(n_bodies):
            map_matrix = map[i] # Gives an m x n matrix where m is # of sites, n is # of springs
            site_spr_force = map_matrix @ fspr # give m x 3 matrix of spring forces at each site
            f_net = vec3(0,0,0)
            tau_net = vec3(0,0,0)

            for j in ti.static(range(max_sites_per_body)):
                f_net += site_spr_force[j,:]
                r = globals[i,j] - rbs[i].state.pos
                tau_net += tm.cross(r, site_spr_force[j, :])

            # Add gravity
            f_net += rbs[i].mass * g
            # print(f_net)

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

        
        # Update global site locations:
        update_globals()
        # print(globals[0,0], globals[0,1])

        # Update spring endpoint locations (FOR NOW HARDCODED)
        springs[0].x1 = vec3(-1, 0, 6)
        springs[0].x2 = globals[0, 0]

        springs[1].x1 = vec3(1, 0, 6)
        springs[1].x2 = globals[0, 1]

        # Update time: 
        currT[None] += dt
        ct+=1




simulate2()
# Checked that rigid body and springs have same initialization but not the same forces?


                






