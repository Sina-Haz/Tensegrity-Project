import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)


from world import *


world = World('3-bar-cfg.json')

# 2) Define simulation hyperparameters and simulate:
duration = 3e-3
dt = 1e-3
# currT has to be a taichi field to use it in a taichi kernel
currT = ti.field(dtype=default_dtype, shape=())  # Current time
currT[None] = 0.0  # Initialize current time

fixed = ti.field(dtype=ti.int32, shape = (world.n_bodies, )) 
fixed.fill(0)
fixed[0] = 1 # Set all bodies to 0 (not fixed), except for the first body


@ti.kernel
def simulate(world: ti.template()):
    cnt = 1
    world.update_locals()
    while currT[None] < duration:
        print(f'\ntimestep: {cnt}')

        # 1) Compute spring site velocities and populate spring forces
        fspr = ti.Matrix.zero(dt = default_dtype, n = world.n_springs, m = 3) # 3D force vector for each spring
        for i in ti.static(range(world.n_springs)):
            v1, v2 = world.compute_endpoint_velocities(i)
            fspr[i, :] = world.springs[i].force(v1, v2)
            
        # 2) Compute net f_ext and tau_ext using spring force mappings
        for i in ti.static(range(world.n_bodies)): # ti.static may cause a bug here
            map_matrix = world.spring_force_map[i]
            site_spr_force = map_matrix @ fspr # give m x 3 matrix of spring forces at each site
            f_net = vec3(0,0,0)
            tau_net = vec3(0,0,0)

            for j in ti.static(range(world.max_sites)):
                f_net += site_spr_force[j,:]
                r = world.globals[i,j] - world.rbs[i].state.pos
                t = tm.cross(r, site_spr_force[j, :])
                tau_net += t

            # Add gravity
            f_net += world.rbs[i].mass * world.g

            # Check to see if a body is fixed, if so cancel any force or torque acting on it:
            if fixed[i] == 1:
                f_net = ti.zero(f_net)
                tau_net = ti.zero(tau_net)

            # Use net force and torque to compute linear and angular acceleration
            a = f_net / world.rbs[i].mass
            alpha = world.rbs[i].I_t_inv() @ tau_net

            # Update rigid body state
            world.rbs[i].state.v += a * dt
            world.rbs[i].state.w += alpha * dt
            world.rbs[i].state.pos += world.rbs[i].state.v * dt
            world.rbs[i].state.quat = quat_mul(world.rbs[i].state.quat, quat_exp(.5 * dt * vec4(0, world.rbs[i].state.w)))

            # Print state for non-fixed bodies:
            if i != 0:
                print(f'Body {i}:\nLinear Acceleration: {a:.3f}\nAngular Acceleration: {alpha:.3f}\nPosition: {world.rbs[i].state.pos:.3f}\nOrientation: {world.rbs[i].state.quat:.3f}\nLinear Velocity: {world.rbs[i].state.v:.3f}\nAngular Velocity: {world.rbs[i].state.w:.3f}')

        # 3) Update global site locations and update spring attachments
        world.update_globals()
        world.update_spring_attachments()

        # 4) Update time
        currT[None] += dt
        cnt+=1

simulate(world)