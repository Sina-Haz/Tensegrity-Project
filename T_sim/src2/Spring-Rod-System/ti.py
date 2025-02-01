import os
import sys

# Get the grandparent directory (two levels up)
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Define the paths for `src` and `Spring-Rod` inside `src`
src_dir = os.path.join(grandparent_dir, 'src')
spring_rod_dir = os.path.join(src_dir, 'Spring-Rod')

# Add these directories to `sys.path`
sys.path.insert(0, src_dir)
sys.path.insert(0, spring_rod_dir)

# print("Added paths:")
# print(sys.path[:2])  # Print to confirm

from world import *

# Build world from config:
world = World('spring_rod_cfg.json')

# 2) Define simulation hyperparameters and simulate:
duration = 2e-3
dt = 1e-3
# currT has to be a taichi field to use it in a taichi kernel
currT = ti.field(dtype=ti.f32, shape=())  # Current time
currT[None] = 0.0  # Initialize current time

@ti.kernel
def simulate(world: ti.template()):
    world.update_locals()
    while currT[None] < duration:

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

            # print(site_spr_force)

            for j in ti.static(range(world.max_sites)):
                f_net += site_spr_force[j,:]
                r = world.globals[i,j] - world.rbs[i].state.pos
                tau_net += tm.cross(r, site_spr_force[j, :])

            # Add gravity
            f_net += world.rbs[i].mass * world.g
            
            # Use net force and torque to compute linear and angular acceleration
            a = f_net / world.rbs[i].mass
            print(f'Inertia tensor: {world.rbs[i].I_body}')
            print(f'Inertia tensor inv in world: {world.rbs[i].I_t_inv()}')
            alpha = world.rbs[i].I_t_inv() @ tau_net

            # Update rigid body state
            world.rbs[i].state.v += a * dt
            world.rbs[i].state.w += alpha * dt
            world.rbs[i].state.pos += world.rbs[i].state.v * dt
            world.rbs[i].state.quat = quat_mul(world.rbs[i].state.quat, quat_exp(.5 * dt * vec4(0, world.rbs[i].state.w)))

            # Print state:
            # print(f'angular acceleration: {alpha}')
            print(f'\nposition: {world.rbs[i].state.pos:.12f}\n orientation: {world.rbs[i].state.quat:.12f}\n linear velocity: {world.rbs[i].state.v:.12f}\n angular velocity: {world.rbs[i].state.w:.12f}')

        # 3) Update global site locations and update spring attachments
        world.update_globals()
        world.update_spring_attachments()

        # 4) Update time
        currT[None] += dt

simulate(world)
