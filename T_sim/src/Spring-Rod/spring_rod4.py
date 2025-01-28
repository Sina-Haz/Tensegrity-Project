import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from world import *

# Build world from config:
world = World('spring_rod_cfg.json')

# 2) Define simulation hyperparameters and simulate:
duration = 1
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

            for j in ti.static(range(world.max_sites)):
                f_net += site_spr_force[j,:]
                r = world.globals[i,j] - world.rbs[i].state.pos
                tau_net += tm.cross(r, site_spr_force[j, :])

            # Add gravity
            f_net += world.rbs[i].mass * world.g
            
            # Use net force and torque to compute linear and angular acceleration
            a = f_net / world.rbs[i].mass
            alpha = world.rbs[i].I_t_inv() @ tau_net

            # Update rigid body state
            world.rbs[i].state.v += a * dt
            world.rbs[i].state.w += alpha * dt
            world.rbs[i].state.pos += world.rbs[i].state.v * dt
            world.rbs[i].state.quat = quat_mul(world.rbs[i].state.quat, quat_exp(.5 * dt * vec4(0, world.rbs[i].state.w)))

            # Print state:
            print(f'position: {world.rbs[i].state.pos:.3f}, orientation: {world.rbs[i].state.quat:.3f}, linear velocity: {world.rbs[i].state.v:.3f}, angular velocity: {world.rbs[i].state.w:.3f}')

        # 3) Update global site locations and update spring attachments
        world.update_globals()
        world.update_spring_attachments()

        # 4) Update time
        currT[None] += dt

simulate(world)


# This used to part of the world class before I saw that we can write simulate separately: 
'''

    @ti.kernel
    def Euler_Simulate(self, duration: float, dt: float):
        while(self.time[None] < duration):

            # 1) Compute spring site velocities and populate spring forces
            fspr = ti.Matrix.zero(dt = default_dtype, n = self.n_springs, m = 3) # 3D force vector for each spring
            for i in ti.static(range(self.n_springs)):
                v1, v2 = self.compute_endpoint_velocities(i)
                fspr[i, :] = self.springs[i].force(v1, v2)


            # 2) Compute net f_ext and tau_ext using spring force mappings
            for i in ti.static(range(self.n_bodies)): # ti.static may cause a bug here
                map_matrix = self.spring_force_map[i]
                site_spr_force = map_matrix @ fspr # give m x 3 matrix of spring forces at each site
                f_net = vec3(0,0,0)
                tau_net = vec3(0,0,0)

                for j in ti.static(range(self.max_sites)):
                    f_net += site_spr_force[j,:]
                    r = globals[i,j] - self.rbs[i].state.pos
                    tau_net += tm.cross(r, site_spr_force[j, :])
                
                # Add gravity
                f_net += self.rbs[i].mass * self.g
                
                # Use net force and torque to compute linear and angular acceleration
                a = f_net / self.rbs[i].mass
                alpha = self.rbs[i].I_t_inv() @ tau_net

                # Update rigid body state
                self.rbs[i].state.v += a * dt
                self.rbs[i].state.w += alpha * dt
                self.rbs[i].state.pos += self.rbs[i].state.v * dt
                self.rbs[i].state.quat = quat_mul(self.rbs[i].state.quat, quat_exp(.5 * dt * vec4(0, self.rbs[i].state.w)))

                # Print state:
                print(f'position: {self.rbs[i].state.pos:.3f}, orientation: {self.rbs[i].state.quat:.3f}, linear velocity: {self.rbs[i].state.v:.3f}, angular velocity: {self.rbs[i].state.w:.3f}')
            
            # 3) Update global site locations and update spring attachments
            self.update_globals()
            self.update_spring_attachments()

            # 4) Update time
            self.time[None] += dt

'''