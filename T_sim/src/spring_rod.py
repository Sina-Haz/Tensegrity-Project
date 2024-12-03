import taichi as ti
import taichi.math as tm
from data import *
from quat import *
from cables import Spring
from rigid import RigidBody, rigid_state

ti.init(arch=ti.cpu)

# These points are where the springs are fixed (can think fixed to a wall)
fixed_points = ti.Vector.field(3, dtype=default_dtype, shape = (2,))
fixed_points[0] = vec3(-1, 0, 6)
fixed_points[1] = vec3(1, 0, 6)

# These points are where the springs are connected to the rod
attachment_points = ti.Vector.field(3, dtype=default_dtype, shape=(2,))
attachment_points[0] = vec3(-1, 0, 4)
attachment_points[1] = vec3(1, 0, 4)

# Need to use taichi fields to store our data so we can initialize and simulate them within kernels
springs = Spring.field(shape=2)
rbs = RigidBody.field(shape=1)

# Globally defined rod length since we don't have cylinder object yet
rod_len = 2
rod_radius = .5
quat = vec4(tm.cos(tm.pi/ 4), tm.sin(tm.pi/ 4)*vec3(0,1,0))
print(quat)
# We need to initialize our taichi objects within taichi scope if we want to use them so we need to make an init function:
@ti.kernel
def init_objects():
    # Initialize our springs
    spr1 = Spring(ke=20, kd = 10, L0 = 2, x1 = fixed_points[0], x2 = attachment_points[0])
    spr2 = Spring(ke=10, kd = 10, L0 = 2, x1 = fixed_points[1], x2 = attachment_points[1])
    springs[0] = spr1
    springs[1] = spr2

    # Initialize our rod
    # For the initial state we use position 1 below the "wall" and still aligned on x and y axes
    # we give it some small up velocity and w as well
    rod_state = rigid_state(pos = vec3(0, 0, 4), quat = quat, v = vec3(0, 0, 0), w = (0, 0, 0) )

    # rod properties:
    mass = 1
    rlen = rod_len
    I_x = I_y = (1/12)*mass*rlen**2 + (1/4)*mass*rod_radius**2# moment of inertia about x and y
    I_z = .5 *mass*rod_radius**2# neglegible for thin rod
    I_body = ti.Matrix([[I_x, 0, 0], [0, I_y, 0], [0, 0, I_z]])  # Diagonal inertia
    I_body_inv = I_body.inverse()

    rod = RigidBody(mass = 1, I_body = I_body, I_body_inv = I_body_inv, state = rod_state)
    rbs[0] = rod

# compute endpoints based on CoM of the rod and orientation by mapping from local to world
@ti.func
def get_endpts(rod: RigidBody):
    locals = [vec3(0,0,-rod_len/2), vec3(0,0,rod_len/2)]
    world = [rod.body_to_world(locals[0]), rod.body_to_world(locals[1])]
    return world

# Now we want to integrate this system for a Duration of 3 seconds with small timesteps dt = 0.1
duration = 0.5
dt = 1e-3
# currT has to be a taichi field to use it in a taichi kernel
currT = ti.field(dtype=ti.f32, shape=())  # Current time
currT[None] = 0.0  # Initialize current time

# only 3 forces acting on our system: f_g, f_spr1, f_spr2
# At every timestep:
#  - Assert fixed points + attachment points
#  - Compute endpoint velocity from v_t, w_t of rod
#  - Sum spring forces and gravity and come up with a and alpha
#  - apply this to rod

@ti.kernel
def simulate():
    # Get our objects back from the gloablly defined fields
    spr1 = springs[0]
    spr2 = springs[1]
    rod = rbs[0]
    ct = 0
    while currT[None] < duration:
        # Assert points:
        spr1.x1 = fixed_points[0]
        spr2.x1 = fixed_points[1]
        endpts = get_endpts(rod)
        spr1.x2 = endpts[0]
        spr2.x2 = endpts[1]

        # print(f'spr1 endpoints: {spr1.x1}, {spr1.x2}, \n spr2 endpoints: {spr2.x1}, {spr2.x2}')

        # Compute endpoint velocities:
        # eqn: v_endpt = v_CoM + w crossprod (x_endpt - x_CoM)
        v_e1 = rod.state.v + tm.cross(rod.state.w, spr1.x2 - rod.state.pos)
        v_e2 = rod.state.v + tm.cross(rod.state.w, spr2.x2 - rod.state.pos)

        # Compute spring force and f_g
        fspr1 = spr1.force(vec3(0,0,0), v_e1)
        fspr2 = spr2.force(vec3(0,0,0), v_e2)
        f_g = rod.mass * vec3(0, 0, -9.81)

        # Compute net force and torque
        net_force = fspr1+fspr2+f_g
        # Equation for torque: torque_i = (r_t - CoM) cross F_ext
        net_torque = tm.cross((spr1.x2 - rod.state.pos), fspr1) + tm.cross((spr2.x2 - rod.state.pos), fspr2)
        # if ct % 100 == 0: print(net_force)

        # Using the net force and torque we can calc linear and angular accel
        a = net_force / rod.mass
        alpha = rod.I_t_inv() @ net_torque

        # Update our velocities using this:
        rod.state.v += a * dt
        rod.state.w += alpha * dt

        # Update position and orientation:
        rod.state.pos += rod.state.v * dt # x += dx/dt * dt
        # # For quaternions: dq/dt = 1/2 * (q * w_q)
        # rod.state.quat += .5 * quat_mul(rod.state.quat, vec4(0, *rod.state.w)) * dt
        # rod.state.quat = rod.state.quat.normalized()
        # Exponentiated update rule for quaternions (more stable):
        rod.state.quat = quat_mul(rod.state.quat, quat_exp(.5 * dt * vec4(0, rod.state.w)))

        print(f'position: {rod.state.pos:.3f}, orientation: {rod.state.quat:.3f}, linear velocity: {rod.state.v:.3f}, angular velocity: {rod.state.w:.3f}')
        currT[None]+= dt
        ct += 1

init_objects()
simulate()



