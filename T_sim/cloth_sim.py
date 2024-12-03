import taichi as ti

# simulate a cloth falling on a ball 

ti.init(arch=ti.gpu)

# represent cloth as an nxn grid of mass points connected by sprigns
# System is affected by: 
#  - gravity
#  - internal forces spring
#  - damping
#  - collision

# Simulation advances time by dt, each timestep:
#  - estimates effects of these 4 factors on the mass spring system
#  - updates position and velocity accordingly

# Ball represented only by center and radius

# Cloth representation: (1 m^2 area)
n = 256
# x is an n x n field consisting of 3D vectors representing the mass points' positions
x = ti.Vector.field(3, dtype=float, shape=(n, n))
# v is an n x n field consisting of 3D vectors representing the mass points' velocities
v = ti.Vector.field(3, dtype=float, shape=(n, n))


# distance between each mass point on the grid is 1 / n (x axis and z axis)
x_z_dist = 1. / n
y_dist = 0.6
# Elastic coefficient of springs
spring_Y = 3e4
# Damping coefficient caused by
# the relative movement of the two mass points
# The assumption here is:
# A mass point can have at most 12 'influential' points
dashpot_damping = 1e4
gravity = ti.Vector([0, -9.81, 0])
drag_damping = 1
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

# Define the ball:
ball_radius = 0.3
# Use a 1D field for storing the position of the ball center
# The only element in the field is a 3-dimentional floating-point vector
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
# Place the ball center at the original point
ball_center[0] = [0, 0, 0]



# kernel decorator, automatically parallelizes all loops in the function
@ti.kernel
def init_mass_pts():
    # apply random offset to each mass point in the x and z axes
    rand_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    
    # x is n x n, so this loop is parallelized over every element in the field? 
    for i, j in x:
        # Every mass point is going to be y_dist above the ball (cloth is that amt higher than the ball but on the 
        # y - axis)
        x[i, j] = [
            i * x_z_dist - 0.5 + rand_offset[0],
            y_dist,
            j * x_z_dist - 0.5 + rand_offset[1]
        ]

        # initial velocity of each mass point is 0
        v[i, j] = [0, 0, 0]





# The cloth is modeled as a mass-spring grid. Assume that:
# a mass point, whose relative index is [0, 0],
# can be affected by at most 12 surrounding points
#
# spring_offsets is such a list storing
# the relative indices of these 'influential' points
spring_offsets = []
for i in range(-1, 2):
    for j in range(-1, 2):
        if (i, j) != (0, 0):
            spring_offsets.append(ti.Vector([i, j]))

# Works out the effects of the 4 forces/factors of the system for every timestep
# Assume that the internal forces of springs:
#  - each mass point influenced by at most 12 neighboring points
#  - internal forces of the neighbors exerted through the springs
@ti.kernel
def step():
    # First apply gravity to every point
    # The for loop iterates over all elements in x as if it's a 1D array, (groups iterators inside i)
    for i in ti.grouped(x):
        # Apply gravity
        v[i] += gravity * dt
        
    
    # Now apply internal spring forces
    for i in ti.grouped(x):
        # Start with 0 total force and accumulate
        force = ti.Vector([0.0, 0.0, 0.0])

        # At compile time, ti.static unrolls the loop 
        for offset in ti.static(spring_offsets):
            # remember that the offsets are relative the the center being [0, 0]
            # Here we treat i as the origin, j is the 'absolute' index of influential point
            j = i + offset

            # ensure that j is within bounds so that we can compare
            if 0 <= j[0] < n and 0 <= j[1] < n:
                # Find relative displacement and velocity of the two points
                x_ij, v_ij = x[i] - x[j], v[i] - v[j]

                # d is normalized vector of displacement
                d = x_ij.normalized()
                curr_dist = x_ij.norm()
                orig_dist = x_z_dist * float(i - j).norm()

                # Internal force of the spring:
                force += -spring_Y * d * (curr_dist / orig_dist - 1)

                # Damping force of the spring
                force += -v_ij.dot(d) * d * dashpot_damping * x_z_dist
        
        v[i] += force * dt

    # Lastly we apply the damping from oscillation of the springs and collisions
    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        # compute distance to center
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            # velocity projection
            normal = offset_to_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal
        # After working out the accumulative v[i],
        # work out the positions of each mass point
        x[i] += dt * v[i]

# Taichi has a GPU based GUI -> GGUI for 3D rendering
# Renders 2 types of objects, triangle meshes and particles, render the 
# cloth as triangle mesh and ball as particle

# represent triangle mesh as two fields, vertices and indices
# Vertices is a 1D field where each elt is a 3D vector representing position of a vertex
# Every point mass is a triangle vertex so can copy from x to vertices

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_spr = False

# Call this every frame to copy position points into vertices
@ti.kernel
def update_vertices():
    for i,j in ti.ndrange(n,n):
        vertices[i*n + j] = x[i,j]


# n x n grid of points, can think of (n-1) * (n-1) grid of squares
# each square represented by 2 trianges so (n-1) * (n-1) * 2 triangles
# This structure is captured in this function
@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # First triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # Second triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)


# Now we actually start the simulation and rendering loop
initialize_mesh_indices()
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
init_mass_pts()

while window.running:
    if current_t > 2.0:
        # Reset
        init_mass_pts()
        current_t = 0

    for i in range(substeps):
        step()
        current_t += dt
    update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()