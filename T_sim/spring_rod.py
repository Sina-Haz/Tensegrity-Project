import taichi as ti

ti.init(arch=ti.cpu)

# Want to model a system with a rod and 2 springs. Springs attached to fixed points + rod endpoints
sim_substeps = 10.
fps = 60.
# Define consts for springs:
ke = 5 # stiffness
kd = 0.2 # damping
L = 1.0 # rest len
frame_dt = 1. / fps
sim_dt = frame_dt / sim_substeps

# define variables for rod
rCoM = ti.Vector([0]*3)


