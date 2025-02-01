import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)


from serialize import *
from sim import *

rbs, sites, tendons, env = read_json('spring_rod.json')
np.set_printoptions(precision=4, suppress=False)

env.duration = 1
def print_bodies(rbs: Bodies):
    for i in range(rbs.n_bodies):
        if i not in env.fixed:
            print(f'Body: {i}')
            # Print in exact same format as Taichi simulation
            print(f'position: {rbs.P[i]}\n '
            f'orientation: {rbs.Q[i]}\n '
            f'linear velocity: {rbs.V[i]}\n '
            f'angular velocity: {rbs.W[i]}\n')


steps = int(env.duration / env.dt)

for _ in range(steps):
    euler_step(rbs, sites, tendons, env)

    print_bodies(rbs)