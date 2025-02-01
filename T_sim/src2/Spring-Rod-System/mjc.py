import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

import mujoco
import time
import mujoco_viewer as mjv
import numpy as np


# Create the MJCF XML model
XML = '''
<?xml version="1.0" ?>
<mujoco>
    <compiler angle="degree" coordinate="global" inertiafromgeom="true">
            <lengthrange timestep="0.001"/>
    </compiler>
    <option gravity="0 0 -9.81" integrator="Euler">
    </option>
    
    <worldbody>
        <!-- Light sources -->
        <light pos="0 0 10" diffuse="1 1 1" specular="1 1 1" />
        <light pos="5 5 10" diffuse="1 1 1" specular="1 1 1" />
        <light pos="-5 -5 10" diffuse="1 1 1" specular="1 1 1" />

        <!-- Fixed points for springs -->
        <site name="anchor1" pos="-1 0 6" size="0.02"/>
        <site name="anchor2" pos="1 0 6" size="0.02"/>
        
        <!-- Rod -->
        <body name="rod">
            <joint type="free"/>
            <geom type="cylinder" fromto="-1 0 4 1 0 4" size="0.5" mass="1"/>
            <!-- Spring attachment sites -->
            <site name="r1" pos="-1 0 4" size="0.02"/>
            <site name="r2" pos="1 0 4" size="0.02"/>
        </body>
    </worldbody>
    
    <tendon>
        <spatial name="spring1" stiffness="50" damping="20" springlength="2">
            <site site="anchor1"/>
            <site site="r1"/>
        </spatial>
        <spatial name="spring2" stiffness="10" damping="10" springlength="2">
            <site site="anchor2"/>
            <site site="r2"/>
        </spatial>
    </tendon>
</mujoco>
'''

# Load the model and create simulation
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

np.set_printoptions(precision=4, suppress=False)

data.qvel[0:3] = [0, 0, 0]  # linear velocity
data.qvel[3:6] = [0, 0, 0]  # angular velocity

viewer = mjv.MujocoViewer(model, data)

viewer.cam.lookat[0] = 0  # x-coordinate of the point to look at
viewer.cam.lookat[1] = 0  # y-coordinate of the point to look at
viewer.cam.lookat[2] = 4  # z-coordinate of the point to look at
viewer.cam.distance = 10   # Distance from the camera to the look-at point

# Simulation parameters
duration = 1
model.opt.timestep = dt = 1e-3
steps = int(duration / dt)


start_sim1 = time.time()
# Run simulation
for step in range(steps):
    # Step the simulation
    mujoco.mj_step(model, data)
    # viewer.render()
    
    # Extract position (already in xyz)
    position = data.qpos[0:3]
    
    # Convert quaternion from mujoco (wxyz) to matching format (xyzw)
    quat = data.qpos[3:7]  # wxyz format
    
    # Extract velocities (already in correct format)
    linear_vel = data.qvel[0:3]
    angular_vel = data.qvel[3:6]
    
    # Print in exact same format as Taichi simulation
    print(f'position: {position}\n '
          f'orientation: {quat}\n '
          f'linear velocity: {linear_vel}\n '
          f'angular velocity: {angular_vel}\n')
    
    
end_sim1 = time.time()
print(f'Mujoco simulation time: {end_sim1 - start_sim1}')