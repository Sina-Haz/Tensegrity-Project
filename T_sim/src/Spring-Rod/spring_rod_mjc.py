import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

import mujoco
import time
import mujoco_viewer as mjv


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
            <site name="rod_attach1" pos="-1 0 4" size="0.02"/>
            <site name="rod_attach2" pos="1 0 4" size="0.02"/>
        </body>
    </worldbody>
    
    <tendon>
        <spatial name="spring1" stiffness="20" damping="10" springlength="2">
            <site site="anchor1"/>
            <site site="rod_attach1"/>
        </spatial>
        <spatial name="spring2" stiffness="10" damping="20" springlength="2">
            <site site="anchor2"/>
            <site site="rod_attach2"/>
        </spatial>
    </tendon>
</mujoco>
'''

# Load the model and create simulation
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)



# Set initial conditions
# Position is already set in XML (0, 0, 4)
# Set initial velocity (0, 0, 0.3) and angular velocity (0, 0.2, 0)
# data.qpos[:] = [0,0,4,0.70710678, 0. , 0.70710678, 0. ]
# data.qpos[:] = [0,0,4,1,0,0,0]

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
    print(f'position: [{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}], '
          f'orientation: [{quat[0]:.3f} {quat[1]:.3f} {quat[2]:.3f} {quat[3]:.3f}], '
          f'linear velocity: [{linear_vel[0]:.3f} {linear_vel[1]:.3f} {linear_vel[2]:.3f}], '
          f'angular velocity: [{angular_vel[0]:.3f} {angular_vel[1]:.3f} {angular_vel[2]:.3f}]')
    
end_sim1 = time.time()
print(f'Mujoco simulation time: {end_sim1 - start_sim1}')
