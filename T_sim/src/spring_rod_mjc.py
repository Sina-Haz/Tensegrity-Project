import mujoco
import time

# Calculate the same inertia values as in Taichi simulation
mass = 1.0
rod_len = 2.0
I_x = I_y = (1/12) * mass * rod_len**2
I_z = 0.1  # negligible for thin rod

# Create the MJCF XML model
XML = '''
<?xml version="1.0" ?>
<mujoco>
    <option gravity="0 0 -9.81" integrator="Euler">
    </option>
    
    <worldbody>
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
        <spatial name="spring2" stiffness="10" damping="10" springlength="2">
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
data.qpos[0:3] = [0,0,4]
data.qvel[0:3] = [0, 0, 0]  # linear velocity
data.qvel[3:6] = [0, 0, 0]  # angular velocity


# Simulation parameters
duration = 0.5
model.opt.timestep = dt = 1e-3
steps = int(duration / dt)


start_sim1 = time.time()
# Run simulation
for step in range(steps):
    # Step the simulation
    mujoco.mj_step(model, data)
    
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
