import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

import mujoco
import numpy as np

# Set numpy print options
np.set_printoptions(precision=3, suppress=False)

model = mujoco.MjModel.from_xml_path('3bar_cfg.xml')
data = mujoco.MjData(model)

duration = 3e-3
dt = 1e-3
steps = int(duration / dt)

def get_sensor_info(sensor_name: str):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    start_idx = model.sensor_adr[id]
    dim = model.sensor_dim[id]

    return data.sensordata[start_idx: start_idx + dim]


print(f"[Taichi] version 1.7.2, llvm 15.0.7, commit 0131dce9, osx, python 3.11.6\n[Taichi] Starting on arch=arm64")
cnt = 1
for i in range(steps):
    mujoco.mj_step(model, data)

    print(f'\ntimestep: {cnt}')

    # Get sensor data for body r23
    linacc_r23 = np.copy(get_sensor_info("linacc_r23"))
    angacc_r23 = np.copy(get_sensor_info("angacc_r23"))

    # Update the sensor info for velocity and such
    mujoco.mj_forward(model, data)
    pos_r23 = get_sensor_info("pos_r23")
    linvel_r23 = get_sensor_info("linvel_r23")
    angvel_r23 = get_sensor_info("angvel_r23")
    quat_r23 = get_sensor_info("quat_r23")

    # Get sensor data for body r45
    linacc_r45 = get_sensor_info("linacc_r45")
    angacc_r45 = get_sensor_info("angacc_r45")
    pos_r45 = get_sensor_info("pos_r45")
    linvel_r45 = get_sensor_info("linvel_r45")
    angvel_r45 = get_sensor_info("angvel_r45")
    quat_r45 = get_sensor_info("quat_r45")

    # Print quantities for r23
    print(f'Body {1}:\nLinear Acceleration: {linacc_r23}\nAngular Acceleration: {angacc_r23}\nPosition: {pos_r23}\nOrientation: {quat_r23}\nLinear Velocity: {linvel_r23}\nAngular Velocity: {angvel_r23}')
    print(f'Body {2}:\nLinear Acceleration: {linacc_r45}\nAngular Acceleration: {angacc_r45}\nPosition: {pos_r45}\nOrientation: {quat_r45}\nLinear Velocity: {linvel_r45}\nAngular Velocity: {angvel_r45}')

    cnt+=1
