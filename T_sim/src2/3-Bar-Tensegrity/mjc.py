import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

import mujoco
import numpy as np
from icecream import ic

# Set numpy print options
np.set_printoptions(precision=4, suppress=False)

model = mujoco.MjModel.from_xml_path('3-bar.xml')
data = mujoco.MjData(model)

duration = 2e-3
dt = 1e-3
steps = int(duration / dt)

def get_sensor_info(sensor_name: str):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    start_idx = model.sensor_adr[id]
    dim = model.sensor_dim[id]

    return data.sensordata[start_idx: start_idx + dim]

def get_site_position(site_name: str):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[id]

def get_body_state(body_name: str):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # Get joint ID
    joint_id = model.body_jntadr[body_id]

    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
        qpos_start = model.jnt_qposadr[joint_id]
        qvel_start = model.jnt_dofadr[joint_id]

        pos = data.qpos[qpos_start:qpos_start+3]   # Position (x, y, z)
        quat = data.qpos[qpos_start+3:qpos_start+7]  # Orientation quaternion
        linvel = data.qvel[qvel_start:qvel_start+3]  # Linear velocity (world frame)
        angvel_local = data.qvel[qvel_start+3:qvel_start+6]  # Angular velocity (local frame)

        # Convert angular velocity to world frame using rotation matrix
        R = data.xmat[body_id].reshape(3, 3)  # 3x3 rotation matrix
        angvel_global = R @ angvel_local  # Transform local to global

        return pos, quat, linvel, angvel_global

    else:
        print(f"Body {body_name} does not have a freejoint.")
        return None



for i in range(steps):
    mujoco.mj_step(model, data)


    # Print site positions and tendon lengths
    ic(data.site_xpos)
    ic(data.ten_length)


    pos_r23, quat_r23, linvel_r23, angvel_r23 = get_body_state("r23")
    pos_r45, quat_r45, linvel_r45, angvel_r45 = get_body_state("r45")

    print(f'Body: 1')
    print(f'position: {pos_r23}\n'
          f'orientation: {quat_r23}\n'
          f'linear velocity: {linvel_r23}\n'
          f'angular velocity: {angvel_r23}\n')

    print(f'Body: 2')
    print(f'position: {pos_r45}\n'
          f'orientation: {quat_r45}\n'
          f'linear velocity: {linvel_r45}\n'
          f'angular velocity: {angvel_r45}\n')






### SENSOR BASED PRINTS

    # # Get sensor data for body r23
    # linacc_r23 = np.copy(get_sensor_info("linacc_r23"))
    # angacc_r23 = np.copy(get_sensor_info("angacc_r23"))

    # # Get sensor data for body r45
    # linacc_r45 = np.copy(get_sensor_info("linacc_r45"))
    # angacc_r45 = np.copy(get_sensor_info("angacc_r45"))

    # # Update the sensor info for velocity and such
    # mujoco.mj_forward(model, data)
    # pos_r23 = get_sensor_info("pos_r23")
    # linvel_r23 = get_sensor_info("linvel_r23")
    # angvel_r23 = get_sensor_info("angvel_r23")
    # quat_r23 = get_sensor_info("quat_r23")

    # pos_r45 = get_sensor_info("pos_r45")
    # linvel_r45 = get_sensor_info("linvel_r45")
    # angvel_r45 = get_sensor_info("angvel_r45")
    # quat_r45 = get_sensor_info("quat_r45")

    # # print(f'a1: {linacc_r23}, alpha1: {angacc_r23}')
    # # print(f'a2: {linacc_r45}, alpha2: {angacc_r45}')

    # print(f'Body: {1}')
    # print(f'position: {pos_r23}\n'
    # f'orientation: {quat_r23}\n'
    # f'linear velocity: {linvel_r23}\n'
    # f'angular velocity: {angvel_r23}\n')

    # print(f'Body: {2}')
    # print(f'position: {pos_r45}\n'
    # f'orientation: {quat_r45}\n'
    # f'linear velocity: {linvel_r45}\n'
    # f'angular velocity: {angvel_r45}\n')
