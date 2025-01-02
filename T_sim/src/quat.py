import taichi as ti
import taichi.math as tm
from data import default_dtype, vec3, vec4, mat33
import numpy as np


@ti.func
def quat_conjugate(q: vec4):
    conj = vec4(q * -1)
    conj[0] *= -1
    return conj

@ti.func
def quat_mul(q1, q2) -> vec4:
    s1, s2 = q1[0], q2[0]
    v1, v2 = q1[1:], q2[1:]
    return vec4(s1 * s2 - tm.dot(v1, v2), s1*v2 + s2*v1 + tm.cross(v1, v2))

# Returns conjugate but first ensures it's normalized
@ti.func
def quat_inv(q) -> vec4:
    return quat_conjugate(q.normalized())

@ti.func
def quat_to_matrix(q) -> mat33:
    w, x, y, z = q
    return ti.Matrix([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

@ti.func
def quat_exp(q) -> vec4:
    s, v = q[0], q[1:]

    norm_v = tm.clamp(v.norm(), 1e-8, float('inf'))
    e_q = vec4((tm.e**s) * tm.cos(norm_v), v*(tm.e**s) * tm.sin(norm_v) * (1 / norm_v))
    return e_q

def quat_from_endpts(p1: list, p2: list) -> vec4:
    '''
    Takes in two endpoints as lists and computes the quaternion rotation
     - Rotation is w.r.t. principal axis z = (0, 0, 1)
     - We do this with numpy under the hood and then convert to taichi's vec4

    Do this by computing rotation axis as cross product of z and normalized unit vector b/w endpoints
    '''
    p1, p2 = np.array(p1, dtype=np.float64), np.array(p2, np.float64)
    v = p2 - p1
    v_norm = np.linalg.norm(v)

    # Endpoints shouldn't be the same
    assert v_norm != 0

    # Normalize v
    v = v / v_norm
    z = vec3(0,0,1)

    # Rotation axis:
    r = np.cross(z, v)
    r_norm = np.linalg.norm(r)
    theta = np.arccos(np.dot(z, v))

    # Edge cases: (r is parallel to z, no rotation or flipped)
    if r_norm == 0:
        if r == z: return vec4(1,0,0,0) # Unit quaternion
        else: return vec4(0, 1, 0, 0)

    # Normalize r if r_norm neq to 0
    r = r / r_norm
    w = np.cos(theta / 2)
    xyz = r * np.sin(theta / 2)

    return vec4(w, xyz)




@ti.func
def quat_from_matrix(R) -> vec4:
    '''
    WARNING: Numerically unstable
    '''
    # Handle special cases to avoid numerical instability
    trace = R.trace()
    w,y,x,z = 0.,0.,0.,0.
    if trace > 0:
        S = ti.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = ti.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = ti.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = ti.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    
    # Normalize the quaternion
    q = ti.Vector([w, x, y, z])
    q = q.normalized()
    
    return q


# Test the quaternion operations in a kernel
@ti.kernel
def test_quaternion_operations():
    # Define a quaternion (w, x, y, z)
    q = vec4(1.0, 2.0, 3.0, 4.0)
    print("Original Quaternion:", q)

    # Conjugate of quaternion
    q_conj = quat_conjugate(q)
    print("Conjugate of Quaternion:", q_conj)

    # Normalize and get the inverse
    q_inv = quat_inv(q)
    print("Inverse of Quaternion:", q_inv)

    # Convert quaternion to rotation matrix
    R = quat_to_matrix(q.normalized())
    print("Rotation Matrix from Quaternion:", R)

    # Convert matrix back to quaternion
    q_reconstructed = quat_from_matrix(R)
    print("Reconstructed Quaternion from Matrix:", q_reconstructed)
    print(q.normalized(), q_reconstructed)

    # Test quaternion multiplication
    q1 = vec4(1.0, 0.0, 0.0, 0.0)
    q2 = vec4(0.0, 1.0, 0.0, 0.0)
    q_mul = quat_mul(q1, q2)
    print("Quaternion Multiplication Result:", q_mul)

# Run the test kernel
if __name__ == '__main__':
    ti.init(default_fp=default_dtype)

    test_quaternion_operations()
                 


    