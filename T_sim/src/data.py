import taichi as ti

default_dtype = ti.f64

# Example for 3D vectors and 4D vectors (could be useful in other contexts)
vec3 = ti.types.vector(3, default_dtype)
vec4 = ti.types.vector(4, default_dtype)
mat33 = ti.types.matrix(3, 3, default_dtype)

zero3 = vec3([0., 0., 0.])
zero4 = vec4([0., 0., 0., 0.])
eye3 = mat33([[1., 0., 0], [0, 1, 0], [0, 0, 1]])
id_quat = vec4([1., 0., 0., 0.])