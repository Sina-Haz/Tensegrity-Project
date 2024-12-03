import taichi as ti
import numpy as np

default_dtype = ti.f64

# Example for 3D vectors and 4D vectors (could be useful in other contexts)
vec3 = ti.types.vector(3, default_dtype)
vec4 = ti.types.vector(4, default_dtype)
mat33 = ti.types.matrix(3, 3, default_dtype)