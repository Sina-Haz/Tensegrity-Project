import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

# define the sdf (signed distance function) for a sphere
def sphere_sdf(x, y, z, r):
    return np.sqrt(x**2 + y**2 + z**2) - r


def generate_sphere(resolution=100, radius=1.0):
    # We create a 3D grid of points around where the sphere will be
    x, y, z = np.meshgrid(np.linspace(-1.5, 1.5, resolution),
                          np.linspace(-1.5, 1.5, resolution),
                          np.linspace(-1.5, 1.5, resolution))
    
    # We feed this grid into the sdf and it computes the distance to nearest point on the sphere
    sdf_values = sphere_sdf(x, y, z, radius)
    
    vertices, faces, _, _ = measure.marching_cubes(sdf_values, level=0)
    
    return vertices, faces

# Generate the sphere
vertices, faces = generate_sphere()

# Visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces)
plt.show()