import warp as wp
import numpy as np

##############################################
# TESTING AND LEARNING ABOUT KERNELS IN WARP
##############################################

# In warp and GPU framework functions designed to be run in parallel so we use a "kernel" paradigm

# Here is an example of this in action

# use decorator over a python fn to designate a kernel
# Because kernel is mapped down to native C++/CUDA code you need to tell it the datatypes
@wp.kernel
def wp_dots(a: wp.array(dtype=wp.vec4),
             b: wp.array(dtype=wp.vec4),
             c: wp.array(dtype=wp.float32)):
    # So basically for this kernel a and b are 2D arrays, each element is 4d vector
    # c is a 1D array of floats where we store the results of our computation

    # get thread index b/c remember kernel is being run on a bunch of threads in parallel
    tid = wp.tid()

    # load the data based on which thread it is
    x,y = a[tid], b[tid]

    # compute result (this is differentiable if need be)
    r = wp.dot(x,y)
    # Write result to memory instead of returning (GPU/kernel paradigm is to not return but instead write directly to mem)
    c[tid] = r

# Going to do the same thing as previous kernel but in numpy for comparison
def np_dots(a,b,c=[]):
    for i in range(len(a)):
        res = np.dot(a[i], b[i])
        c.append(res)
    return np.array(c)

# We are going to compare the computations from warp to numpy to ensure everything went smoothly
n=1024

d1,d2 = np.random.rand(n, 4), np.random.rand(n, 4)
a = wp.from_numpy(d1, dtype=wp.vec4)
b = wp.from_numpy(d2, dtype=wp.vec4)
c = wp.empty(shape=n, dtype=wp.float32)

# Now we can run the kernel using wp.launch where we specify which kernel we want to run and how many threads we want to use etc
wp.launch(kernel=wp_dots,
         dim=n,
         inputs = [a,b,c])

c2 = np_dots(d1, d2)

print(np.allclose(c, c2)) # So as we can see the computation is the same



