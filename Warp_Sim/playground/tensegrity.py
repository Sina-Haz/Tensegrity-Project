import warp as wp
import warp.sim as wps
import numpy as np
import warp.sim.render as wpsr
from math import pi

# WONT WORK B/C WARP DOESNT SUPPORT things like cables yet

# Our goal in this document is to parse the core.urdf file into a tensegrity bot and just watch it fall onto the ground
# Basing this off the quadruped example
# All our class needs is self and stage path. The rest we will hard code for now
class Tensegrity:
    def __init__(self, stage_path = 'depictions/ten_ex.usd'):
        # Initialize a model builder that will add things to model of our world
        builder = wps.ModelBuilder()
        # Have this same builder take in arguments from urdf file and initialize them into the Model
        wps.parse_urdf('model_assets/RodAssembly.urdf',
                    builder,
                    xform=wp.transform([0.0, 0.7, 5], wp.quat_identity())) # For now we will not add all these other params as we are just trying to make something work
        
        # Next we set up some simulation parameters
        self.sim_time = 0.0
        fps = 100
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        builder.gravity = (0, 0, -9.81)

        # More code from example
        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()
        self.model.ground = True # Ensures there's a ground plane in the world for contact

        # Usually a good integrator for Robotics
        # self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
        self.integrator = wps.XPBDIntegrator()


        # We set up the renderer (not found in documentation, it says it should be SimRendererUsd or SimRendererOpenGL?)
        self.renderer = wpsr.SimRenderer(self.model, stage_path)

        # Warp state object holds all time-varying data for a model such as positions, velocities, forces, etc
        # NOTE: for body_qd and body_f, Warp uses a spatial vector where 1st 3 are angular last 3 are linear
        # We get 2 states so we can use them as a buffer for smooth transitions, (i.e. state0=curr, state1=nxt) and etc.
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # We evaluate forward kinematics based on our model of the world, joint positions and velocities, and update our
        # state information based on this. The None is for the mask which says which joints do you want to fix/disable
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        # Some optimization code for if this simulation is created with CUDA enabled GPUs
        # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None


    # will look into all this later
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

if __name__ == '__main__':
        with wp.ScopedDevice('cpu'):
            ten = Tensegrity()

            for _ in range(100):
                ten.step()
                ten.render()

            if ten.renderer:
                ten.renderer.save()



