# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Force
#
# Shows how to apply an external force (torque) to a rigid body causing
# it to roll.
#
###########################################################################

import warp as wp
import warp.sim as wps
import warp.sim.render


class Example:
    def __init__(self, stage_path="depictions/rigid_force.usd", use_opengl=False):
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = wps.ModelBuilder()

        # adding a rigid body to the model 
        # The transform function takes in a position and a quaternion rotation and applies both of them
        # In general the origin parameter expects an input dim of 7 
        # So here we basically define position and orientation
        b = builder.add_body(origin=wp.transform((0.0, 5.0, 0.0), wp.quat_identity()))
        # Adds the shape at the specified body (which we designated)
        # builder.add_shape_box(body=b, hx=1.0, hy=1.0, hz=1.0, density=100.0)
        builder.add_shape_sphere(body=b, radius=1, density=50.0)

        # Transfers all the data to warp data structures and vectors (shouldn't add more rigid bodies to the builder after this)
        self.model = builder.finalize()

        # Enables ground plane and contact
        self.model.ground = True

        # XPBD is a good integrator, very stable even with larger timesteps
        self.integrator = wps.XPBDIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if use_opengl:
            self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage_path, device=wp.get_device("cpu"))
        elif stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path)
        else:
            self.renderer = None

        # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            wps.collide(self.model, self.state_0)

            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.state_0.body_f.assign(
                [
                    [0.0, 0.0, -7000.0, 0.0, 0.0, 0.0],
                ]
            )

            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_rigid_force.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument(
        "--opengl",
        action="store_true",
        help="Open an interactive window to play back animations in real time. Ignores --num_frames if used.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, use_opengl=args.opengl)

        if args.opengl:
            while example.renderer.is_running():
                example.step()
                example.render()
        else:
            for _ in range(args.num_frames):
                example.step()
                example.render()

        if example.renderer:
            example.renderer.save()
