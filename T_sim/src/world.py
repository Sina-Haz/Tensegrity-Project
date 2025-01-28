import taichi as ti
import taichi.math as tm
from data import *
from quat import *
from cables import Spring, Cable
from rigid import RigidBody, rigid_state
from inertia_tensor import *
from shapes import *
import json

ti.set_logging_level('error')
ti.init(arch=ti.cpu, default_fp=ti.f64)

shapes = {'cylinder': cylinder_from_json, 'composite': composite_from_json}

@ti.kernel
def init_field(field: ti.template(), val: ti.template()):
    for i in ti.grouped(field):
        field[i] = val

@ti.data_oriented
class World:
    def __init__(self, json_file) -> None:
        '''
        Initializes fields for rigid bodies, cables, springs, 
        '''
        with open(json_file, 'r') as json_file:
            self.data = json.load(json_file)

        self.n_bodies = len(self.data['bodies'])
        self.n_springs = len(self.data['springs'])
        self.max_sites = max(max([len(rb['sites']) for rb in self.data['bodies']]), 2)

        self.rbs = RigidBody.field(shape=self.n_bodies)
        self.springs = Spring.field(shape=self.n_springs) 

        # Spring specific mappings
        self.spring_site_inds = ti.Vector.field(2, dtype=ti.i32, shape=(self.n_springs, 2))
        self.spring_force_map = ti.Matrix.field(self.max_sites, self.n_springs, dtype=ti.i32, shape = (self.n_bodies, ))

        # Store sites in both local and global reference (local for each rigid body that it's a part of)
        self.locals = ti.Vector.field(3, dtype=default_dtype, shape=(self.n_bodies, self.max_sites))
        self.globals = ti.Vector.field(3, dtype=default_dtype, shape=(self.n_bodies, self.max_sites))

        self.parse_rbs()
        self.parse_springs()

        self.time = ti.field(dtype=ti.f32, shape=())
        self.time[None] = 0.0

        self.g = vec3(self.data['gravity']) if 'gravity' in self.data else vec3([0, 0, -9.81])

    
    def parse_rbs(self):
        for i, body_data in enumerate(self.data['bodies']):
            # Initialize the shape class based on JSON data provided and what shape it indicates the rigid body being
            shape_fn = shapes[body_data['type'].lower()]
            shape = shape_fn(body_data)
            self.rbs[i] = shape.body

            for j in range(len(body_data['sites'])):
                site = vec3(body_data['sites'][j])
                self.globals[i, j] = site 
                # Need to initialize locals in the simulate kernel where we define our integration scheme


    def parse_springs(self):
        # First initialize map to all zeros:
        init_field(self.spring_force_map, ti.Matrix([[0] * self.n_springs for _ in range(self.max_sites)]))

        # Now we load in the springs
        for i, spr_data in enumerate(self.data['springs']):
            # Load in x1:
            x1 = spr_data['x1']
            if 'fixed' in x1:
                x1 = vec3(x1['fixed']) # Don't need to apply a transformation to this point
                self.spring_site_inds[i, 0] = ti.Vector([-1, -1], dt=ti.int16) # Add to spring site inds
            else:
                idx = x1['site']
                x1 = self.globals[*idx] # Use globals as site location
                self.spring_site_inds[i, 0] = ti.Vector(idx, dt=ti.int16) # Update spring site inds
                # If there is an attachment point on a rigid body we need to update the spring force map
                body_idx, site_idx= idx
                self.spring_force_map[body_idx][site_idx, i] = -1
            
            x2 = spr_data['x2']
            if 'fixed' in x2:
                x2 = vec3(x2['fixed']) # Don't need to apply a transformation to this point
                self.spring_site_inds[i, 1] = ti.Vector([-1, -1], dt=ti.int16)

            else:
                idx = x2['site']
                x2 = self.globals[*idx]
                self.spring_site_inds[i, 1] = ti.Vector(idx, dt=ti.int16)
                # If there is an attachment point on a rigid body we need to update the spring force map
                body_idx, site_idx = idx
                self.spring_force_map[body_idx][site_idx, i] = 1


            spr = Spring(ke=spr_data['ke'], kd = spr_data['kd'], L0 = spr_data['L0'], x1=x1, x2=x2)
            self.springs[i] = spr


    ########## Helper methods

    @ti.pyfunc
    def update_globals(self):
        for i, j in ti.ndrange(self.locals.shape[0], self.locals.shape[1]):
            self.globals[i, j] = self.rbs[i].body_to_world(self.locals[i, j])
        

    @ti.pyfunc
    def update_locals(self):
        for i, j in ti.ndrange(self.locals.shape[0], self.locals.shape[1]):
            self.locals[i, j] = self.rbs[i].world_to_body(self.globals[i, j])

    @ti.pyfunc
    def compute_endpoint_velocities(self, spr_idx):
        # Compute v1:
        v1 = vec3(0,0,0)
        if self.spring_site_inds[spr_idx, 0][0] != -1:
            body_idx, _ = self.spring_site_inds[spr_idx, 0]
            rbstate = self.rbs[body_idx].state
            v1 = rbstate.v + tm.cross(rbstate.w, self.springs[spr_idx].x1 - rbstate.pos)
            
        # Compute v2:
        v2 = vec3(0,0,0)
        if self.spring_site_inds[spr_idx, 1][0] != -1:
            body_idx, _ = self.spring_site_inds[spr_idx, 1]
            rbstate = self.rbs[body_idx].state
            v2 = rbstate.v + tm.cross(rbstate.w, self.springs[spr_idx].x2 - rbstate.pos)

        return v1, v2
        
    @ti.pyfunc
    def update_spring_attachments(self):
        for spr_idx in ti.static(range(self.n_springs)):
            # Update x1
            x1_idx = self.spring_site_inds[spr_idx, 0]
            if x1_idx[0] != -1: # i.e. if it's not fixed
                self.springs[spr_idx].x1 = self.globals[*x1_idx]

            x2_idx = self.spring_site_inds[spr_idx, 1]
            if x2_idx[0] != -1: # i.e. if it's not fixed
                self.springs[spr_idx].x2 = self.globals[*x2_idx]

                

