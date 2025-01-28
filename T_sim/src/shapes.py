import taichi as ti
from rigid import *
from data import zero3
from inertia_tensor import *

"""
Define classes that will wrap rigid with specific basic shapes. For now let's just implement cylinder and sphere

Right now we are using them to parse, in future maybe could use these for contacts too?
"""



@ti.dataclass
class Cylinder:
    body: RigidBody
    length: default_dtype
    radius: default_dtype

    @ti.pyfunc
    def get_endpoints(self):
        '''
        Get endpoints of a cylinder in terms of absolute position
        '''
        # Axis vector = principal axis in local frame, scale this by half length to get 2 endpoints in local frame
        local_vec = vec3([0., 0., 1.]) * (1/2) * self.length
        local1, local2 = local_vec, -1 * local_vec

        e1, e2 = self.body.body_to_world(local1), self.body.body_to_world(local2)
        return e1, e2

@ti.dataclass
class Sphere:
    body: RigidBody
    radius: default_dtype


@ti.data_oriented
class Composite_Body:
    def __init__(self, body: RigidBody, components: list):
        self.body = body
        self.components = components
    




######## JSON PARSER FUNCTIONS

def cylinder_from_json(body_data: dict):
    # Assert statements
    assert 'endpoints' in body_data
    assert 'mass' in body_data
    assert 'radius' in body_data

    radius = body_data['radius']
    mass = body_data['mass']

    # Get state information
    if 'state' in body_data:
        v = body_data['state'].get('velocity', zero3)
        w = body_data['state'].get('omega', zero3)
    else:
        v = zero3
        w = zero3

    endpts = [vec3(pt) for pt in body_data['endpoints']]
    position = (endpts[0] + endpts[1]) / 2
    length = (endpts[1] - endpts[0]).norm()
    rot = quat_from_endpts(*endpts)

    # Get rigid body state
    rstate = rigid_state(pos = position, quat = rot, v = v, w = w)

    I_body = cylinder_inertia(mass,length, body_data['radius'])
    I_body_inv = I_body.inverse()
    body = RigidBody(state = rstate, mass = mass, I_body = I_body, I_body_inv = I_body_inv)

    return Cylinder(body, length, radius)


def sphere_from_json(body_data: dict):
    assert 'radius' in body_data
    assert 'mass' in body_data
    assert 'state' in body_data and 'pos' in body_data['state']

    radius = body_data['radius']
    mass = body_data['mass']

    v, w = body_data['state'].get('velocity', zero3), body_data['state'].get('omega', zero3)
    pos = body_data['state']['pos']
    rot = vec4([1, 0, 0, 0]) # Just give it identity rotation for now, can't see why this would matter for a sphere anyways

    rstate = rigid_state(pos=pos, quat=rot, v=v, w=w)

    I_body = solid_sphere_inertia(mass, radius)
    I_body_inv = I_body.inverse()

    body = RigidBody(state=rstate, mass=mass, I_body=I_body, I_body_inv=I_body_inv)
    return Sphere(body=body, radius=radius)


# a dictionary to map strings to non-composite rigid body types
shapes_no_composite = {'cylinder': cylinder_from_json, 'sphere': sphere_from_json}

def composite_from_json(data):
    assert 'bodies' in data
    shapes = []
    for b in data['bodies']:
        # For each body process it and add it to the list
        shape_fn = shapes_no_composite[b['type'].lower()]
        shapes.append(shape_fn(b))
    
    mass = sum(s.body.mass for s in shapes)
    com = sum(s.body.state.pos * (s.body.mass / mass) for s in shapes)
    
    # We are to assume the orientation of the cylinder for our use case
    cylinders = [obj for obj in shapes if isinstance(obj, Cylinder)]

    assert len(cylinders)<=1 # Should only be one cylinder
    rot = cylinders[0].body.state.quat

    # For now we are going to assume composite state is as such
    comp_state = rigid_state(pos = com, quat = rot, v = zero3, w = zero3)

    I_b = composite_inertia(com, shapes, rot)
    I_b_inv = I_b.inverse()

    comp_body = RigidBody(mass = mass, I_body = I_b, I_body_inv = I_b_inv, state=comp_state)

    return Composite_Body(body=comp_body, components=shapes)


