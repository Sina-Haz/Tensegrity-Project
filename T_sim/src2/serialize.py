import json
import numpy as np
from data import *
from quat import *
from inertia import *
from collections import defaultdict


"""
In this file we define functions that take in a JSON file and load in all of our simulation data into our 
SoA data structures defined in data.py

JSON file format is pretty much exactly the same as before, except now site's are defined by a single index when referenced
by a tendon, and each tendon has a "type" attribute - 0 if it's a spring, 1 if it's a cable
"""


def read_json(file_name):
    with open(file_name, 'r') as fp:
        data = json.load(fp)

    assert 'env' in data
    assert 'bodies' in data
    assert 'tendons' in data

    n_bodies, n_tendons = len(data['bodies']), len(data['tendons'])

    body_attrs, site_attrs = read_bodies_sites(data['bodies'])
    tendon_attrs = read_tendons(data['tendons'])
    env_attrs = data['env']

    n_sites = len(site_attrs['body_id'])

    # Make sure all the lists are changed into numpy arrays
    body_attrs = {k: np.array(v) for k, v in body_attrs.items()}
    site_attrs = {k: np.array(v) for k, v in site_attrs.items()}
    tendon_attrs = {k: np.array(v) for k, v in tendon_attrs.items()}

    # Initialize all data structures using parsed json data
    rbs = Bodies(n_bodies, 
                 body_attrs['pos'],
                 body_attrs['quat'],
                 body_attrs['V'],
                 body_attrs['W'],
                 body_attrs['mass'],
                 body_attrs['I_b'])
    
    sites = Sites(n_sites, 
                  site_attrs['body_id'],
                  site_attrs['global_pos'],
                  rbs.P, rbs.Q)
    
    tendons = Tendons(n_tendons,
                      tendon_attrs['stiffness'],
                      tendon_attrs['damping'],
                      tendon_attrs['rest_len'],
                      tendon_attrs['attach1_id'], 
                      tendon_attrs['attach2_id'],
                      tendon_attrs['type'])
    
    env = Env(
        n_bodies,
        n_sites,
        n_tendons,
        env_attrs['dt'],
        env_attrs['g'] if 'g' in env_attrs else np.array([0, 0, -9.81]),
        env_attrs['duration'] if 'duration' in env_attrs else 1,
        env_attrs['fixed'] if 'fixed' in env_attrs else None
    )
    
    return rbs, sites, tendons, env



def read_bodies_sites(bodies):
    # Function map to determine which function to use to parse each body dictionary
    fn_map = {'cylinder': cylinder_from_json, 'sphere': sphere_from_json}

    # Defines the lists which we will input to initialize the Bodies data structure:
    body_attrs = defaultdict(list)

    # Define the data needed for initializing the Sites data structure
    site_attrs = defaultdict(list)
    for i, body in enumerate(bodies):
        assert 'type' in body
        fn = fn_map[body['type'].lower()]
        attrs = fn(body)

        for key, val in attrs.items():
            #ensures we have the same keys as the read fns (which should have uniform key names), but here we collect into list
            body_attrs[key].append(val) 

        assert 'sites' in body
        for site in body['sites']:
            site_attrs['body_id'].append(i)
            site_attrs['global_pos'].append(site)
    
    return body_attrs, site_attrs


def read_tendons(tendons):
    res = defaultdict(list)

    for t in tendons:
        res['stiffness'].append(t['ke'])
        res['damping'].append(t['kd'])
        res['rest_len'].append(t['L0'])
        res['type'].append(t['type'])

        if 'fixed' in t['x1']:
            res['attach1_id'].append(-1)
        else:
            res['attach1_id'].append(t['x1']['site'])
        
        if 'fixed' in t['x2']:
            res['attach2_id'].append(-1)
        else:
            res['attach2_id'].append(t['x2']['site'])

    return res



def cylinder_from_json(body_data: dict):
    # Assert statements
    assert 'endpoints' in body_data
    assert 'mass' in body_data
    assert 'radius' in body_data

    radius, mass = body_data['radius'], body_data['mass']

    # Get state information
    if 'state' in body_data:
        v = body_data['state'].get('velocity', np.zeros(3))
        w = body_data['state'].get('omega', np.zeros(3))
    else:
        v, w = np.zeros(3), np.zeros(3)
    
    endpts = [np.array(pt) for pt in body_data['endpoints']]
    position = (endpts[0] + endpts[1]) / 2
    length = np.linalg.norm(endpts[1] - endpts[0])
    rot = quat_from_endpts(*endpts)

    I_body = cylinder_inertia(mass,length,radius)

    return {
        "pos": position,
        "quat": rot,
        "V": v,
        "W": w,
        "mass": mass,
        "I_b": I_body
    } # May eventually need to return a second dict or add additional dict items for contact data


def sphere_from_json(body_data: dict):
    assert 'radius' in body_data
    assert 'mass' in body_data
    assert 'state' in body_data and 'pos' in body_data['state']

    radius = body_data['radius']
    mass = body_data['mass']

    v, w = body_data['state'].get('velocity', np.zeros(3)), body_data['state'].get('omega', np.zeros(3))
    pos = body_data['state']['pos']
    rot = np.array([1, 0, 0, 0]) # Just give it identity rotation for now, can't see why this would matter for a sphere anyways

    I_body = solid_sphere_inertia(mass, radius)

    return {
        "pos": pos,
        "quat": rot,
        "V": v,
        "W": w,
        "mass": mass,
        "I_b": I_body
    }