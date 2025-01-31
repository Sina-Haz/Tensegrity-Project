import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)


from serialize import *
from sim import *

rbs, sites, tendons = read_json('spring_rod_cfg.json')

# global_pos, site_V = updateSites(rbs.P, rbs.V, rbs.W, rbs.Q, sites.body_id, sites.local_pos)

# print(np.array_equal(global_pos, sites.global_pos)) # should be true since body position and orientation didn't change
# print(site_V) # Should be all 0's

