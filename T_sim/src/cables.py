import taichi as ti
import taichi.math as tm
from data import default_dtype, vec3
from quat import *

@ti.dataclass
class Spring:
    ke: default_dtype # Spring stiffness
    kd: default_dtype # Spring damping
    L0: default_dtype # Rest length
    x1: vec3 # Endpoint 1
    x2: vec3 # Endpoint 2

    @ti.func
    def len(self) -> default_dtype:
        '''
        Computes length of a cable based on its endpoint position
        '''
        return (self.x2-self.x1).norm()
    
    @ti.func
    def force(self, v1:vec3, v2:vec3) -> vec3:
        '''
        Takes in velocities of the endpoints of Spring spr.
        Computes the spring force using equation F = ke * (currLen - restLen) - kd * relative velocity
        Force is relative to (endpt2 - endpt1), unit vector
        returns a vec3 force vector 
        '''
        x_rel, v_rel = (self.x2 - self.x1), v2 - v1
        unit = tm.normalize(x_rel)

        # Total force is the sum of spring force and kd force
        # Hooke's Law: F_spr = -k*(L - L0) where L0 is rest length
        fs = - self.ke * (self.len() - self.L0)

        # spring damping law
        fd = - self.kd * tm.dot(v_rel, unit)

        f_tot = (fs + fd) * unit
        return f_tot


@ti.dataclass
class Cable:
    ke: default_dtype # Spring stiffness
    kd: default_dtype # Spring damping
    L0: default_dtype # Rest length
    x1: vec3 # Endpoint 1
    x2: vec3 # Endpoint 2

    @ti.func
    def len(self) -> default_dtype:
        '''
        Computes length of a cable based on its endpoint position
        '''
        return (self.x2-self.x1).norm()
    
    @ti.func
    def force(self, v1:vec3, v2:vec3) -> vec3:
        '''
        Computes the spring force using equation F = ke * (currLen - restLen) - kd * relative velocity
        Only applies force if the cable is stretched, no pushing force for when it's compressed
        Force is relative to (endpt2 - endpt1), unit vector
        returns a vec3 force vector 
        '''
        x_rel, v_rel = (self.x2 - self.x1), v2 - v1
        unit = tm.normalize(x_rel)
        ln = self.len()

        f_tot = vec3(0.0, 0.0, 0.0)
        # Only applies pulling force, not pushing force
        if ln > self.L0:
            # Total force is the sum of spring force and kd force
            # Hooke's Law: F_spr = -k*(L - L0) where L0 is rest length
            fs = - self.ke * (ln - self.L0)

            # spring damping law
            fd = - self.kd * tm.dot(v_rel, unit)

            f_tot += (fs + fd) * unit
        
        return f_tot



@ti.kernel
def test():
    spr = Spring(
        ke=10.0,      # example stiffness
        kd=1.0,       # example damping coefficient
        L0=1.0,       # rest length
        x1=vec3(0.0, 0.0, 0.0),  # endpoint 1
        x2=vec3(1.0, 0.0, 0.0)   # endpoint 2 (1 unit away from x1)
    )

        # Example velocities for each endpoint
    v1 = vec3(0.0, 0.0, 0.0)
    v2 = vec3(0.5, 0.0, 0.0)

    # Compute the current length of the spring
    current_length = spr.len()
    print("Current length:", current_length)

    # Compute the force exerted by the spring
    force = spr.force(v1, v2)
    print("Spring force:", force)

    cable = Cable(
        ke=10.0,      # example stiffness
        kd=1.0,       # example damping coefficient
        L0=1.0,       # rest length
        x1=vec3(0.0, 0.0, 0.0),  # endpoint 1
        x2=vec3(1.0, 0.7, 0.0)   # endpoint 2 (1 unit away from x1)
    )

        # Example velocities for each endpoint
    v1 = vec3(0.0, 0.0, 0.0)
    v2 = vec3(0.5, 0.0, 0.0)

    # Compute the current length of the spring
    current_length = cable.len()
    print("Cable length:", current_length)

    # Compute the force exerted by the spring
    force = cable.force(v1, v2)
    print("Cable force:", force)


if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    test()