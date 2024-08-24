#include <iostream>
#include <vector>
#include <armadillo>
#include <functional>

# define NBODIES 1
# define STATE_VARIABLES 18
# define deltaT 1./24.

// using v3 as type alias for 3d vector
using v3 = arma::vec3;
using mat = arma::mat;
using quat = arma::vec4;


// this namespace declaration may lead to conflicts later but for now I am using it to simplify the code
using namespace std;


/* THIS SECTION IS ALL HELPER FUNCTIONS*/


// This function is to convert a vector into symmetric matrix as seen by star notation
mat star(v3& a){
    mat A = {{0, -a[2], a[1]},
             {a[2], 0, -a[0]},
             {-a[1], a[0], 0}};
    return A;
}

// This function is to overload the multiply operator for quaternion multiplication
quat operator*(const quat& q1, const quat& q2){
    double s1 = q1(0), s2 = q2(0);
    v3 v1 = q1.subvec(1,3), v2 = q2.subvec(1,3);
    quat res;
    res(0) = s1*s2 - arma::dot(v1, v2);
    res.subvec(1,3) = s1*v2 + s2*v1 + arma::cross(v1,v2);
    return res;
}

// overload to get proper quaternion multiplication b/w 3d vector and quaternion
quat operator*(const v3& v1, const quat& q2){
    quat q1;
    q1(0) = 0;
    q1.subvec(1,3) = v1;
    return q1*q2;
}

mat QuaternionToMatrix(const quat& q) {
    quat norm_q = arma::normalise(q);
    double w = norm_q(0), x = norm_q(1), y = norm_q(2), z = norm_q(3);

    return mat({
        {1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)},
        {2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)},
        {2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)}
    });
}


/*THIS SECTION IS MAINLY RIGID BODY STRUCT IMPLEMENTATION*/
// This implementation mainly pulled from https://graphics.pixar.com/pbm2001/pdf/notesg.pdf
struct rigid_body{
    //Constants
    double mass;
    mat Ibody, Ibodyinv; //pre-computed init inertia tensor

    //State Variables
    v3 x; //CoM position
    quat q; //unit quaternion q to represent orientation
    v3 P, L; //linear and angular momentum (use these bc momentum is conserved)

    //Derived Quantities
    mat Iinv, R; //Inverted inertia tensor and rotation matrix (3x3)
    v3 v, omega; //linear and angular velocity

    //Computed Quantities
    v3 force, torque;
};

// Global array of rigid bodies in the world
rigid_body bodies[NBODIES];

/*computes the force F(t) and torque τ(t) acting on the rigid body *rb at time t and stores in rb->force and rb->torque respectively.
ComputeForceAndTorque takes into account all forces and torques: gravity, wind, interaction with other bodies etc.*/
void compute_f_t(double t, rigid_body* rb);

// get derived quantities based off of state
void getAttrFromState(rigid_body* rb){
    // p = mv
    rb->v = rb->P / rb->mass; 

    rb->R = QuaternionToMatrix(arma::normalise(rb->q));

    // I^-1 = R I_body^-1 R^T
    rb->Iinv = rb->R * rb->Ibodyinv * arma::trans(rb->R);

    // Iw = L so w = I^-1 L
    rb->omega = rb->Iinv * rb->L;
}

// Move body state forward in time by deltaT amount
void stepBody(double t, rigid_body* rb){
    // Make sure all derived quantities are up to date based on current state
    getAttrFromState(rb);

    /* First we compute current force and torque on the body*/
    compute_f_t(t, rb);

    /*Next we get dX/dt (derivative of state at time t)*/
    v3 xdot = rb->v;
    quat qdot = .5 * rb->omega * rb->q; //Compute dq/dt = 1/2 w(t) q(t) 
    v3 Pdot = rb->force;
    v3 Ldot = rb->torque;

    /* Lastly we use some numerical method or solver to get X(t+deltaT) given dX/dt
    For now we are using explicit Euler, but look into boose.ODEint library for runge_kutta solvers*/
    rb->x += xdot*deltaT;
    rb->q += qdot*deltaT;
    rb->P += Pdot*deltaT;
    rb->L += Ldot*deltaT;
}

void RunSimulation(vector<rigid_body>& Bodies, function<void(const vector<rigid_body>&)> displayBodies, double total_time){
    for(double t = 0; t < total_time; t+= deltaT){
        // update each rigid body based on their prior state, computed force and torque, etc.
        for(auto body: Bodies){
            // in this function we compute derived quantities, forces and torque, and we step using dX/dt
            stepBody(t, &body); 
        }

        // use some inputted function to display the bodies (graphically, through printing state, etc. )
        displayBodies(Bodies);
    }


}

int main(){

}

/* 
Still needed:
 - Function to initialize rigid bodies (i.e. smth like initCube, initSphere, etc.)
 - Implementation of compute force and torque function (idk how we can do this, vector fields, list of external forces, etc?)
 - Collision detection module
 - Collision response module
 - improved numerical integration
 - graphics/visual output
*/