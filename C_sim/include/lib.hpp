#ifndef RIGID_BODY_SIMULATOR_H
#define RIGID_BODY_SIMULATOR_H

#include <iostream>
#include <vector>
#include <armadillo>
#include <functional>

#define NBODIES 1
#define STATE_VARIABLES 18
#define deltaT 1./24.

// Type alias for 3D vector, matrix, and quaternion
using v3 = arma::vec3;
using mat = arma::mat;
using quat = arma::vec4;

using namespace std;

/* Rigid Body Structure Definition */
struct rigid_body {
    // Constants
    double mass;
    mat Ibody, Ibodyinv; // Pre-computed inertia tensor

    // State Variables
    v3 x;    // Center of Mass position
    quat q;  // Orientation quaternion
    v3 P, L; // Linear and angular momentum

    // Derived Quantities
    mat Iinv, R; // Inverted inertia tensor and rotation matrix
    v3 v, omega; // Linear and angular velocity

    // Computed Quantities
    v3 force, torque;
};

/* Global Array of Rigid Bodies */
extern rigid_body bodies[NBODIES];

/* Function Declarations */

// Converts a vector into a symmetric matrix (star notation)
mat star(v3& a);

// Overload for quaternion multiplication
quat operator*(const quat& q1, const quat& q2);

// Overload for quaternion and vector multiplication
quat operator*(const v3& v1, const quat& q2);

// Converts a quaternion to a rotation matrix
mat QuaternionToMatrix(const quat& q);

// Computes force F(t) and torque τ(t) for a rigid body at time t
void compute_f_t(double t, rigid_body* rb);

// Updates derived quantities based on the current state
void getAttrFromState(rigid_body* rb);

// Moves the state of the rigid body forward in time by deltaT
void stepBody(double t, rigid_body* rb);

// Runs the simulation over a given time, with a function to display the bodies
void RunSimulation(vector<rigid_body>& Bodies, function<void(const vector<rigid_body>&)> displayBodies, double total_time);

#endif // RIGID_BODY_SIMULATOR_H

