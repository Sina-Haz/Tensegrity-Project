#include "lib.hpp" // Correct include path with quotes
#include <iostream>

// Function to print a dv3 vector
void print(const dv3& v) {
    std::cout << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")" << std::endl;
}

// commented out for now until we are done with implementation of lib functions and our header file is stable
//will be using main from lib for now
// int main() {
//     // code
//     return 0;
// }
