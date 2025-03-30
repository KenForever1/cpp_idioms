#include "libE.h"
#include <iostream>

const std::string global_message = "Hello from library E!";

void printMessage() {
    std::cout << global_message << " (Address: " << &global_message << ")" << std::endl;
}