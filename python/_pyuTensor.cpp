#include <iostream>

#include "cstddef"
#include "play_fastop.hpp"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void talk(void) {
  std::cout << "Hello plugin with pybind11!" << std::endl;
  return;
}

PYBIND11_MODULE(_pyuTensor, m) {
  m.doc() = "pybind11 uTensor plugin";  // optional module docstring
  m.def("talk", &talk, "A function returning hello");
}
