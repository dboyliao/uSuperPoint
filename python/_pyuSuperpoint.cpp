#include <iostream>

#include "cstddef"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "superpoint.hpp"
#include "uTensor.h"

namespace py = pybind11;

void talk(void) {
  Superpoint superpoint;
  std::cout << "Hello plugin with pybind11!" << std::endl;
  return;
}

class pySuperpoint {
 public:
  pySuperpoint() : superpoint(), meta_alloc(), ram_alloc() {
    Context::get_default_context()->set_metadata_allocator(&meta_alloc);
    Context::get_default_context()->set_ram_data_allocator(&ram_alloc);
  }
  ~pySuperpoint() = default;

  // take numpy array as input, size : 1 x rows x cols x 1
  // output numpy array, size : 1 x 15 x 11 x 256
  py::array_t<float> inference(py::array_t<float> input) {
    auto buf = input.request();
    float *ptr = (float *)buf.ptr;
    uint16_t rows = buf.shape[1], cols = buf.shape[2];
    Tensor t_img = new RamTensor({1, rows, cols, 1}, flt);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        t_img(r, c) = ptr[r * cols + c];
      }
    }

    // output size: 1 x 15 x 11 x 256
    Tensor out_encode = new RamTensor({1, 15, 11, 256}, flt);
    superpoint.set_inputs({{Superpoint::input_0, t_img}})
        .set_outputs({{Superpoint::output_0, out_encode}})
        .eval();

    // output numpy array
    py::array_t<float> output({1, 15, 11, 256});
    auto out_buf = output.request();
    float *out_ptr = (float *)out_buf.ptr;
    for (int r = 0; r < 15; ++r) {
      for (int c = 0; c < 11; ++c) {
        for (int d = 0; d < 256; ++d) {
          out_ptr[r * 11 * 256 + c * 256 + d] = out_encode(0, r, c, d);
        }
      }
    }

    return output;
  }

 private:
  Superpoint superpoint;
  localCircularArenaAllocator<1024 * 10> meta_alloc;
  localCircularArenaAllocator<1024 * 300, uint32_t> ram_alloc;
};

PYBIND11_MODULE(_pyuSuperpoint, m) {
  m.doc() = "pybind11 superpoint plugin";  // optional module docstring
  m.def("talk", &talk, "A function returning hello");
  // pySuperpoint class
  py::class_<pySuperpoint>(m, "Superpoint")
      .def(py::init<>())
      .def("inference", &pySuperpoint::inference, "Superpoint inference");
}
