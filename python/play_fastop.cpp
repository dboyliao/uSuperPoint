#include "play_fastop.hpp"

using uTensor::Tensor;

py::array_t<float> FastMatrixMultOperator::output_array() {
  Tensor &out_tensor = outputs[output].tensor();
  void *buffer;
  size_t s = get_writeable_block(out_tensor, buffer,
                                 (uint16_t)out_tensor->num_elems(), 0);
  TensorShape out_shape = out_tensor->get_shape();
  py::ssize_t elem_size = static_cast<py::ssize_t>(sizeof(float));
  py::buffer_info info =
      py::buffer_info(buffer, elem_size, py::format_descriptor<float>::format(),
                      3, {out_shape[0], out_shape[1], out_shape[2]},
                      {elem_size * out_shape[1] * out_shape[2],
                       elem_size * out_shape[2], elem_size});
  return py::array_t<float>(info);
}