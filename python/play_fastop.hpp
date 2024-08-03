#include <pybind11/numpy.h>

#include "uTensor.h"
#include "uTensor/core/operatorBase.hpp"
#include "uTensor/ops/Matrix.hpp"

using uTensor::FastOperator;
using uTensor::ReferenceOperators::MatrixMultOperatorV2;
namespace py = pybind11;

class FastMatrixMultOperator : public MatrixMultOperatorV2<float>,
                               FastOperator {
 public:
  py::array_t<float> output_array();
};