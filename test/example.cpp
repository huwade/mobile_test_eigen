#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
//#include <pybind11/numpy.h>
//#include <stdio.h>

//#include <pybind11/eigen.h>
#include <Eigen/LU>

//#include <Eigen/Dense>
using namespace Eigen;
// -------------
// pure C++ code
// -------------


double det(const Eigen::MatrixXf &xs)
{
  return xs.determinant();
}

namespace py = pybind11;

PYBIND11_MODULE(example,m)
{
    m.doc() = "pybind11 example plugin";

   
    m.def("det", &det);
}
