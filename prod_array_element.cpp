#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>

// -------------
// pure C++ code
// -------------

float multiply(const std::vector<float>& input)
{
  float output = 1;
  
  for ( int i = 0 ; i < input.size() ; ++i )
  {
    output = output*static_cast<float>(input[i]);
  }

  return output;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;



// wrap as Python module
PYBIND11_MODULE(multiply_element,m)
{
  m.doc() = "pybind11 example plugin";

  m.def("multiply", &multiply, "multiply python array's element ");
}