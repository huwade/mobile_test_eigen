#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <math.h>  
using namespace Eigen;
using Eigen::Tensor;



MatrixXf create_zero_array(int &tensor_channel_len)
{
    MatrixXf vec1(tensor_channel_len, 1);
    vec1.setZero();
    //std::cout << "vec1.setOnes();:\n" << vec1 << std::endl; 
    return vec1;
}



int multiply_elements(const std::vector<float>& input)
{
  int output = 1;
  
  for ( int i = 0 ; i < input.size() ; ++i )
  {
    output = output*static_cast<float>(input[i]);
  }

  return output;
}

namespace py = pybind11;

PYBIND11_MODULE(layer6_create,m)
{
    m.doc() = "pybind11 example plugin";

    m.def("create_zero_array", &create_zero_array, "multiply python array's element ");
    m.def("multiply_elements", &multiply_elements, "multiply python array's element ");
   
}