#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "test.h"
using Eigen::Tensor;

int hello()
{
    Tensor<float, 2> tensor3(4,3);
    tensor3 = test();
    std::cout << "test " << "\n";
    std::cout << tensor3 << "\n";
    
    Tensor<float, 3> tensor4(2,3,2);
    tensor4.setRandom();
    
    Tensor<float, 2> tensor5(4,3);
    
    tensor5 = test_passed_by_value(tensor4);
    std::cout << "                    " << "\n";
    std::cout << "test_passed_by_value" << "\n";
    std::cout << tensor4                << "\n";
    std::cout << "                    " << "\n";
    std::cout << tensor5                << "\n";
}


Tensor<float, 2> test_passed_by_reference(Eigen::Tensor<float, 3> &tensor1)
{
    
    Tensor<float, 2> output_tesnor(4,3);
    
    Tensor<float, 2>::Dimensions dim1(4,3);

    output_tesnor = tensor1.reshape(dim1);

    //std::cout << tensor1 << "\n";
    return output_tesnor;

}

namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::array_t<float> py_test_passed_by_reference(py::array_t<float, py::array::c_style | py::array::forcecast> array)
{
  // allocate std::vector (to pass to the C++ function)
  Tensor<float, 3> tensor1(2,3,2);
    
  // copy py::array -> std::vector
  std::memcpy(tensor1.data(), array.data(), array.size()*sizeof(float));
    
  // call pure C++ function
  Tensor<float, 2> output_tesnor(4,3);
  output_tesnor = test_passed_by_reference(tensor1);

  // allocate py::array (to pass the result of the C++ function to Python)
  // copy std::vector -> py::array
 
  ssize_t              ndim    = 2;
  std::vector<ssize_t> shape   = { 4 , 3 };
  std::vector<ssize_t> strides = { sizeof(float)*3 , sizeof(float) };

  // return 2-D NumPy array
  return py::array(py::buffer_info(
    output_tesnor.data(),                           /* data as contiguous array  */
    sizeof(float),                          /* size of one scalar        */
    py::format_descriptor<float>::format(), /* data type                 */
    ndim,                                    /* number of dimensions      */
    shape,                                   /* shape of the matrix       */
    strides                                  /* strides for each axis     */
  ));


}



PYBIND11_MODULE(eigen_reshape,m)
{
  m.doc() = "pybind11 example plugin";

  //m.def("hello", &hello);
  m.def("test_passed_by_reference", &hello);

}
