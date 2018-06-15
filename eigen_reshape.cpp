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




Tensor<float, 2> test_passed_by_reference(Eigen::Tensor<float, 3> &tensor1)
{
    
    Tensor<float, 2> output_tesnor(4,3);
    
    Tensor<float, 2>::Dimensions dim1(4,3);

    output_tesnor = tensor1.reshape(dim1);

    //std::cout << tensor1 << "\n";
    return output_tesnor;

}


void tensor_to_matrix_slice(Tensor<float, 3> &tensor1)
{
    
    std::cout << " *** " << tensor1 << "\n";
}


namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::array_t<float> py_test_passed_by_reference(py::array_t<float, py::array::c_style | py::array::forcecast> array)
{
  // allocate std::vector (to pass to the C++ function)
  Tensor<float, 3> tensor1(1,6,2);
    
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

// wrap C++ function with NumPy array IO
py::array_t<float> py_test_matrix_slice(py::array_t<float, py::array::c_style | py::array::forcecast> array)
{
    // allocate std::vector (to pass to the C++ function)
    Tensor<float, 3> tensor1(1,6,2);

    // copy py::array -> std::vector
    std::cout << " ******** " << "\n";
    std::memcpy(tensor1.data(), array.data(), array.size()*sizeof(float));
    
   
    //vec1 = tensor_to_matrix_slice(tensor1);
    tensor_to_matrix_slice(tensor1);



    // return  NumPy array
    //return vec1;


}

PYBIND11_MODULE(eigen_reshape,m)
{
  m.doc() = "pybind11 example plugin";

  //m.def("hello", &hello);
  m.def("test_passed_by_reference", &py_test_passed_by_reference);
  m.def("py_test_matrix_slice", &py_test_matrix_slice);

}
