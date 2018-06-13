#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>

#include <Eigen/Dense>
using namespace Eigen;
// -------------
// pure C++ code
// -------------
MatrixXf multiply_matrix(const MatrixXf &mat1, const MatrixXf &mat2)
{

    MatrixXf output(1,256);
    output = mat1*mat2;
    std::cout << "Here is output matrix:\n" << output << std::endl; 
    return output;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::array_t<float> py_multiply_matrix(py::array_t<float, py::array::f_style | py::array::forcecast> array, 
                                      py::array_t<float, py::array::f_style | py::array::forcecast> array1)
{
    
    // allocate std::vector (to pass to the C++ function)
    MatrixXf mat1(1,16), mat2(16,256), outmat(1,256);

    // copy py::array -> std::vector
    
    std::memcpy(mat2.data(), array1.data(), array1.size()*sizeof(float));
    std::memcpy(mat1.data(), array.data() , array.size()*sizeof(float));
    
    outmat = multiply_matrix(mat1, mat2);
    
    
    
    // allocate py::array (to pass the result of the C++ function to Python)
    auto result        = py::array_t<float>(outmat.size());
    auto result_buffer = result.request();
    float *result_ptr    = (float *) result_buffer.ptr;

    // copy std::vector -> py::array
    std::memcpy(result_ptr, outmat.data() ,outmat.size()*sizeof(float));


    
    return result;
}


// wrap as Python module
PYBIND11_MODULE(matrix_dot,m)
{
    m.doc() = "pybind11 example plugin";

    m.def("layer7_core_tmp1_mat2_dot", &py_multiply_matrix, "multiply python array's element ");
}