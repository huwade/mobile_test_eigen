#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>


using Eigen::Tensor;
using Eigen::MatrixXd;
using Eigen::VectorXf;

Eigen::VectorXf hello(Eigen::Tensor<float, 3> &tensor1)
{
    
    //std::cout << " *** " << tensor1 << "\n";
    
    Tensor<float, 3> slice2(1,1,16);
    Eigen::DSizes<ptrdiff_t, 3> indices2(0,0,0);
    Eigen::DSizes<ptrdiff_t, 3> sizes2(1,1,16);
    slice2 = tensor1.slice(indices2, sizes2);
    

    //std::cout << " ******** " << "\n";
    
    
    VectorXf v1(16);
    
    for (int i = 0; i < 16; ++i) 
    {
        v1(i) = slice2(0,0,i);    
    }
    //std::cout << " *** " << v1 << "\n";
    return v1;
    
}


namespace py = pybind11;

py::array_t<float> py_test_passed_by_reference(py::array_t<float, py::array::c_style | py::array::forcecast> array)
{
    // allocate std::vector (to pass to the C++ function)
    Tensor<float, 3> tensor1(1,1,16);

    // copy py::array -> std::vector
    std::memcpy(tensor1.data(), array.data(), array.size()*sizeof(float));

    // call pure C++ function
    VectorXf output_tesnor(16);
    output_tesnor = hello(tensor1);
    
    //std::cout << " ******** " << output_tesnor << "\n";
    // allocate py::array (to pass the result of the C++ function to Python)
    auto result        = py::array_t<float>(output_tesnor.size());
    auto result_buffer = result.request();
    float *result_ptr    = (float *) result_buffer.ptr;

    // copy std::vector -> py::array
    std::memcpy(result_ptr, output_tesnor.data(), output_tesnor.size()*sizeof(float));
    
    
    return result;
}



PYBIND11_MODULE(eigen_slice,m)
{
  m.doc() = "pybind11 example plugin";

  m.def("hello", &py_test_passed_by_reference);

}

/* below is example
int hello()
{
    Tensor<float, 3> tensor(2,3,2);
    tensor.setRandom();
    
    std::cout << tensor << "\n";
    std::cout << "                    " << "\n";
    
    Tensor<float, 3> slice2(2,3,1);
    Eigen::DSizes<ptrdiff_t, 3> indices2(0,0,0);
    Eigen::DSizes<ptrdiff_t, 3> sizes2(2,3,1);
    std::cout << "                    " << "\n";
    slice2 = tensor.slice(indices2, sizes2);
    
    for (int i = 0; i < 2; ++i) 
    {
        for (int k = 0; k < 3; ++k) 
        {
        std::cout << i << "," << k << " " << slice2(i,k,0) << "\n";    
        }
    }
    
    
    Tensor<float, 3> slice3(2,1,2);
    Eigen::DSizes<ptrdiff_t, 3> indices3(0,0,0);
    Eigen::DSizes<ptrdiff_t, 3> sizes3(2,1,2);
    std::cout << "                    " << "\n";
    slice3 = tensor.slice(indices3, sizes3);
    
    for (int i = 0; i < 2; ++i) 
    {
        for (int k = 0; k < 2; ++k) 
        {
        std::cout << i << "," << k << " " << slice3(i,0,k) << "\n";    
        }
    }
    
    
    std::cout << "        " <<"\n";    
    
    
    for (int i = 0; i < 2; ++i) 
    {
        for (int j = 0; j < 3; ++j) 
        {
            for(int k = 0; k < 2; ++k) {
                std::cout << i << "," << j << "," << k << "   " << tensor(i,j,k) << "\n";    
            }
        }
    }
    
}
*/

/*
int hello()
{

    Tensor<float, 5> tensor(5,3,5,7,11);
    tensor.setRandom();

    Tensor<float, 5> slice1(1,1,1,1,1);
    Eigen::DSizes<ptrdiff_t, 5> indices(1,2,3,4,5);
    Eigen::DSizes<ptrdiff_t, 5> sizes(1,1,1,1,1);
    slice1 = tensor.slice(indices, sizes);
    std::cout << "slice1(0,0,0,0,0)" << "\n";
    std::cout << slice1(0,0,0,0,0) << "\n";
    std::cout << "tensor(1,2,3,4,5)" << "\n";
    std::cout << tensor(1,2,3,4,5) << "\n";

    
    
    Tensor<float, 5> slice2(2,1,1,1,3);
    Eigen::DSizes<ptrdiff_t, 5> indices2(1,1,1,1,5);
    Eigen::DSizes<ptrdiff_t, 5> sizes2(2,1,1,1,3);
    slice2 = tensor.slice(indices2, sizes2);
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 3; ++k) 
        {
        std::cout << slice2(i,0,0,0,k) << "       "<< tensor(1+i,1,1,1,5+k) << "\n";    
        }
    }
    
    
   
    MatrixXd m1(2,3);
    for (int i = 0; i < 2; ++i) 
    {
        for (int k = 0; k < 3; ++k) 
        {
        m1(i,k) = slice2(i,0,0,0,k);    
        }
    }
    
    std::cout << m1 << "\n";
    
    Tensor<float, 2> tensor1(2,3);
    Tensor<float, 2>::Dimensions dim1(2,3);
    tensor1 = slice2.reshape(dim1);
    

    
    std::cout << tensor1 << "\n";

}
*/
