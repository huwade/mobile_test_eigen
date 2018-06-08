#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
using Eigen::Tensor;


Tensor<float, 2> test()
{
    Tensor<float, 3> tensor1(2,3,2);
    tensor1.setRandom();

    Tensor<float, 2> tensor2(4,3);
    
    Tensor<float, 2>::Dimensions dim1(4,3);

    tensor2 = tensor1.reshape(dim1);

    //std::cout << tensor1 << "\n";
    //std::cout << tensor2 << "\n";
    return tensor2;
}


Tensor<float, 2> test_passed_by_value(Eigen::Tensor<float, 3> tensor1)
{
    
    Tensor<float, 2> tensor2(4,3);
    
    Tensor<float, 2>::Dimensions dim1(4,3);

    tensor2 = tensor1.reshape(dim1);

    //std::cout << tensor1 << "\n";
    return tensor2;

}










