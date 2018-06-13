#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <Eigen/Dense>
using namespace Eigen;
// -------------
// pure C++ code
// -------------
MatrixXf multiply_matrix_mat0_mat1(const MatrixXf &mat1, const MatrixXf &mat2)
{
    MatrixXf output(1,16);
    output = mat1*mat2;
    //std::cout << "Here is output matrix:\n" << output << std::endl; 
    return output;
}


MatrixXf multiply_matrix_tmp1_mat2(const MatrixXf &mat1, const MatrixXf &mat2)
{
    MatrixXf output(1,256);
    output = mat1*mat2;
    //std::cout << "Here is output matrix:\n" << output << std::endl; 
    return output;
}

MatrixXf create_index(int &tensor_channel_len, int &j)
{
    MatrixXf vec1(tensor_channel_len, 1);
    vec1.setOnes();
    //std::cout << "vec1.setOnes();:\n" << vec1 << std::endl; 
    vec1 = vec1*j;
    return vec1;
}


VectorXf one_d_arrange(int &len)
{

    VectorXf vec1(len, 1);
    vec1.setLinSpaced(len,0,len-1);

    return vec1;
}


double det(const Eigen::MatrixXd &xs)
{
    return xs.determinant();
}


MatrixXi four_d_ravel_multi_index(const MatrixXi &ind1, const MatrixXi &ind2, const MatrixXi &ind3, const MatrixXi &ind4, 
                                int &tensor_row_len, int &tensor_column_len, int &tensor_depth_len, int &tensor_channel_len)
{
    MatrixXi index(tensor_channel_len, 1);
    for(int i = 0;i < tensor_channel_len; i++)
    {    
        index(i,0) = ind1(i,0) + ind2(i,0) * tensor_row_len + ind3(i,0) * tensor_row_len * tensor_column_len 
            + ind4(i,0) * tensor_row_len * tensor_column_len * tensor_depth_len;
    }
    
    return index;
}



namespace py = pybind11;

PYBIND11_MODULE(matrix_dot,m)
{
    m.doc() = "pybind11 example plugin";

    m.def("layer7_core_mat0_mat1_dot", &multiply_matrix_mat0_mat1, "multiply python array's element ");
    m.def("layer7_core_tmp1_mat2_dot", &multiply_matrix_tmp1_mat2, "multiply python array's element ");
    m.def("layer7_create_index", &create_index,                "multiply python array's element ");
    m.def("layer7_one_d_arrange", &one_d_arrange,              "multiply python array's element ");
    m.def("det", &det);
    m.def("layer7_four_d_ravel_multi_index", &four_d_ravel_multi_index,"multiply python array's element ");
    
}