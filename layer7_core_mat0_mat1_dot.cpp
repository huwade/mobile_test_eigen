#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <math.h>  
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



MatrixXi mod(const MatrixXi &ind1, int b)
{
    MatrixXi res;
    res = ind1.unaryExpr([&](const int x) { return x%b; });
    //std::cout << "res" << res << std::endl; 
    return res;
}


using std::floor; 

VectorXf floor_array(const MatrixXf &array)
{
    int len;
    len = array.size();
    VectorXf vec1(len, 1);
    
    
    vec1 = array.unaryExpr([&](const float x) { return floor(x); });
    return vec1;
    
}

MatrixXf multiply_matrix_multiply_element_wise(const MatrixXf &mat1, const MatrixXf &mat2)
{
    MatrixXf output;
    output = mat1.array () *mat2.array () ;
    //std::cout << "Here is output matrix:\n" << output << std::endl; 
    return output;
}


MatrixXf multiply_matrix(const MatrixXf &mat1, const MatrixXf &mat2)
{
    MatrixXf output;
    output = mat1*mat2;
    //std::cout << "Here is output matrix:\n" << output << std::endl; 
    return output;
}


MatrixXf Get_feature_x_by_column_index(const MatrixXf &feature_x, const MatrixXf &column_index)
{
    int len = column_index.size();
    MatrixXf output(1,len);
    int idx;
        
    for(int i = 0; i < len; i++)
    {
        idx = column_index(i,0);
        output(0,i) = feature_x(idx,0);
    }


    return output;
}



namespace py = pybind11;

PYBIND11_MODULE(matrix_dot,m)
{
    m.doc() = "pybind11 example plugin";

    //m.def("layer7_core_mat0_mat1_dot", &multiply_matrix_mat0_mat1, "multiply python array's element ");
    //m.def("layer7_core_tmp1_mat2_dot", &multiply_matrix_tmp1_mat2, "multiply python array's element ");
    m.def("dot_matrix", &multiply_matrix, "multiply python array's element ");
    m.def("layer7_create_index", &create_index,                "multiply python array's element ");
    m.def("layer7_one_d_arrange", &one_d_arrange,              "multiply python array's element ");
    m.def("det", &det);
    m.def("layer7_four_d_ravel_multi_index", &four_d_ravel_multi_index,"multiply python array's element ");
    m.def("layer7_mod", &mod,"multiply python array's element ");
    m.def("layer7_floor_array", &floor_array,"multiply python array's element ");
    m.def("layer7_multiply_matrix_multiply_element_wise", &multiply_matrix_multiply_element_wise,"multiply python array's element ");
    m.def("layer7_Get_feature_x_by_index", &Get_feature_x_by_column_index,"multiply python array's element ");
    
}