# calcul  y = Wx, W is fully connect layer, W = G_1*G_2*...G_d

from __future__ import print_function
from mpi4py import MPI
from inspect import currentframe, getframeinfo



import keras
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import itertools
import time
from numba import jit
import cupy as cp
#import user define c++ function
import multiply_element
import eigen_slice
import matrix_dot


@jit
def tt_construct_layer7(filename, feature_x):

    mat_dict = {}
    mat_dict.update(scipy.io.loadmat(filename))

    a = mat_dict['TT_mat']

    dim_arr  = a['d'][0][0][0,0]
    #n_arr    = a['n'][0][0][0:,0]
    #ps_arr   = a['ps'][0][0][0:,0]
    #core_arr = a['core'][0][0][0:,0]
    ranks    = a['r'][0][0][0:,0]
    bias     = a['bias'][0][0]
    
    right_modes = a['right_modes'][0][0][0,0:]
    right_modes = np.array(right_modes, dtype=np.int32)

    left_modes  = a['left_modes'][0][0][0,0:]
    left_modes  = np.array(left_modes, dtype=np.int32)


    L_cpp      = multiply_element.multiply(left_modes)
    L          = np.prod(left_modes)

    R_cpp      = multiply_element.multiply(right_modes)
    R          = np.prod(right_modes)

    print(L_cpp-L)
    print(R_cpp-R)

    core_tensor_0 = a['tensor'][0][0][0,0]
    core_tensor_1 = a['tensor'][0][0][0,1]
    core_tensor_2 = a['tensor'][0][0][0,2]
    core_tensor_3 = a['tensor'][0][0][0,3]



    column_len   = L
    row_len      = R

    if dim_arr > 2:
        shape = left_modes*right_modes.ravel()
    else:
        shape = [left_modes[0],right_modes[0]]
    
    tensor_column_len  = shape[0]
    tensor_row_len     = shape[1]
    tensor_depth_len   = shape[2]
    tensor_channel_len = shape[3]



    y_out      = np.zeros((L,1), dtype=float, order='F')

    
    #t0 = time.time()
    
    core_mat_0 = core_tensor_0[:,1,:]
    
    ind1 = 1
    print('core_tensor_0.shape', core_tensor_0.shape, 'core_mat_0.dtype', core_tensor_0.dtype, type(core_tensor_0), ind1)
    core_mat_0_bk = matrix_dot.layer7_tensor_to_matrix_slice_0(core_tensor_0,ind1)
    
    

    #print('core_mat_0_bk.shape', core_mat_0_bk.shape, 'core_mat_0_bk.dtype',core_mat_0_bk.dtype)
    #print('np.linalg.norm(core_mat_0_bk-core_mat_0,1)', np.linalg.norm(core_mat_0_bk-core_mat_0,1))
    
    
    
    core_mat_2 = core_tensor_2[:,1,:]
    print('core_tensor_2.shape', core_tensor_2.shape,'core_mat_2.shape',core_mat_2.shape)
    

    core_mat_2_bk = matrix_dot.layer7_tensor_to_matrix_slice_2(core_tensor_2,ind1)
     
    print('core_mat_2_bk.shape', core_mat_2_bk.shape, 'core_mat_1_bk.dtype',core_mat_2_bk.dtype)
    print('np.linalg.norm(core_mat_2_bk-core_mat_2,1)', np.linalg.norm(core_mat_2_bk-core_mat_2,1))
    '''
    core_mat_2 = core_tensor_2[:,1,:]
    
    
    core_mat_1 = core_tensor_1[:,1,:]
    tmp1 = np.dot(core_mat_0, core_mat_1)
    #tmp1_bk = matrix_dot.layer7_core_mat0_mat1_dot(core_mat_0, core_mat_1)
    tmp1_bk = matrix_dot.dot_matrix(core_mat_0, core_mat_1)
    print(np.linalg.norm(tmp1-tmp1_bk,1))        
           
                
    core_mat_2 = core_tensor_2[:,1,:]
    tmp2 = np.dot(tmp1, core_mat_2)
    #tmp2_bk = matrix_dot.layer7_core_tmp1_mat2_dot(tmp1, core_mat_2)
    tmp2_bk = matrix_dot.dot_matrix(tmp1, core_mat_2)
    
    print(tmp1_bk.shape, tmp2_bk.shape, tmp1.shape, tmp2.shape)
    print(np.linalg.norm(tmp2-tmp2_bk,1))        

    
    print('**********************************************************************************')
    
    
    ind1 = np.zeros(tensor_channel_len) + 1
    ind2 = np.zeros(tensor_channel_len) + 2
    ind3 = np.zeros(tensor_channel_len) + 3
    
    ind1_bk = matrix_dot.layer7_create_index(tensor_channel_len,1)
    ind2_bk = matrix_dot.layer7_create_index(tensor_channel_len,2)
    ind3_bk = matrix_dot.layer7_create_index(tensor_channel_len,3)    
    
    print('ind1.shape', ind1.shape ,'ind1_bk.shape' , ind1_bk.shape, 'ind1.dtype' , ind1.dtype
          ,'ind1_bk.dtype', ind1_bk.dtype , 'np.linalg.norm(ind1-ind1_bk,1)',np.linalg.norm(ind1-ind1_bk,1))
    

    ind4 = np.arange(tensor_channel_len)
    ind4_bk = matrix_dot.layer7_one_d_arrange(tensor_channel_len)
    
    
    print('ind4.shape', ind4.shape ,'ind4_bk.shape' , ind4_bk.shape, 'ind4.dtype' , ind4.dtype
          ,'ind4_bk.dtype', ind4_bk.dtype , 'np.linalg.norm(ind4-ind4_bk,1)',np.linalg.norm(ind4-ind4_bk,1))
    
    
    indy = np.ravel_multi_index([ind1.astype('int64'),ind2.astype('int64'),ind3.astype('int64'),ind4.astype('int64')],
                              (tensor_row_len, tensor_column_len, tensor_depth_len, tensor_channel_len), order='F')   
    

    
    
    
    indy_bk = matrix_dot.layer7_four_d_ravel_multi_index( ind1.astype('int64'),ind2.astype('int64'),ind3.astype('int64'),
                                                          ind4.astype('int64'),tensor_row_len, tensor_column_len, 
                                                          tensor_depth_len, tensor_channel_len
                                                        )
    
    indy = np.reshape(indy,[256,1])
    print('np.linalg.norm(indy-indy_bk,1)',np.linalg.norm(indy-indy_bk,1), 
          'indy.shape', indy.shape, 'indy_bk.shape' , indy_bk.shape)
    
    print(indy[1:5], indy_bk[1:5])
    row_index = np.mod(indy, column_len)
    #print('row_index', row_index.shape)
    
    
    row_index_bk = matrix_dot.layer7_mod(indy_bk, column_len)
    #print('row_index_bk', row_index_bk.shape)
    
    
    #print('indy / column_len', np.floor( indy / column_len).astype('int64'))
    
    column_index = np.floor( indy / column_len).astype('int64')
    column_index_bk = matrix_dot.layer7_floor_array(indy / column_len).astype('int64')
    
    print('column_index', column_index[1:5], 'column_index_bk', column_index_bk[1:5])
    
    
    
    
    tmp3 = np.dot(tmp2, core_tensor_3)
    tmp3_bk = matrix_dot.dot_matrix(tmp2, core_tensor_3)
    
    print('tmp3', tmp3[0,1:5], 'tmp3_bk', tmp3_bk[0,1:5], tmp3_bk.shape, tmp3.shape)
    
    
    x = feature_x[column_index].reshape([1,tensor_channel_len],order = 'F')
    x_bk = matrix_dot.layer7_Get_feature_x_by_index(feature_x, column_index)
    print('np.linalg.norm(x-x_bk,1)', np.linalg.norm(x-x_bk,1), x_bk.shape)
    
    
    
    
    tmp4 = np.multiply(tmp3,feature_x[[column_index]].reshape([1,tensor_channel_len],order = 'F'))
    tmp4_bk = matrix_dot.layer7_multiply_matrix_multiply_element_wise(tmp3, x_bk)
    print('np.linalg.norm(tmp4-tmp4_bk,1)', np.linalg.norm(tmp4-tmp4_bk,1), tmp4_bk.shape)
    
    


                for l in range(0,tensor_channel_len):
                    y_out[row_index[l]] = y_out[row_index[l]] + tmp4[0,l]
'''
    
    
    
	


    
def	Relu_Function(x):
    out = np.maximum(x, 0)
    return out

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def haarMatrix(n, normalized=1):
    # Allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haarMatrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part 
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h

'''
    ####reconstruct weight matrix
	
    a=core_arr[ps_arr[0]-1:ps_arr[1]-1]

    for i in range(1,dim_arr):
        cr=core_arr[ps_arr[i]-1:ps_arr[i+1]-1]
        print('i:',i)

        cr=np.reshape(cr,[ranks[i],n_arr[i]*ranks[i+1]], order="F")
        print('cr.shape',cr.shape)
        a=np.reshape(a,[-1,ranks[i]], order="F")
        #print('a.shape,a[0:3,0:3]',a.shape,a[0:3,0:3])
        a=np.dot(a,cr)
        #print('a.shape,a[0:3,0:3]',a.shape,a[0:3,0:3])

    weight_mat = np.reshape(a, [L, R], order="F");
    ####reconstruct weight matrix
'''
if __name__ == '__main__':
    filename = '/home/wade/Document2/TT-haar-devolope/weights/mnist_y_out_layer7.mat'
    feature_x = np.random.randn(4096,1) ;
    tt_construct_layer7(filename, feature_x)		
    