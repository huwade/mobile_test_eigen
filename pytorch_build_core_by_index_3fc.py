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





def tt_construct_mobile(filename, feature_x):
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

    
    L_cpp = multiply_element.multiply(left_modes)
    L     = np.prod(left_modes)
    
    
    R_cpp = multiply_element.multiply(right_modes)
    R     = np.prod(right_modes)

    
    print('L_cpp-L',L_cpp-L)
    print('R_cpp-R',R_cpp-R)

    core_tensor_0 = a['tensor'][0][0][0,0]
    core_tensor_1 = a['tensor'][0][0][0,1]

    column_len   = L
    row_len      = R


    if dim_arr > 2:
        shape = left_modes*right_modes.ravel()
    else:
        shape = [left_modes[0],right_modes[0]]


    tensor_column_len  = shape[0]
    tensor_row_len     = shape[1]


    y_out      = np.zeros((L,1), dtype=float, order='F')
    y_out_bk   = matrix_dot.create_zero_array(L)

    #feature_x  = np.random.randn(R,1) ;

    core_mat_0 = core_tensor_0[0,:,:]
    core_mat_1 = core_tensor_1
    
    print('core_mat_0.shape, core_mat_1.shape, feature_x.shape',core_mat_0.shape, core_mat_1.shape, feature_x.shape)
    
    tmp    = np.dot(core_mat_1, feature_x)
    tmp_bk = matrix_dot.dot_matrix(core_mat_1, feature_x)
    
    print('tmp-tmp_bk', np.linalg.norm(tmp-tmp_bk,1))
    
    
    
    y_out    = np.dot(core_mat_0, tmp)
    y_out_bk = matrix_dot.dot_matrix(core_mat_0, tmp)

    print('y_out-y_out_bk', np.linalg.norm(y_out-y_out_bk,1))
    
    print('bias.shape',bias.shape, 'y_out.shape',y_out.shape)
    
    y_out = y_out + bias
    return y_out

def	Relu_Function(x):
    out = np.maximum(x, 0)
    out_bk = matrix_dot.activation_function_ReLU(x)
    print('out-out_bk', np.linalg.norm(out-out_bk,1))

    return out

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

    

if __name__ == '__main__':
    filename = '/home/wade/Document2/experiment/TT-haar-devolope/weights/mobile_net_fc1_out.mat'
    feature_x = np.random.randn(784,1) ;
    y = tt_construct_mobile(filename, feature_x)		
    
    Relu_Function(y)


