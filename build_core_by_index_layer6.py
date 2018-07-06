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
import layer6_create





def tt_construct_layer6(filename, feature_x):
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

    L = np.prod(left_modes)
    R = np.prod(right_modes)

    L_cpp      = layer6_create.multiply_elements(left_modes)
    R_cpp      = layer6_create.multiply_elements(right_modes)
    
    print('L_cpp - L', L_cpp - L, type(L), type(L_cpp))

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
    y_out_bk   = layer6_create.create_zero_array(L_cpp)
    #feature_x  = np.random.randn(R,1) ;

    core_mat_0 = np.reshape(core_tensor_0[0,:,:],[4096, ranks[1]],order = 'F')
    print('core_tensor_0[0,:,:].shape', 'core_tensor_0.shape', core_tensor_0[0,:,:].shape, core_tensor_0.shape)
'''
    core_mat_1 = core_tensor_1
    
 
    w = np.dot(core_mat_0, core_mat_1)
    y_out  = np.dot(w, feature_x)

    y_out = y_out + bias
    return y_out
'''
    
if __name__ == '__main__':
    filename = '/home/wade/Document2/TT-haar-devolope/weights/mnist_y_out_layer6.mat'
    feature_x = np.random.randn(3200,1) ;
    tt_construct_layer6(filename, feature_x)		



