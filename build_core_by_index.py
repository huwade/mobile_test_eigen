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



@jit
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
    #feature_x  = np.random.randn(R,1) ;

    core_mat_0 = np.reshape(core_tensor_0[0,:,:],[4096, ranks[1]],order = 'F')
    core_mat_1 = core_tensor_1
    
    w = np.dot(core_mat_0, core_mat_1)
    y_out  = np.dot(w, feature_x)

    y_out = y_out + bias
    return y_out

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

    L = np.prod(left_modes)
    R = np.prod(right_modes)


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
    for i in range(0,tensor_row_len):
        core_mat_0 = np.reshape(core_tensor_0[:,i,:],[ranks[0], ranks[1]],order = 'F')
        for j in range(0,tensor_column_len):
            core_mat_1 = np.reshape(core_tensor_1[:,j,:],[ranks[1], ranks[2]],order = 'F')
            tmp1 = np.dot(core_mat_0, core_mat_1)
            for k in range(0,tensor_depth_len):
                
                core_mat_2 = np.reshape(core_tensor_2[:,k,:],[ranks[2], ranks[3]],order = 'F')
                tmp2 = np.dot(tmp1, core_mat_2)

                ind1 = np.zeros(tensor_channel_len) + i
                ind2 = np.zeros(tensor_channel_len) + j
                ind3 = np.zeros(tensor_channel_len) + k
                ind4 = np.arange(tensor_channel_len)
                 
                indy = np.ravel_multi_index([ind1.astype('int64'),ind2.astype('int64'),ind3.astype('int64'),ind4.astype('int64')],(tensor_row_len, tensor_column_len, tensor_depth_len, tensor_channel_len), order='F')

                row_index = np.mod(indy, column_len)

                column_index = np.floor( indy / column_len).astype('int64')

                tmp3 = np.dot(tmp2,core_tensor_3.reshape([ranks[3], tensor_channel_len],order = 'F')  )


                tmp4 = np.multiply(tmp3,feature_x[[column_index]].reshape([1,tensor_channel_len],order = 'F'))
                for l in range(0,tensor_channel_len):
                    y_out[row_index[l]] = y_out[row_index[l]] + tmp4[0,l]


    
    #t1 = time.time()
    #print(y_out)
    #frameinfo = getframeinfo(currentframe())
    #print(frameinfo.filename, frameinfo.lineno)
    #print(t1-t0)
    y_out = y_out + bias
    return y_out


#if __name__ == '__main__':
#    filename = '/home/wade/Documents/TT-haar-devolope/weights/mnist_weights_2_23.mat'
#    feature_x = np.random.randn(4096,1) ;
#    test_tt_construct(filename, feature_x)		
	


@jit
def tt_construct_layer8(filename, feature_x):
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
    #feature_x  = np.random.randn(R,1) ;

    t0 = time.time()
    for row_index in range(0, L):
        for column_index in range(0, R):

            core_mat_0 = np.reshape(core_tensor_0[:,row_index,:],[ranks[0], ranks[1]],order = 'F')
            core_mat_1 = np.reshape(core_tensor_1[:,column_index],[ranks[1], ranks[2]],order = 'F')


            tmp = np.dot(core_mat_0, core_mat_1)
            tmp = np.dot(tmp, feature_x[column_index]) 

            y_out[row_index][0] = y_out[row_index][0] + tmp; 
            tmp = 0


    
    t1 = time.time()
    #print(y_out)
    frameinfo = getframeinfo(currentframe())
    

    y_out = y_out + bias
    return y_out

    
    
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

    