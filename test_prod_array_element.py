import numpy as np
import example
import os
import scipy.io



filename   = '/home/wade/Documents/TT-haar-devolope/weights/mnist_y_out_layer6.mat'

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


L_cpp      = example.multiply(left_modes)
L          = np.prod(left_modes)

R_cpp      = example.multiply(right_modes)
R          = np.prod(right_modes)

print(L_cpp)
print(R_cpp)








