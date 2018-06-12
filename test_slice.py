import numpy as np
import eigen_slice

'''
x = np.arange(256.0).reshape(16,16)
#np.arange(10).reshape(5,2)
x = np.reshape(x,[1,16,16])
print(x.shape)



a = eigen_slice.hello(x)


print(a)
'''
import matrix_dot



x = np.arange(8,dtype=float).reshape(2,4)
y = np.arange(8,dtype=float).reshape(4,2)
b = np.zeros((2,2), dtype=float, order='F')

print(x.dtype, type(x), x.shape, y.shape, b)
b = matrix_dot.layer7_core_mat0_mat1_dot(x, y)


print('********************')
print(b, type(b), b.shape)