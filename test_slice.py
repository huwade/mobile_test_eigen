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



x = np.arange(16,dtype=float).reshape(1,16)
y = np.arange(256,dtype=float).reshape(16,16)


print(x.dtype, type(x), x.shape, y.shape)
print('example.det(A) = \n'      , matrix_dot.det(y))


b = matrix_dot.layer7_core_mat0_mat1_dot(x, y)
d = matrix_dot.layer7_create_index(5,1)

c = np.dot(x,y)


print('********************')
print( b.shape, c.shape, np.linalg.norm(b-c,2), d)
e = matrix_dot.layer7_one_d_arrange(5)
print(e,e.dtype, type(e))

