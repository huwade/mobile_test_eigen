import numpy as np
import eigen_reshape


x = np.array([[1, 2, 3, 4, 5, 6], [7,8,9,10,11,12]])
#np.arange(10).reshape(5,2)
x = np.reshape(x,[2,3,2])
print(x.shape)



a = eigen_reshape.test_passed_by_reference(x)
print(x)
print(x[:,:,0])
print(x[:,:,1])
print('-----------')
print(np.reshape(x,[4,3]))
print('-----------')
print(np.reshape(x,[4,3],  order='F'))
print('-----------')
print(a)



