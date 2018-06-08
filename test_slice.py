import numpy as np
import eigen_slice


x = np.arange(256.0).reshape(16,16)
#np.arange(10).reshape(5,2)
x = np.reshape(x,[1,16,16])
print(x.shape)



a = eigen_slice.hello(x)


print(a)



