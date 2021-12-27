import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b  = np.array([a, a * 2])
c = np.array([0.1, 2.0]).reshape(1,2)
print(a)
print(b)
print(c)

print(np.dot(c,b.reshape(2,-1)).reshape(3,3))
