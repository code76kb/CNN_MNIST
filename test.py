import numpy as np


a = np.random.uniform(size=(2,3))
b = np.random.normal(size=(3,4))

print 'a :',a, '\nb:',b


c = np.dot(a,b)
d = a.dot(b)

print '\n c:',c,' \nd:',d
